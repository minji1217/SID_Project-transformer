"""Priority 2 — SID Direct Scorer.

기존 V1은 decoder의 생성확률을 더해 candidate를 매겼다.

    score = log p(c1|H) + log p(c2|H,c1) + log p(c3|H,c1,c2)

Priority 2는 생성확률을 ranking score로 쓰지 않는다. history를 user
vector u로, candidate SID를 candidate vector v로 만들고 둘의 적합도를
직접 학습한다.

    u  = attention_pool(article vectors)
    vi = Linear(concat(c1_emb, c2_emb, c3_emb))
    Di = u^T W vi                       (P2-A Bilinear)

기존 NewsEncoderDecoderTransformer를 수정하지 않는다. embedding과
encoder만 빌려 쓰고, decoder / BOS / c1c2c3 head는 forward에서 건드리지
않는다.

P2-A는 V1 checkpoint를 쓰지 않는다. random initialization에서
from-scratch로 학습한다. V1과 공정하게 비교하기 위해서다.

backbone(embedding + encoder)은 V1과 같은 생성 순서를 거쳐야 같은
seed에서 같은 초기값이 나온다. 그래서 decoder까지 포함한 전체
NewsEncoderDecoderTransformer를 그대로 만든 뒤, 새 module을 그 다음에
만든다. decoder를 만들지 않으면 그 뒤의 _reset_parameters()가 보는
RNG 상태가 달라져 embedding 초기값이 V1과 어긋난다.

alpha1 / alpha2 / alpha3, L1 / L2 / L3, weighted score는 여기에 없다.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import gin
import torch

from torch import Tensor, nn

from modules.model import NewsEncoderDecoderTransformer


NUM_CANDIDATE_SID_LEVELS = 3
NUM_CANDIDATES = 5

# history 기사 하나가 차지하는 code token 수 (use_sep=False 기준)
TOKENS_PER_ARTICLE = 4

SCORER_TYPES = ("bilinear", "mlp")

# checkpoint에는 있지만 Priority 2 forward에서 쓰지 않는 parameter.
# requires_grad=False로 두고 optimizer에서도 뺀다.
UNUSED_BACKBONE_PREFIXES = (
    "decoder.",
    "decoder_dummy_embedding.",
    "bos_embedding",
    "c1_head.",
    "c2_head.",
    "c3_head.",
)

# backbone 중 Priority 2가 실제로 학습하는 부분.
BACKBONE_TRAINABLE_PREFIXES = (
    "c1_embedding.",
    "c2_embedding.",
    "c3_embedding.",
    "c4_embedding.",
    "encoder.",
    "encoder_dummy_embedding.",
)


class DirectScoreOutput(NamedTuple):
    """train_transformer.update_metrics가 쓰는 필드 이름을 맞춘다."""
    candidate_scores: Tensor      # [B, 5]
    user_vector: Tensor           # [B, d_model]
    candidate_vectors: Tensor     # [B, 5, d_model]
    attention_weights: Tensor     # [B, H]


def masked_attention_pool(
    article_vectors: Tensor,
    article_mask: Tensor,
    proj: nn.Linear,
    score: nn.Linear,
) -> Tuple[Tensor, Tensor]:
    """learnable attention pooling. padding 기사는 제외한다.

    article_vectors: [B, H, d]
    article_mask:    [B, H]  (True = 실제 기사)

    H는 batch마다 다르다. collate가 그 batch의 최대 history 길이로
    padding하기 때문이다. 50이나 200을 가정하지 않는다.
    """
    logits = score(torch.tanh(proj(article_vectors))).squeeze(-1)   # [B, H]

    mask = article_mask.to(torch.bool)

    # padding 위치를 softmax에서 완전히 뺀다.
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    weights = torch.softmax(logits, dim=1)

    # history가 아예 비어 있는 행이 있으면 softmax가 NaN이 된다.
    # dataset이 drop_empty_history=True로 걸러 주지만 방어해 둔다.
    has_any = mask.any(dim=1, keepdim=True)
    weights = torch.where(has_any, weights, torch.zeros_like(weights))

    user_vector = torch.einsum("bh,bhd->bd", weights, article_vectors)

    return user_vector, weights


@gin.configurable
class DirectScorer(nn.Module):
    """from-scratch로 학습하는 direct relevance scorer."""

    def __init__(
        self,
        backbone: NewsEncoderDecoderTransformer,
        scorer_type: str = "bilinear",
        mlp_hidden: int = 256,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if scorer_type not in SCORER_TYPES:
            raise ValueError(
                f"scorer_type은 {SCORER_TYPES} 중 하나여야 합니다: {scorer_type}"
            )

        if backbone.use_sep:
            raise ValueError(
                "use_sep=True는 지원하지 않습니다. article vector를 만들 때 "
                "c1/c2/c3/c4 4개 토큰만 평균하도록 되어 있어 SEP 토큰이 "
                "섞이면 안 됩니다. V1 final config는 use_sep=False입니다."
            )

        self.backbone = backbone
        self.scorer_type = scorer_type
        self.d_model = backbone.d_model

        d = self.d_model

        # history article 단위 attention pooling
        self.pool_proj = nn.Linear(d, d)
        self.pool_score = nn.Linear(d, 1, bias=False)

        # candidate c1/c2/c3 embedding concat -> candidate vector
        self.candidate_projection = nn.Linear(
            NUM_CANDIDATE_SID_LEVELS * d, d
        )

        if scorer_type == "bilinear":
            # D = u^T W v.  standard initialization을 쓴다.
            # identity 초기화는 이번 실험에서 쓰지 않는다.
            self.bilinear = nn.Linear(d, d, bias=False)
            self.mlp = None
        else:
            # [u, v, u*v, |u-v|] -> scalar
            self.bilinear = None
            self.mlp = nn.Sequential(
                nn.Linear(4 * d, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden, 1),
            )

        self.freeze_unused_backbone()

    # ------------------------------------------------------------ parameters

    def freeze_unused_backbone(self) -> List[str]:
        """forward에서 안 쓰는 decoder 계열을 학습에서 뺀다."""
        frozen: List[str] = []

        for name, param in self.backbone.named_parameters():
            if name.startswith(UNUSED_BACKBONE_PREFIXES):
                param.requires_grad = False
                frozen.append(name)

        return frozen

    def backbone_trainable_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [
            (f"backbone.{name}", param)
            for name, param in self.backbone.named_parameters()
            if name.startswith(BACKBONE_TRAINABLE_PREFIXES)
        ]

    def new_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [
            (name, param)
            for name, param in self.named_parameters()
            if not name.startswith("backbone.")
        ]

    def unused_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [
            (f"backbone.{name}", param)
            for name, param in self.backbone.named_parameters()
            if name.startswith(UNUSED_BACKBONE_PREFIXES)
        ]

    # ------------------------------------------------------------ forward

    def encode_user(
        self, history_sids: Tensor, history_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        encoder_output = self.backbone.encode(
            history_sids=history_sids, history_mask=history_mask
        )

        hidden = encoder_output.hidden_states          # [B, H*4, d]
        batch_size, seq_len, d_model = hidden.shape

        if seq_len % TOKENS_PER_ARTICLE != 0:
            raise ValueError(
                f"encoder 출력 길이 {seq_len}이 "
                f"{TOKENS_PER_ARTICLE}의 배수가 아닙니다."
            )

        history_length = seq_len // TOKENS_PER_ARTICLE

        if history_length != history_mask.shape[1]:
            raise ValueError(
                f"history 길이가 맞지 않습니다: encoder {history_length} vs "
                f"mask {history_mask.shape[1]}"
            )

        # [B, H*4, d] -> [B, H, 4, d] -> c1/c2/c3/c4 평균 -> [B, H, d]
        article_vectors = hidden.view(
            batch_size, history_length, TOKENS_PER_ARTICLE, d_model
        ).mean(dim=2)

        return masked_attention_pool(
            article_vectors, history_mask, self.pool_proj, self.pool_score
        )

    def encode_candidates(self, candidate_sids: Tensor) -> Tensor:
        if candidate_sids.ndim != 3:
            raise ValueError(
                "candidate_sids must have shape [B,5,3]. "
                f"Received: {tuple(candidate_sids.shape)}"
            )

        if candidate_sids.shape[-1] != NUM_CANDIDATE_SID_LEVELS:
            raise ValueError("candidate_sids must contain exactly c1,c2,c3.")

        if candidate_sids.shape[1] != NUM_CANDIDATES:
            raise ValueError(
                f"Expected {NUM_CANDIDATES} candidates, "
                f"but received {candidate_sids.shape[1]}."
            )

        # candidate c4는 쓰지 않는다.
        c1 = self.backbone.c1_embedding(candidate_sids[:, :, 0])
        c2 = self.backbone.c2_embedding(candidate_sids[:, :, 1])
        c3 = self.backbone.c3_embedding(candidate_sids[:, :, 2])

        # [B, 5, 3d] -> [B, 5, d]
        return self.candidate_projection(torch.cat([c1, c2, c3], dim=-1))

    def score(self, user_vector: Tensor, candidate_vectors: Tensor) -> Tensor:
        if self.scorer_type == "bilinear":
            # D_i = u^T W v_i
            transformed = self.bilinear(candidate_vectors)        # [B, 5, d]
            return torch.einsum("bd,bcd->bc", user_vector, transformed)

        # mlp: [u, v, u*v, |u-v|]
        expanded = user_vector.unsqueeze(1).expand_as(candidate_vectors)

        features = torch.cat(
            [
                expanded,
                candidate_vectors,
                expanded * candidate_vectors,
                torch.abs(expanded - candidate_vectors),
            ],
            dim=-1,
        )

        return self.mlp(features).squeeze(-1)

    def forward(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
        candidate_sids: Tensor,
    ) -> DirectScoreOutput:
        user_vector, attention_weights = self.encode_user(
            history_sids, history_mask
        )

        candidate_vectors = self.encode_candidates(candidate_sids)
        candidate_scores = self.score(user_vector, candidate_vectors)

        return DirectScoreOutput(
            candidate_scores=candidate_scores,
            user_vector=user_vector,
            candidate_vectors=candidate_vectors,
            attention_weights=attention_weights,
        )

    # ------------------------------------------------------------ 보고용

    def parameter_report(self) -> Dict[str, object]:
        def total(pairs) -> int:
            return sum(param.numel() for _, param in pairs)

        backbone_trainable = self.backbone_trainable_parameters()
        new = self.new_parameters()
        unused = self.unused_parameters()

        return {
            "backbone_trainable": {
                "count": total(backbone_trainable),
                "tensors": len(backbone_trainable),
                "names": [name for name, _ in backbone_trainable],
            },
            "new_modules": {
                "count": total(new),
                "tensors": len(new),
                "names": [name for name, _ in new],
            },
            "unused_frozen": {
                "count": total(unused),
                "tensors": len(unused),
                "names": [name for name, _ in unused],
            },
            "trainable_total": total(backbone_trainable) + total(new),
            "model_total": sum(p.numel() for p in self.parameters()),
        }
