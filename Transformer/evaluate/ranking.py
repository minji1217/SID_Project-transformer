"""
학습 루프에서 batch 단위로 계산하는 ranking metric.

evaluate/metrics.py는 impression마다 Python 루프를 돌기 때문에
매 epoch validation에 쓰기에는 느리다.
이 모듈은 같은 정의를 tensor 연산으로 벡터화한 것이며,
두 구현의 결과가 일치하는지는 evaluate/test_ranking.py가 검증한다.

동점 처리는 metrics.py의 동작을 그대로 따른다.
  - AUC   : auc_from_scores()와 같이 동점을 0.5점으로 계산
  - rank  : make_ranks()의 np.argsort(-scores, kind="stable")와 같이
            동점이면 candidate index가 작은 쪽이 상위 rank
  - Top-1 : metrics.py의 np.argmax(첫 번째 최댓값)와 같은 결과가 되도록
            rank == 1 로 판정
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from torch import Tensor


class RankingMetricOutput(NamedTuple):
    # 모두 impression 단위 값 [B]
    rank: Tensor
    top1: Tensor
    auc: Tensor
    mrr: Tensor
    ndcg: Tensor


@torch.no_grad()
def compute_ranking_metrics(
    candidate_scores: Tensor,
    candidate_labels: Tensor,
    k: int = 5,
) -> RankingMetricOutput:
    # candidate_scores: [B, C]
    # candidate_labels: [B, C], 각 행에 positive가 정확히 1개
    if candidate_scores.ndim != 2:
        raise ValueError(
            "candidate_scores must have shape [B,C]. "
            f"Received: {tuple(candidate_scores.shape)}"
        )

    if candidate_labels.shape != candidate_scores.shape:
        raise ValueError(
            "candidate_labels shape must match candidate_scores shape."
        )

    scores = candidate_scores.detach().float()
    labels = candidate_labels.detach()

    batch_size, num_candidates = scores.shape

    if num_candidates < 2:
        raise ValueError(
            "Ranking metrics require at least 2 candidates. "
            f"Received: {num_candidates}"
        )

    # 각 행의 positive 위치와 그 score
    positive_index = labels.argmax(dim=1)
    positive_scores = scores.gather(
        dim=1,
        index=positive_index.unsqueeze(1),
    )

    column_index = (
        torch.arange(num_candidates, device=scores.device)
        .unsqueeze(0)
        .expand(batch_size, num_candidates)
    )

    is_negative = column_index != positive_index.unsqueeze(1)

    # positive와 각 negative의 대소 관계
    negative_lower = (scores < positive_scores) & is_negative
    negative_equal = (scores == positive_scores) & is_negative
    negative_greater = (scores > positive_scores) & is_negative

    num_negative = num_candidates - 1

    # AUC: positive가 negative보다 위에 오는 비율 (동점은 0.5점)
    auc = (
        negative_lower.sum(dim=1).float()
        + 0.5 * negative_equal.sum(dim=1).float()
    ) / num_negative

    # rank: 동점이면 index가 작은 후보가 상위
    ranked_above = negative_greater | (
        negative_equal & (column_index < positive_index.unsqueeze(1))
    )
    rank = 1 + ranked_above.sum(dim=1)

    rank_float = rank.float()

    # MRR: 정답의 reciprocal rank
    mrr = 1.0 / rank_float

    # nDCG@k: positive가 1개이므로 IDCG = 1
    # rank가 k를 넘으면 0
    ndcg = torch.where(
        rank <= k,
        1.0 / torch.log2(rank_float + 1.0),
        torch.zeros_like(rank_float),
    )

    # Top-1: rank == 1
    top1 = (rank == 1).float()

    return RankingMetricOutput(
        rank=rank,
        top1=top1,
        auc=auc,
        mrr=mrr,
        ndcg=ndcg,
    )
