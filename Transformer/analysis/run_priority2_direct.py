"""Priority 2 — SID Direct Scorer 학습 (P2-A Bilinear).

생성확률 합산(L1+L2+L3)을 ranking score로 쓰지 않고, history user
vector와 candidate SID vector의 직접 적합도를 학습한다.

    D_i = u^T W v_i

loss는 기존 5-candidate listwise CE를 그대로 쓴다. 바꾸는 것은 score
함수이지 loss가 아니다.

V1 checkpoint를 쓰지 않는다. random initialization에서 from-scratch로
학습한다. 학습 조건을 V1 seed42 final과 같게 맞춰서, 달라지는 것이
score 구조 하나만 남게 한다.

    V1 : Decoder -> L1 + L2 + L3
    P2 : article pooling + candidate projection -> u^T W v

alpha 가중치, L1/L2/L3 결합, hybrid, overlap feature, candidate c4,
hard-negative loss, Test 사용은 전부 없다.

    cd Transformer
    python -m analysis.run_priority2_direct \
        --priority0-dir sweep_out/ebnerd_v2/priority0_shuffled/seed42 \
        --out sweep_out/ebnerd_v2/priority2_direct/bilinear/seed42
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gin
import numpy as np
import pandas as pd
import torch

from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.sequence import NewsSequenceDataset, collate_news_sequences
from evaluate.metrics import evaluate_ranking
from modules.direct_scorer import BACKBONE_TRAINABLE_PREFIXES, DirectScorer
from modules.loss import TransformerLoss
from modules.model import NewsEncoderDecoderTransformer
from sweep.run_stage import format_binding, write_csv
from train_transformer import (
    create_metric_dict,
    finalize_metrics,
    set_seed,
    update_metrics,
)


BASE_DIR = Path(__file__).resolve().parent.parent

NUM_CANDIDATES = 5
GROUP_COLUMN = "sample_index"

# V1 final config. Priority 0가 기록한 값이 이것과 다르면 실행을 멈춘다.
EXPECTED_CONFIG: Dict[str, Any] = {
    "NewsSequenceDataset.max_history_length": 50,
    "NewsEncoderDecoderTransformer.use_sep": False,
    "NewsEncoderDecoderTransformer.d_model": 256,
    "NewsEncoderDecoderTransformer.num_heads": 8,
    "NewsEncoderDecoderTransformer.num_layers": 2,
    "NewsEncoderDecoderTransformer.d_ff": 1024,
    # dropout_rate는 --dropout으로 정한다. 기본 0.0이 V1 final config다.
}

DROPOUT_KEY = "NewsEncoderDecoderTransformer.dropout_rate"

# 비교표. 29.312%는 validation에서 고른 참고값이지 합격선이 아니다.
BASELINES: List[Dict[str, Any]] = [
    {
        "config": "Original V1",
        "detail": "L1 + L2 + L3",
        "tuned_on": "Train",
        "top1_accuracy": 0.27059,
        "note": "생성확률 동일 가중 합산",
    },
    {
        "config": "Learnable alpha",
        "detail": "a1*L1 + a2*L2 + a3*L3 (Train에서 학습)",
        "tuned_on": "Train",
        "top1_accuracy": 0.27166,
        "note": "alpha를 Train에서 학습",
    },
    {
        "config": "P2-A c1234 mean",
        "detail": "direct u^T W v, history = c1c2c3c4, 단순 평균",
        "tuned_on": "Train",
        "top1_accuracy": 0.26137,
        "note": "c4를 encoder token으로 넣은 경우",
    },
    {
        "config": "P2-A c123 mean",
        "detail": "direct u^T W v, history = c1c2c3, 단순 평균",
        "tuned_on": "Train",
        "top1_accuracy": 0.26650,
        "note": "이번 실험의 직접 비교 대상",
    },
    {
        "config": "Fixed weighted grid",
        "detail": "L1 + 0.5*L2 + 0.1*L3",
        "tuned_on": "Validation",
        "top1_accuracy": 0.29312,
        "note": "validation-tuned weighted reference (합격선 아님)",
    },
]


def section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def resolve_path(path: str) -> Path:
    path_obj = Path(path).expanduser()
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}초"
    if seconds < 3600:
        return f"{seconds / 60:.1f}분"
    return f"{seconds / 3600:.1f}시간"


# ---------------------------------------------------------------- 설정


def bindings_from_priority0(
    p0_meta: Dict[str, Any], extra: Optional[List[str]] = None
) -> Tuple[List[str], str]:
    """Priority 0가 쓴 gin override를 되살린다.

    Priority 0는 selected_final.json 경로면 "config" 딕셔너리에,
    --checkpoint 경로면 "gin_bindings" 목록에 기록한다.
    """
    if extra:
        return list(extra), "명령행 --gin-binding"

    binding_strings = p0_meta.get("gin_bindings")

    if binding_strings:
        return list(binding_strings), "Priority 0 metadata의 gin_bindings"

    config = p0_meta.get("config")

    if config:
        return (
            [format_binding(key, config[key]) for key in sorted(config)],
            "Priority 0 metadata의 config",
        )

    return [], "없음 (base config만 사용)"


def configure_gin(config_path: Path, bindings: List[str]) -> None:
    """gin은 Dataset보다 먼저 읽어야 max_history_length가 적용된다."""
    gin.parse_config_file(str(config_path), skip_unknown=True)

    if bindings:
        gin.parse_config(bindings, skip_unknown=True)


def verify_config(expected_dropout: float = 0.0) -> Dict[str, Any]:
    """유효 gin 값이 V1 final config와 같은지 본다.

    다르면 warning이 아니라 실행 중단이다. 설정이 어긋난 채로 학습하면
    checkpoint load가 통과해도 결과를 V1과 비교할 수 없다.
    """
    section("config 확인 — V1 final config와 일치하는가")

    rows: List[Dict[str, Any]] = []
    mismatched: List[str] = []

    expected_config = dict(EXPECTED_CONFIG)
    expected_config[DROPOUT_KEY] = float(expected_dropout)

    print(f"  {'parameter':<52} {'기대값':>10} {'실제값':>10}")
    print("  " + "-" * 76)

    for key, expected in expected_config.items():
        try:
            actual: Any = gin.query_parameter(key)
        except (ValueError, KeyError, TypeError):
            actual = "<미설정>"

        ok = actual == expected

        # bool은 1/0과 같다고 판정되지 않도록 타입까지 본다.
        if isinstance(expected, bool):
            ok = isinstance(actual, bool) and actual == expected
        elif isinstance(expected, float) and isinstance(actual, (int, float)):
            ok = float(actual) == float(expected)

        rows.append({
            "parameter": key,
            "expected": expected,
            "actual": actual,
            "match": bool(ok),
        })

        if not ok:
            mismatched.append(key)

        print(
            f"  {key:<52} {str(expected):>10} {str(actual):>10}  "
            f"{'OK' if ok else 'MISMATCH'}"
        )

    print()

    if mismatched:
        print("  => V1 final config와 다릅니다. 실행을 중단합니다.")
        print("     Priority 0 metadata의 설정이 이번 실험 조건과 다릅니다.")
        for key in mismatched:
            print(f"       {key}")
    else:
        print("  => 전부 일치합니다.")

    return {"rows": rows, "all_match": not mismatched, "mismatched": mismatched}


# ---------------------------------------------------------------- 모델


def build_backbone(device: torch.device) -> NewsEncoderDecoderTransformer:
    """V1과 같은 생성 순서로 backbone을 만든다.

    decoder / BOS / c1c2c3 head는 Priority 2 forward에서 쓰지 않지만
    그래도 만든다. 만들지 않으면 그 뒤의 _reset_parameters()가 보는
    RNG 상태가 달라져서, 공유하는 embedding의 초기값이 V1과 어긋난다.
    만들어 둔 뒤 requires_grad=False로 두고 optimizer에서도 뺀다.
    """
    return NewsEncoderDecoderTransformer().to(device)


def snapshot(model: nn.Module, prefixes: Tuple[str, ...]) -> Dict[str, Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.startswith(prefixes)
    }


def max_deviation(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> Optional[float]:
    if set(a) != set(b):
        return None

    return max(
        (float((a[name] - b[name]).abs().max().item()) for name in a),
        default=0.0,
    )


def verify_initialization(
    model: DirectScorer,
    before: Dict[str, Tensor],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """공유 parameter가 V1과 같은 초기값에서 출발하는지 본다.

    두 가지를 확인한다.

    1. DirectScorer가 새 module을 만들면서 backbone 초기값을 건드리지
       않았는가. 새 module이 backbone보다 먼저 만들어지면 RNG가 밀려서
       embedding/encoder 초기값이 V1과 달라진다.

    2. dataset / dataloader 생성이 RNG를 소비하는가. V1은
       set_seed -> Dataset -> DataLoader -> Model 순서라 이 스크립트도
       같은 순서를 따른다. 소비하지 않는다면 순서는 무관하다.
    """
    section("초기화 확인 — V1과 같은 random init에서 출발하는가")

    result: Dict[str, Any] = {"checks": []}

    def record(name: str, detail: str, passed: bool, fatal: bool) -> None:
        result["checks"].append({
            "name": name, "detail": detail,
            "passed": passed, "fatal": fatal,
        })
        mark = "OK  " if passed else ("FAIL" if fatal else "참고")
        print(f"  {mark}  {name:<44} {detail}")

    prefixes = tuple(
        f"backbone.{prefix}" for prefix in BACKBONE_TRAINABLE_PREFIXES
    )

    after = snapshot(model, prefixes)
    deviation = max_deviation(before, after)

    record(
        "DirectScorer 생성이 backbone 초기값을 보존",
        ("parameter 집합이 다릅니다" if deviation is None
         else f"최대편차 {deviation:.3e}"),
        deviation == 0.0,
        True,
    )

    # dataset/loader 생성이 RNG를 소비하는지 확인한다.
    set_seed(seed)
    reference = NewsEncoderDecoderTransformer().to(device)

    reference_snapshot = {
        f"backbone.{name}": param.detach().clone()
        for name, param in reference.named_parameters()
        if name.startswith(BACKBONE_TRAINABLE_PREFIXES)
    }

    order_deviation = max_deviation(reference_snapshot, after)

    del reference

    record(
        "dataset/loader 생성이 RNG를 소비하지 않음",
        ("parameter 집합이 다릅니다" if order_deviation is None
         else f"최대편차 {order_deviation:.3e}"),
        order_deviation == 0.0,
        False,
    )

    result["backbone_preserved"] = deviation == 0.0
    result["seed_order_independent"] = order_deviation == 0.0
    result["all_fatal_passed"] = deviation == 0.0

    print()

    if order_deviation == 0.0:
        print("  dataset/loader가 RNG를 쓰지 않으므로, 이 스크립트의 공유")
        print("  parameter 초기값은 V1 seed42와 같습니다.")
    else:
        print("  dataset/loader가 RNG를 소비합니다. 이 스크립트는 V1과 같은")
        print("  set_seed -> Dataset -> DataLoader -> Model 순서를 그대로")
        print("  따르므로 초기값은 여전히 V1과 같습니다.")

    if not result["all_fatal_passed"]:
        print()
        print("  => backbone 초기값이 보존되지 않았습니다. 실행을 중단합니다.")

    return result


def print_parameter_report(
    report: Dict[str, Any], learning_rate: float
) -> None:
    section("parameter / optimizer group")

    trainable = report["backbone_trainable"]
    new_modules = report["new_modules"]
    unused = report["unused_frozen"]

    print(f"  {'group':<36} {'tensor':>8} {'parameter':>14} {'LR':>10}")
    print("  " + "-" * 72)
    print(
        f"  {'A  embeddings + encoder':<36} {trainable['tensors']:>8,} "
        f"{trainable['count']:>14,} {learning_rate:>10.1e}"
    )
    print(
        f"  {'B  new direct modules':<36} {new_modules['tensors']:>8,} "
        f"{new_modules['count']:>14,} {learning_rate:>10.1e}"
    )
    print(
        f"  {'-  unused (frozen, optimizer 제외)':<36} "
        f"{unused['tensors']:>8,} {unused['count']:>14,} {'-':>10}"
    )
    print("  " + "-" * 72)
    print(f"  {'학습 대상 합계':<36} {'':>8} {report['trainable_total']:>14,}")
    print(f"  {'모델 전체 (frozen 포함)':<36} {'':>8} "
          f"{report['model_total']:>14,}")

    print()
    print("  A / B 모두 같은 LR입니다. differential LR을 쓰지 않습니다.")
    print("  warm-start가 없으므로 freeze 구간도 없습니다.")

    print()
    print("  B (새 module)")
    for name in new_modules["names"]:
        print(f"    {name}")

    print()
    print("  frozen — forward에서 쓰지 않고 optimizer에도 없음")

    groups: Dict[str, int] = {}
    for name in unused["names"]:
        head = name.split(".")[1] if name.startswith("backbone.") else name
        head = head.split(".")[0]
        groups[head] = groups.get(head, 0) + 1

    for head in sorted(groups):
        print(f"    backbone.{head}  ({groups[head]} tensor)")


def print_shapes(
    model: DirectScorer,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """학습 전에 한 batch의 tensor shape를 확인한다."""
    section("tensor shape 확인 (한 batch)")

    batch = next(iter(loader))

    history_sids = batch["history_sids"].to(device)
    history_mask = batch["history_mask"].to(device)
    candidate_sids = batch["candidate_sids"].to(device)

    shapes = model.describe_shapes(history_sids, history_mask, candidate_sids)

    batch_size = history_sids.shape[0]
    history_length = history_mask.shape[1]
    levels = model.history_levels

    print(f"  B = {batch_size}, H = {history_length} (이 batch의 최대 history 길이), "
          f"L = {levels}, d_model = {model.d_model}")
    print()

    expected = {
        "history raw": (batch_size, history_length, levels),
        "history embedding": (batch_size, history_length, levels, model.d_model),
        "encoder input": (batch_size, history_length * levels, model.d_model),
        "encoder output": (batch_size, history_length * levels, model.d_model),
        "article vectors": (batch_size, history_length, model.d_model),
        "user vector": (batch_size, model.d_model),
        "candidate vectors": (batch_size, NUM_CANDIDATES, model.d_model),
        "candidate scores": (batch_size, NUM_CANDIDATES),
    }

    rows: List[Dict[str, Any]] = []
    all_ok = True

    print(f"  {'단계':<20} {'실제':<24} {'기대':<24}")
    print("  " + "-" * 74)

    for name, shape in shapes:
        want = expected[name]
        ok = shape == want
        all_ok = all_ok and ok

        rows.append({
            "stage": name, "actual": list(shape),
            "expected": list(want), "match": ok,
        })

        print(f"  {name:<20} {str(shape):<24} {str(want):<24} "
              f"{'OK' if ok else 'MISMATCH'}")

    print()

    if all_ok:
        print("  전부 일치합니다.")
    else:
        print("  => shape가 기대와 다릅니다. 실행을 중단합니다.")

    if levels == 3:
        print()
        print("  history에 c4를 넣지 않습니다. encoder 입력 자체가 [B, H*3, d]이므로")
        print("  self-attention 단계에서도 c4 정보가 섞이지 않습니다.")

    return rows, all_ok


def print_norms(
    model: DirectScorer,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """첫 batch에서 각 항의 크기를 잰다.

    가중치를 곱한 뒤의 크기를 봐야 실제 기여를 알 수 있다.
    0.1*h3가 h1보다 한참 작으면 c3는 사실상 안 쓰이는 것이고,
    0.01*P(e_c4)가 a_semantic에 비해 무시할 수준이면 identity
    신호가 실제로 약한 것이다.
    """
    section("항별 크기 (첫 batch, padding 제외)")

    batch = next(iter(loader))

    history_sids = batch["history_sids"].to(device)
    history_mask = batch["history_mask"].to(device)

    rows = model.describe_norms(history_sids, history_mask)

    largest = max(value for _, value in rows) or 1.0

    print(f"  {'항':<26} {'평균 norm':>12}   {'상대':>7}")
    print("  " + "-" * 52)

    for label, value in rows:
        print(f"  {label:<26} {value:>12.4f}   {value / largest:>6.1%}")

    result = [{"term": label, "mean_norm": value} for label, value in rows]

    semantic = next(
        (v for k, v in rows if k == "mean ||a_semantic||"), None
    )
    residual = next(
        (v for k, v in rows if k.startswith("mean ||") and "P(e_c4)" in k
         and not k.startswith("mean ||P(")), None
    )

    if semantic and residual:
        print()
        print(f"  identity residual이 a_semantic의 {residual / semantic:.2%} 크기입니다.")

    return result


def build_optimizer(
    model: DirectScorer, learning_rate: float, weight_decay: float = 0.0
) -> AdamW:
    """학습 대상만 담는다. 쓰지 않는 decoder 계열은 넣지 않는다."""
    params = [
        param for _, param in
        model.backbone_trainable_parameters() + model.new_parameters()
    ]

    return AdamW(params, lr=learning_rate, weight_decay=weight_decay)


# ---------------------------------------------------------------- 학습 / 평가


def run_epoch(
    model: DirectScorer,
    loader: DataLoader,
    loss_fn: TransformerLoss,
    device: torch.device,
    optimizer: Optional[AdamW],
    label: str,
    progress_interval: int,
) -> Dict[str, float]:
    training = optimizer is not None

    model.train(training)

    metrics = create_metric_dict()
    num_batches = len(loader)
    started = time.time()

    for batch_idx, batch in enumerate(loader):
        history_sids = batch["history_sids"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
        candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            output = model(
                history_sids=history_sids,
                history_mask=history_mask,
                candidate_sids=candidate_sids,
            )

            loss_output = loss_fn(
                candidate_scores=output.candidate_scores,
                candidate_labels=candidate_labels,
            )

        if training:
            loss_value = float(loss_output.total_loss.detach())

            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"{label} batch {batch_idx + 1}에서 loss가 "
                    f"유한하지 않습니다: {loss_value}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss_output.total_loss.backward()
            optimizer.step()

        update_metrics(
            metrics=metrics,
            loss_output=loss_output,
            model_output=output,
            candidate_labels=candidate_labels,
        )

        done = batch_idx + 1

        if progress_interval > 0 and (
            done % progress_interval == 0 or done == num_batches
        ):
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (num_batches - done) / rate if rate > 0 else 0.0
            print(
                f"    [{label}] {done:,}/{num_batches:,} "
                f"({100.0 * done / num_batches:5.1f}%)  "
                f"loss {float(loss_output.total_loss.detach()):.4f}  "
                f"남은 {format_duration(eta)}",
                flush=True,
            )

    return finalize_metrics(metrics, loss_fn)


@torch.no_grad()
def predict_validation(
    model: DirectScorer,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """validation 후보별 direct score를 저장용으로 뽑는다."""
    model.eval()

    rows: List[Dict[str, Any]] = []
    sample_index = 0

    for batch in loader:
        history_sids = batch["history_sids"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
        candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

        output = model(
            history_sids=history_sids,
            history_mask=history_mask,
            candidate_sids=candidate_sids,
        )

        scores = output.candidate_scores
        probs = torch.softmax(scores, dim=1)

        order = torch.argsort(scores, dim=1, descending=True)
        ranks = torch.empty_like(order)
        batch_size, num_candidates = scores.shape
        ranks.scatter_(
            1, order,
            torch.arange(1, num_candidates + 1, device=device)
            .unsqueeze(0).expand(batch_size, -1),
        )

        impression_ids = batch["impression_ids"]

        for i in range(batch_size):
            for candidate_idx in range(num_candidates):
                sid = candidate_sids[i, candidate_idx].detach().cpu().tolist()

                rows.append({
                    "impression_id": impression_ids[i],
                    GROUP_COLUMN: sample_index,
                    "candidate_index": candidate_idx,
                    "label": float(candidate_labels[i, candidate_idx].item()),
                    "c1": int(sid[0]),
                    "c2": int(sid[1]),
                    "c3": int(sid[2]),
                    "direct_score": float(scores[i, candidate_idx].item()),
                    "probability": float(probs[i, candidate_idx].item()),
                    "rank": int(ranks[i, candidate_idx].item()),
                })

            sample_index += 1

    return pd.DataFrame(rows)


def offline_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """저장된 예측으로 metric을 한 번 더 계산해 학습 경로와 대조한다."""
    work = df[[GROUP_COLUMN, "label"]].copy()
    work["label"] = work["label"].astype(int)
    work["candidate_score"] = df["direct_score"].to_numpy()
    return evaluate_ranking(df=work, group_column=GROUP_COLUMN)


# ---------------------------------------------------------------- 보고


def comparison_rows(
    result: Dict[str, float],
    scorer_type: str,
    history_levels: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in BASELINES]

    codes = "c1234" if history_levels == 4 else "c123"

    if args.history_mode == "weighted":
        w = args.level_weights
        label = f"P2 weighted {codes}"
        detail = (
            f"({w[0]:g}*h1 + {w[1]:g}*h2 + {w[2]:g}*h3) / {sum(w):g}"
        )
    else:
        label = f"P2-A {codes} mean"
        detail = f"direct u^T W v, history = {codes}, 단순 평균"

    if args.use_c4_identity:
        label += f" + {args.beta4:g}*c4"
        detail += f", c4 identity residual beta4={args.beta4:g}"

    label += f" ({scorer_type})"

    rows.append({
        "config": label,
        "detail": detail,
        "tuned_on": "Train",
        "top1_accuracy": result["top1_accuracy"],
        "mrr": result["mrr"],
        "ndcg5": result["ndcg@5"],
        "auc": result["auc"],
        "preference_loss": result["preference_loss"],
        "note": "이번 실험",
    })

    return rows


def print_comparison(rows: List[Dict[str, Any]]) -> None:
    section("비교")

    print(
        f"  {'구분':<26} {'튜닝 데이터':<12} {'Top-1':>10} {'MRR':>9} "
        f"{'nDCG@5':>9} {'AUC':>9}"
    )
    print("  " + "-" * 82)

    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            return f"{value:>9.4f}" if isinstance(value, float) else f"{'-':>9}"

        print(
            f"  {row['config']:<26} {row['tuned_on']:<12} "
            f"{row['top1_accuracy'] * 100:>9.3f}% {fmt('mrr')} "
            f"{fmt('ndcg5')} {fmt('auc')}"
        )

    direct = rows[-1]

    print()
    print("  Fixed weighted grid(29.312%)는 validation에서 42칸을 훑어 고른")
    print("  validation-tuned weighted reference입니다. 합격선이 아닙니다.")
    print("  Train에서 학습한 P2-A와 같은 조건의 비교가 아닙니다.")
    print()

    for row in rows[:-1]:
        delta = (direct["top1_accuracy"] - row["top1_accuracy"]) * 100
        verdict = "넘음" if delta > 0 else "못 넘음"
        print(
            f"    vs {row['config']:<24} "
            f"{delta:+7.3f}%p   {verdict}"
        )


def write_readme(
    path: Path,
    history: List[Dict[str, Any]],
    best: Dict[str, Any],
    comparison: List[Dict[str, Any]],
    config_check: Dict[str, Any],
    init_check: Dict[str, Any],
    param_report: Dict[str, Any],
    metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines: List[str] = []
    add = lines.append

    add("# Priority 2 — SID Direct Scorer (P2-A Bilinear)")
    add("")
    add("생성확률 합산을 ranking score로 쓰지 않고, history user vector와")
    add("candidate SID vector의 직접 적합도를 학습했다.")
    add("")
    add("    u  = masked attention pooling over history article vectors")
    add("    vi = Linear(concat(c1_emb, c2_emb, c3_emb))")
    add("    Di = u^T W vi")
    add("")
    add("loss는 기존 5-candidate listwise CE 그대로다. score 함수만 바꿨다.")
    add("")
    add("**V1 checkpoint를 쓰지 않았다.** random initialization에서 "
        "from-scratch로 학습했다.")
    add("학습 조건을 V1 seed42 final과 같게 맞춰서, 달라지는 것이 score "
        "구조 하나만 남게 했다.")
    add("")
    add("    V1 : Decoder -> L1 + L2 + L3")
    add("    P2 : article pooling + candidate projection -> u^T W v")
    add("")
    add(f"history = `{metadata['history_codes']}`, "
        f"candidate = `{metadata['candidate_codes']}`")
    add("")

    if metadata["history_mode"] == "weighted":
        w = metadata["level_weights"]
        add(f"**article vector는 단순 평균이 아니다.** "
            f"`({w[0]:g}*h1 + {w[1]:g}*h2 + {w[2]:g}*h3) / {sum(w):g}`")
        add("")
        add(f"`{w}`는 V1의 L1/L2/L3 score weight grid에서 Validation 기준")
        add("가장 좋았던 비율이다. embedding 가중치로 최적이라고 증명된 값이")
        add("아니라, 그 비율을 고정 가설로 옮겨와 확인하는 것이다.")
        add("")

    if metadata["use_c4_identity"]:
        add(f"**c4는 encoder token으로 넣지 않는다.** self-attention을 거치지 "
            f"않고")
        add(f"`a_final = a_semantic + {metadata['beta4']:g} * P(e_c4)` 로 "
            f"약한 identity residual로만 더한다.")
        add(f"beta4 = {metadata['beta4']:g}는 고정이며 학습하지 않는다.")
        add("")

    if metadata["history_levels"] == 3:
        add("**c4 ablation.** history 입력 자체를 `[B,H,3]`으로 만들어 c4를")
        add("encoder에 넣지 않았다. encoder에 넣고 평균에서만 빼면 "
            "self-attention 단계에서")
        add("c4 정보가 이미 c1/c2/c3 hidden state에 섞이므로 ablation이 되지 않는다.")
        add("")

    add("## 실행 정보")
    add("")
    add(f"- 학습 방식     : from scratch (warm-start 없음)")
    add(f"- train         : `{metadata['train_path']}`")
    add(f"- validation    : `{metadata['validation_path']}`")
    add(f"- scorer        : {metadata['scorer_type']}")
    add(f"- batch size    : {args.batch_size}")
    add(f"- learning rate : {args.learning_rate}  (단일, freeze 구간 없음)")
    add(f"- dropout       : {args.dropout}")
    add(f"- weight decay  : {args.weight_decay}")
    add(f"- max epoch     : {args.max_epochs}, patience {args.patience}")
    add(f"- best 선택     : Validation Top-1")
    add(f"- seed          : {args.seed}")
    add(f"- git commit    : {metadata.get('git_commit')}")
    add("")

    add("## config 확인")
    add("")
    add("| parameter | 기대값 | 실제값 | 판정 |")
    add("|---|---|---|---|")

    for row in config_check["rows"]:
        verdict = "OK" if row["match"] else "MISMATCH"
        add(f"| `{row['parameter']}` | {row['expected']} | "
            f"{row['actual']} | {verdict} |")

    add("")
    add("기대값과 하나라도 다르면 실행을 중단한다.")
    add("")

    add("## 초기화 확인")
    add("")
    add("| 확인 | 내용 | 판정 |")
    add("|---|---|---|")

    for check in init_check["checks"]:
        verdict = ("OK" if check["passed"]
                   else ("FAIL" if check["fatal"] else "참고"))
        add(f"| {check['name']} | {check['detail']} | {verdict} |")

    add("")
    add("공유하는 embedding / encoder는 V1과 같은 생성 순서를 거치므로 "
        "seed42에서 같은")
    add("초기값에서 출발한다. decoder / BOS / c1c2c3 head도 함께 만든다. "
        "쓰지는 않지만,")
    add("만들지 않으면 그 뒤 `_reset_parameters()`가 보는 RNG 상태가 "
        "달라져 embedding")
    add("초기값이 V1과 어긋나기 때문이다.")
    add("")

    add("## tensor shape")
    add("")
    add("| 단계 | 실제 | 기대 | 판정 |")
    add("|---|---|---|---|")

    for row in metadata["shape_check"]:
        verdict = "OK" if row["match"] else "MISMATCH"
        add(f"| {row['stage']} | {tuple(row['actual'])} | "
            f"{tuple(row['expected'])} | {verdict} |")

    add("")

    add("## 항별 크기 (첫 batch)")
    add("")
    add("| 항 | 평균 norm |")
    add("|---|---|")

    for row in metadata["first_batch_norms"]:
        add(f"| `{row['term']}` | {row['mean_norm']:.4f} |")

    add("")
    add("가중치를 곱한 뒤의 크기다. padding 위치는 제외했다.")
    add("")

    add("## parameter")
    add("")
    add("| group | tensor | parameter | LR |")
    add("|---|---|---|---|")
    add(f"| A embeddings + encoder | "
        f"{param_report['backbone_trainable']['tensors']:,} | "
        f"{param_report['backbone_trainable']['count']:,} | "
        f"{args.learning_rate} |")
    add(f"| B new direct modules | "
        f"{param_report['new_modules']['tensors']:,} | "
        f"{param_report['new_modules']['count']:,} | "
        f"{args.learning_rate} |")
    add(f"| frozen (decoder / BOS / c1c2c3 head) | "
        f"{param_report['unused_frozen']['tensors']:,} | "
        f"{param_report['unused_frozen']['count']:,} | - |")
    add("")
    add(f"학습 대상 합계 {param_report['trainable_total']:,}, "
        f"모델 전체 {param_report['model_total']:,}")
    add("")

    add("## 비교")
    add("")
    add("| 구분 | 상세 | 튜닝 데이터 | Top-1 | MRR | nDCG@5 | AUC |")
    add("|---|---|---|---|---|---|---|")

    for row in comparison:
        def cell(key: str) -> str:
            value = row.get(key)
            return f"{value:.4f}" if isinstance(value, float) else "-"

        add(
            f"| {row['config']} | {row['detail']} | {row['tuned_on']} | "
            f"{row['top1_accuracy'] * 100:.3f}% | {cell('mrr')} | "
            f"{cell('ndcg5')} | {cell('auc')} |"
        )

    add("")
    add("**Fixed weighted grid의 29.312%는 validation-tuned weighted "
        "reference다.** validation에서")
    add("42칸을 훑어 고른 값이라 Train에서 학습한 P2-A와 같은 조건의 "
        "비교가 아니고, 합격선도")
    add("아니다. 그래도 넘는지는 위 표에서 같이 본다.")
    add("")

    add("## epoch")
    add("")
    add("| epoch | train loss | val loss | val Top-1 | MRR | nDCG@5 | AUC |")
    add("|---|---|---|---|---|---|---|")

    for row in history:
        add(
            f"| {row['epoch']} | "
            f"{row['train_loss']:.6f} | {row['validation_loss']:.6f} | "
            f"{row['validation_top1']:.6f} | {row['mrr']:.4f} | "
            f"{row['ndcg5']:.4f} | {row['auc']:.4f} |"
        )

    add("")
    add(f"best epoch {best['epoch']} (Validation Top-1 "
        f"{best['metrics']['top1_accuracy']:.6f})")
    add("")

    add("## 하지 않은 것")
    add("")
    add("- V1 checkpoint warm-start, freeze 구간, differential LR")
    add("- alpha1/alpha2/alpha3, L1/L2/L3 weighted score 결합")
    add("- hybrid (direct + weighted)")
    add("- history overlap feature, candidate c4")
    add("- hard-negative loss, 새로운 loss")
    add("- MLP scorer (P2-B)")
    add("- Test 데이터 사용")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Priority 2 SID Direct Scorer 학습"
    )
    parser.add_argument(
        "--priority0-dir",
        default="sweep_out/ebnerd_v2/priority0_shuffled/seed42",
    )
    parser.add_argument(
        "--out",
        default="sweep_out/ebnerd_v2/priority2_direct/bilinear/seed42",
    )
    parser.add_argument("--config", default="configs/transformer_ebnerd.gin")
    parser.add_argument(
        "--train-path",
        default="datasets/ebnerd/train_sequences_1pos4neg.parquet",
    )
    parser.add_argument("--validation-path", default=None)
    parser.add_argument(
        "--gin-binding", action="append", default=[], metavar="BINDING"
    )

    parser.add_argument(
        "--history-levels", type=int, default=4, choices=[3, 4],
        help="4 = history c1c2c3c4 (기본), 3 = history c1c2c3 (c4 ablation)",
    )
    parser.add_argument(
        "--history-mode", default="mean", choices=["mean", "weighted"],
        help=(
            "mean = (h1+h2+h3)/L, "
            "weighted = (1.0*h1 + 0.5*h2 + 0.1*h3) / 1.6"
        ),
    )
    parser.add_argument(
        "--level-weights", type=float, nargs=3, default=[1.0, 0.5, 0.1],
        metavar=("W1", "W2", "W3"),
    )
    parser.add_argument(
        "--use-c4-identity", action="store_true",
        help="c4를 encoder에 넣지 않고 약한 identity residual로만 더한다.",
    )
    parser.add_argument("--beta4", type=float, default=0.01)
    parser.add_argument("--scorer", default="bilinear", choices=["bilinear", "mlp"])
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="NewsEncoderDecoderTransformer.dropout_rate를 이 값으로 둔다.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-interval", type=int, default=100)

    args = parser.parse_args()

    started_at = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    priority0_dir = resolve_path(args.priority0_dir)
    out_dir = resolve_path(args.out)
    config_path = resolve_path(args.config)
    train_path = resolve_path(args.train_path)

    section("Priority 2 — SID Direct Scorer")

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"출력 폴더가 비어 있지 않습니다. 덮어쓰지 않습니다: {out_dir}")
        return 1

    meta_path = priority0_dir / "run_metadata.json"

    if not meta_path.exists():
        print(f"Priority 0 run_metadata.json이 없습니다: {meta_path}")
        return 1

    p0_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    validation_path = resolve_path(
        args.validation_path if args.validation_path else p0_meta["dataset"]
    )

    bindings, binding_source = bindings_from_priority0(p0_meta, args.gin_binding)

    # dropout은 gin으로 들어가므로 Priority 0 설정 뒤에 덮어쓴다.
    bindings = [
        b for b in bindings
        if b.split("=", 1)[0].strip() != DROPOUT_KEY
    ]
    bindings.append(format_binding(DROPOUT_KEY, float(args.dropout)))

    for path, label in (
        (train_path, "train parquet"),
        (validation_path, "validation parquet"),
        (config_path, "gin config"),
    ):
        if not path.exists():
            print(f"{label}가 없습니다: {path}")
            return 1

    device = get_device()

    print(f"  train       : {train_path}")
    print(f"  validation  : {validation_path}")
    print(f"  scorer      : {args.scorer}")
    print(f"  history     : {'c1c2c3c4' if args.history_levels == 4 else 'c1c2c3'}"
          f"  ({args.history_levels} level, encoder token)")

    if args.history_mode == "weighted":
        w = args.level_weights
        print(f"  article     : ({w[0]:g}*h1 + {w[1]:g}*h2 + {w[2]:g}*h3) "
              f"/ {sum(w):g}   <- 단순 평균 아님")
    else:
        print(f"  article     : level 단순 평균")

    if args.use_c4_identity:
        print(f"  c4          : identity residual, beta4 = {args.beta4:g} (고정)")
    else:
        print(f"  c4          : 사용 안 함")
    print(f"  출력        : {out_dir}")
    print(f"  device      : {device}")
    print(f"  gin binding : {len(bindings)}개  (출처: {binding_source})")

    for binding in bindings:
        print(f"      {binding}")

    print(f"  seed        : {args.seed}")
    print(f"  learning rate: {args.learning_rate}")
    print(f"  dropout     : {args.dropout}")
    print(f"  weight decay: {args.weight_decay}")
    print()
    print("  V1 checkpoint를 쓰지 않습니다. random init에서 from-scratch로")
    print("  학습합니다. 생성확률(L1/L2/L3)을 score로 쓰지 않습니다.")
    print("  Test를 쓰지 않습니다.")

    configure_gin(config_path, bindings)

    config_check = verify_config(expected_dropout=args.dropout)

    if not config_check["all_match"]:
        return 1

    # ---- 데이터
    # V1(train_transformer.train)과 같은 순서를 지킨다.
    #   set_seed -> Dataset -> DataLoader -> Model
    # 이 순서를 지켜야 공유 parameter가 V1 seed42와 같은 초기값이 된다.
    section("데이터")

    set_seed(args.seed)

    train_dataset = NewsSequenceDataset(parquet_path=str(train_path))
    validation_dataset = NewsSequenceDataset(parquet_path=str(validation_path))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )

    print(f"  train impression      : {len(train_dataset):,}")
    print(f"  validation impression : {len(validation_dataset):,}")
    print(f"  epoch당 step          : {len(train_loader):,}")

    # ---- 모델 (from scratch)
    section("모델 생성 (from scratch, checkpoint 없음)")

    backbone = build_backbone(device)

    print("  NewsEncoderDecoderTransformer를 V1과 같은 순서로 만들었습니다.")
    print("  decoder / BOS / c1c2c3 head도 함께 만듭니다. 쓰지는 않지만,")
    print("  만들지 않으면 embedding 초기값이 V1과 달라집니다.")

    before = snapshot(
        backbone, tuple(BACKBONE_TRAINABLE_PREFIXES)
    )
    before = {f"backbone.{name}": value for name, value in before.items()}

    model = DirectScorer(
        backbone=backbone,
        scorer_type=args.scorer,
        mlp_hidden=args.mlp_hidden,
        mlp_dropout=args.mlp_dropout,
        history_levels=args.history_levels,
        history_mode=args.history_mode,
        level_weights=tuple(args.level_weights),
        use_c4_identity=args.use_c4_identity,
        beta4=args.beta4,
    ).to(device)

    init_check = verify_initialization(model, before, args.seed, device)

    if not init_check["all_fatal_passed"]:
        return 1

    param_report = model.parameter_report()

    print_parameter_report(param_report, args.learning_rate)

    shape_rows, shapes_ok = print_shapes(model, train_loader, device)

    if not shapes_ok:
        return 1

    norm_rows = print_norms(model, train_loader, device)

    optimizer = build_optimizer(
        model, args.learning_rate, args.weight_decay
    )
    loss_fn = TransformerLoss().to(device)

    # ---- 학습
    section("학습")

    print(f"  embedding / encoder / 새 module 전부 처음부터 함께 학습합니다.")
    print(f"  LR {args.learning_rate:.1e} 단일, weight decay 0, "
          f"freeze 구간 없음.")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_top1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        print(f"  [epoch {epoch}/{args.max_epochs}]")

        epoch_started = time.time()

        train_metrics = run_epoch(
            model, train_loader, loss_fn, device, optimizer,
            f"epoch {epoch} train", args.progress_interval,
        )

        validation_metrics = run_epoch(
            model, validation_loader, loss_fn, device, None,
            f"epoch {epoch} val", args.progress_interval,
        )

        improved = validation_metrics["top1_accuracy"] > best_top1

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["total_loss"],
            "train_top1": train_metrics["top1_accuracy"],
            "validation_loss": validation_metrics["total_loss"],
            "preference_loss": validation_metrics["preference_loss"],
            "validation_top1": validation_metrics["top1_accuracy"],
            "mrr": validation_metrics["mrr"],
            "ndcg5": validation_metrics["ndcg@5"],
            "auc": validation_metrics["auc"],
            "positive_score": validation_metrics["positive_score"],
            "negative_score": validation_metrics["negative_score"],
            "score_gap": validation_metrics["score_gap"],
            "positive_prob": validation_metrics["positive_prob"],
            "epoch_seconds": round(time.time() - epoch_started, 1),
            "is_best": improved,
        })

        print(
            f"    train loss {train_metrics['total_loss']:.6f} | "
            f"val loss {validation_metrics['total_loss']:.6f} | "
            f"val Top-1 {validation_metrics['top1_accuracy']:.6f} | "
            f"MRR {validation_metrics['mrr']:.4f} | "
            f"nDCG@5 {validation_metrics['ndcg@5']:.4f} | "
            f"AUC {validation_metrics['auc']:.4f}"
            f"{'   <- best' if improved else ''}",
            flush=True,
        )

        if improved:
            best_top1 = validation_metrics["top1_accuracy"]
            epochs_without_improvement = 0
            best = {"epoch": epoch, "metrics": validation_metrics}

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "scorer_type": args.scorer,
                    "validation_metrics": validation_metrics,
                    "gin_bindings": bindings,
                    "trained_from_scratch": True,
                },
                out_dir / "checkpoint_best.pt",
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print()
                print(f"  Validation Top-1이 {args.patience} epoch 동안 "
                      f"좋아지지 않아 멈춥니다.")
                break

    if best is None:
        print("학습이 한 epoch도 완료되지 않았습니다.")
        return 1

    # ---- best checkpoint로 validation 예측 저장
    section("best checkpoint로 validation 예측")

    checkpoint = torch.load(out_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    predictions = predict_validation(model, validation_loader, device)
    predictions.to_parquet(out_dir / "validation_predictions.parquet", index=False)

    offline = offline_metrics(predictions)

    print(f"  {len(predictions):,} row 저장")
    print()
    print(f"  {'지표':<10} {'학습 경로':>14} {'예측 파일':>14} {'차이':>12}")
    print("  " + "-" * 54)

    pairs = [
        ("Top-1", "top1_accuracy", "top1_accuracy"),
        ("MRR", "mrr", "mrr"),
        ("nDCG@5", "ndcg@5", "ndcg@5"),
        ("AUC", "auc", "auc"),
    ]

    worst = 0.0

    for display, train_key, offline_key in pairs:
        got = best["metrics"][train_key]
        want = offline[offline_key]
        delta = abs(got - want)
        worst = max(worst, delta)
        print(f"  {display:<10} {got:>14.10f} {want:>14.10f} {delta:>12.3e}")

    print()

    if worst < 1e-6:
        print("  두 경로의 metric이 일치합니다.")
    else:
        print(f"  !! 두 경로의 metric이 다릅니다 (최대 {worst:.3e}).")

    # ---- 비교
    comparison = comparison_rows(
        best["metrics"], args.scorer, args.history_levels, args
    )
    print_comparison(comparison)

    # ---- 저장
    section("저장")

    write_csv(out_dir / "training_history.csv", history)
    write_csv(
        out_dir / "comparison.csv",
        [
            {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
            for row in comparison
        ],
    )

    (out_dir / "best_metrics.json").write_text(
        json.dumps({
            "best_epoch": best["epoch"],
            "selection_criterion": "validation top1_accuracy",
            "validation": best["metrics"],
            "offline_recomputed": offline,
            "comparison": comparison,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    metadata = {
        "experiment": "priority2_direct_scorer",
        "scorer_type": args.scorer,
        "history_levels": args.history_levels,
        "history_codes": (
            "c1,c2,c3,c4" if args.history_levels == 4 else "c1,c2,c3"
        ),
        "history_mode": args.history_mode,
        "level_weights": list(args.level_weights),
        "use_c4_identity": args.use_c4_identity,
        "beta4": args.beta4,
        "beta4_learnable": False,
        "candidate_codes": "c1,c2,c3",
        "score_formula": (
            "D = u^T W v" if args.scorer == "bilinear"
            else "D = MLP([u, v, u*v, |u-v|])"
        ),
        "trained_from_scratch": True,
        "warm_start": False,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "priority0_dir": str(priority0_dir),
        "config": str(config_path),
        "gin_bindings": bindings,
        "gin_binding_source": binding_source,
        "config_check": config_check,
        "initialization_check": init_check,
        "shape_check": shape_rows,
        "first_batch_norms": norm_rows,
        "parameter_report": param_report,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "selection_criterion": "validation top1_accuracy",
        "seed": args.seed,
        "train_impressions": len(train_dataset),
        "validation_impressions": len(validation_dataset),
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "did_not_do": [
            "V1 checkpoint warm-start",
            "encoder/embedding freeze 구간",
            "differential learning rate",
            "alpha1/alpha2/alpha3 가중치",
            "L1/L2/L3 weighted score 결합",
            "hybrid (direct + weighted)",
            "history overlap feature",
            "candidate c4",
            "hard-negative loss",
            "새로운 loss",
            "MLP scorer (P2-B)",
            "Test 사용",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "git_commit": git_commit_hash(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }

    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    write_readme(
        out_dir / "README.md",
        history, best, comparison, config_check, init_check,
        param_report, metadata, args,
    )

    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {out_dir / name}")

    print()
    print(f"  총 소요 {format_duration(time.time() - started_at)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
