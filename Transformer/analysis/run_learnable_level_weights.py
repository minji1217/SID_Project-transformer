"""level weight alpha1 / alpha2 / alpha3를 Train에서 학습한다.

기존 V1 score

    S = L1 + L2 + L3

를

    S = alpha1 * L1 + alpha2 * L2 + alpha3 * L3

로 바꾸고 alpha 3개만 학습한다. Transformer는 전부 freeze한다.

alpha parameterization

    alpha = 3.0 * softmax(theta)

theta = [0, 0, 0]에서 시작하므로 alpha = [1, 1, 1], 즉 기존 V1과 같다.
coarse grid best를 초기값으로 쓰지 않는다. 학습이 어느 방향으로
weight를 옮기는지 보려면 V1에서 출발해야 한다.

제약:
    alpha1, alpha2, alpha3 > 0
    alpha1 + alpha2 + alpha3 = 3

이 스크립트가 하지 않는 것
  Transformer 재학습, Transformer parameter 갱신, RQ-VAE 재학습,
  SID 재생성, Test 사용, Priority 2

Transformer가 freeze이므로 L1/L2/L3는 상수다. 매 epoch forward를
다시 하지 않고 한 번 추출해 cache로 두고 alpha만 학습한다.
loss가 candidate score 5개만의 함수이므로 (modules/loss.py) 이 방식은
근사가 아니라 완전히 동일하다.

    cd Transformer
    python -m analysis.run_learnable_level_weights \
        --priority0-dir sweep_out/ebnerd_v2/priority0_shuffled/seed42 \
        --out sweep_out/ebnerd_v2/learnable_level_weights/seed42
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
from torch.utils.data import DataLoader

from data.sequence import NewsSequenceDataset, collate_news_sequences
from evaluate.metrics import evaluate_ranking
from modules.loss import TransformerLoss
from modules.model import NewsEncoderDecoderTransformer
from sweep.run_stage import format_binding, write_csv


BASE_DIR = Path(__file__).resolve().parent.parent

NUM_CANDIDATES = 5
NUM_LEVELS = 3
GROUP_COLUMN = "sample_index"

ALPHA_SUM = 3.0

LEVEL_COLUMNS = {"L1": "c1_log_prob", "L2": "c2_log_prob", "L3": "c3_log_prob"}

# 비교 기준. coarse grid best는 합이 1.6이라 CE loss를 V1과 직접
# 비교할 수 없다. 합 3으로 rescale한 행을 같이 둔다.
# rescale해도 순위는 그대로다 (score를 양수배 해도 대소가 안 바뀐다).
COARSE_BEST = (1.0, 0.5, 0.1)
COARSE_BEST_SUM = sum(COARSE_BEST)
COARSE_BEST_RESCALED = tuple(a * ALPHA_SUM / COARSE_BEST_SUM for a in COARSE_BEST)

V1_ALPHA = (1.0, 1.0, 1.0)

# epoch 0의 alpha가 정확히 [1,1,1]인지 볼 때 쓰는 허용치.
# 3 * softmax([0,0,0])은 IEEE double에서 정확히 1.0이 된다.
ALPHA_EXACT_TOLERANCE = 0.0

# epoch 0 metric이 Priority 0의 S123와 같은지 볼 때 쓰는 허용치
SANITY_METRIC_TOLERANCE = 1e-12
SANITY_SCORE_TOLERANCE = 1e-9

# alpha가 한 epoch에 이보다 크게 움직이면 급변으로 보고 경고한다
ALPHA_JUMP_WARN = 0.5


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


# ---------------------------------------------------------------- alpha module


class LevelWeights(nn.Module):
    """alpha = ALPHA_SUM * softmax(theta).

    theta는 상수를 더해도 alpha가 같다. 즉 theta는 shift만큼
    비유일하다. 기록은 하되 해석은 alpha로 한다.
    """

    def __init__(self, alpha_sum: float = ALPHA_SUM) -> None:
        super().__init__()
        self.alpha_sum = alpha_sum
        self.theta = nn.Parameter(torch.zeros(NUM_LEVELS, dtype=torch.float64))

    def alpha(self) -> Tensor:
        return self.alpha_sum * torch.softmax(self.theta, dim=0)

    def forward(self, levels: Tensor) -> Tensor:
        # levels: [B, 5, 3] -> [B, 5]
        return (levels * self.alpha()).sum(dim=-1)


# ---------------------------------------------------------------- L 추출


@torch.no_grad()
def extract_levels(
    model: NewsEncoderDecoderTransformer,
    dataloader: DataLoader,
    device: torch.device,
    label: str,
) -> pd.DataFrame:
    """L1 / L2 / L3를 뽑는다.

    predict_sid.py의 predict()와 같은 조건으로 돈다.
    model.eval() + no_grad + autocast 없음(fp32).

    predict_sid.py는 row마다 dict를 쌓아서 train처럼 큰 데이터에서
    메모리를 많이 쓴다. 여기서는 필요한 6개 컬럼만 numpy로 모은다.
    같은 값이 나오는지는 validation에서 Priority 0 결과와 대조해
    확인한다 (verify_extractor).
    """
    model.eval()

    sample_chunks: List[np.ndarray] = []
    cand_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    l1_chunks: List[np.ndarray] = []
    l2_chunks: List[np.ndarray] = []
    l3_chunks: List[np.ndarray] = []

    sample_index = 0
    num_batches = len(dataloader)
    started = time.time()

    for batch_idx, batch in enumerate(dataloader):
        history_sids = batch["history_sids"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
        candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

        out = model(
            history_sids=history_sids,
            history_mask=history_mask,
            candidate_sids=candidate_sids,
        )

        batch_size = candidate_sids.shape[0]
        num_candidates = candidate_sids.shape[1]

        if num_candidates != NUM_CANDIDATES:
            raise ValueError(
                f"후보가 {NUM_CANDIDATES}개가 아닙니다: {num_candidates}"
            )

        idx = np.arange(sample_index, sample_index + batch_size)
        sample_chunks.append(np.repeat(idx, num_candidates))
        cand_chunks.append(np.tile(np.arange(num_candidates), batch_size))

        label_chunks.append(
            candidate_labels.detach().cpu().numpy().astype(np.float64).reshape(-1)
        )
        l1_chunks.append(
            out.c1_log_probs.detach().cpu().numpy().astype(np.float64).reshape(-1)
        )
        l2_chunks.append(
            out.c2_log_probs.detach().cpu().numpy().astype(np.float64).reshape(-1)
        )
        l3_chunks.append(
            out.c3_log_probs.detach().cpu().numpy().astype(np.float64).reshape(-1)
        )

        sample_index += batch_size

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
            done = batch_idx + 1
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (num_batches - done) / rate if rate > 0 else 0.0
            print(
                f"    [{label}] batch {done:,}/{num_batches:,} "
                f"({100.0 * done / num_batches:5.1f}%)  "
                f"경과 {elapsed / 60:.1f}분  남은 {eta / 60:.1f}분",
                flush=True,
            )

    return pd.DataFrame({
        GROUP_COLUMN: np.concatenate(sample_chunks),
        "candidate_index": np.concatenate(cand_chunks),
        "label": np.concatenate(label_chunks),
        "L1": np.concatenate(l1_chunks),
        "L2": np.concatenate(l2_chunks),
        "L3": np.concatenate(l3_chunks),
    })


def bindings_from_priority0(
    p0_meta: Dict[str, Any], extra: Optional[List[str]] = None
) -> Tuple[List[str], str]:
    """Priority 0가 쓴 gin override를 그대로 되살린다.

    Priority 0는 두 가지 방식 중 하나로 설정을 기록한다.
      - selected_final.json 경로로 돌았으면 "config"에 딕셔너리
      - --checkpoint 경로로 돌았으면 "gin_bindings"에 문자열 목록

    둘 중 채워진 쪽을 쓴다. 한쪽만 보면 모델을 base config 기본값으로
    만들게 되고, checkpoint와 shape이 달라 load_state_dict가 깨진다.
    """
    if extra:
        return list(extra), "명령행 --gin-binding"

    binding_strings = p0_meta.get("gin_bindings")

    if binding_strings:
        return list(binding_strings), "Priority 0 run_metadata.json의 gin_bindings"

    config = p0_meta.get("config")

    if config:
        return (
            [format_binding(key, config[key]) for key in sorted(config)],
            "Priority 0 run_metadata.json의 config",
        )

    return [], "없음 (base config만 사용)"


def configure_gin(config_path: Path, bindings: Optional[List[str]]) -> None:
    """gin을 먼저 읽는다.

    NewsSequenceDataset.max_history_length도 gin으로 설정되므로
    Dataset을 만들기 전에 반드시 호출해야 한다. predict_sid.py도
    같은 순서다. 순서가 뒤집히면 history가 잘리지 않은 채로
    추론되어 Priority 0과 다른 값이 나온다.
    """
    gin.parse_config_file(str(config_path), skip_unknown=True)

    if bindings:
        gin.parse_config(bindings, skip_unknown=True)


def build_model(
    checkpoint_path: Path,
    device: torch.device,
) -> NewsEncoderDecoderTransformer:
    model = NewsEncoderDecoderTransformer().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise RuntimeError(
            "checkpoint와 모델 구조가 맞지 않습니다.\n"
            "gin override가 제대로 적용되지 않았을 가능성이 큽니다. "
            "위에 출력된 'gin override 출처'를 확인하세요.\n"
            f"checkpoint: {checkpoint_path}\n\n{error}"
        ) from error

    # Transformer 전체 freeze
    for param in model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable != 0:
        raise RuntimeError(
            f"Transformer freeze에 실패했습니다. 학습 가능 parameter {trainable}개"
        )

    model.eval()
    return model


def make_cache(
    config_path: Path,
    bindings: Optional[List[str]],
    checkpoint_path: Path,
    data_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    label: str,
) -> pd.DataFrame:
    # gin을 Dataset보다 먼저 읽어야 max_history_length가 적용된다.
    configure_gin(config_path, bindings)

    dataset = NewsSequenceDataset(parquet_path=str(data_path))

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
    )

    print(f"  {label} impression: {len(dataset):,}")

    if bindings:
        print(f"  gin override {len(bindings)}개 적용")

    model = build_model(checkpoint_path, device)
    return extract_levels(model, loader, device, label)


# ---------------------------------------------------------------- 평가


def as_tensors(df: pd.DataFrame, device: torch.device) -> Tuple[Tensor, Tensor]:
    """[N, 5, 3] level과 [N, 5] label로 바꾼다."""
    ordered = df.sort_values(
        [GROUP_COLUMN, "candidate_index"], kind="stable"
    ).reset_index(drop=True)

    num_rows = len(ordered)

    if num_rows % NUM_CANDIDATES != 0:
        raise ValueError(f"row 수가 {NUM_CANDIDATES}의 배수가 아닙니다: {num_rows:,}")

    num_impressions = num_rows // NUM_CANDIDATES

    counts = ordered.groupby(GROUP_COLUMN).size()

    if not (counts == NUM_CANDIDATES).all():
        raise ValueError(f"impression당 후보가 {NUM_CANDIDATES}개가 아닌 행이 있습니다.")

    levels = np.stack(
        [ordered["L1"].to_numpy(), ordered["L2"].to_numpy(), ordered["L3"].to_numpy()],
        axis=-1,
    ).reshape(num_impressions, NUM_CANDIDATES, NUM_LEVELS)

    labels = ordered["label"].to_numpy().reshape(num_impressions, NUM_CANDIDATES)

    positives = labels.sum(axis=1)

    if not np.all(positives == 1):
        raise ValueError("positive가 1개가 아닌 impression이 있습니다.")

    return (
        torch.tensor(levels, dtype=torch.float64, device=device),
        torch.tensor(labels, dtype=torch.float64, device=device),
    )


def scores_for_alpha(levels: Tensor, alpha: Tuple[float, float, float]) -> Tensor:
    weights = torch.tensor(alpha, dtype=levels.dtype, device=levels.device)
    return (levels * weights).sum(dim=-1)


def ce_loss(scores: Tensor, labels: Tensor) -> float:
    targets = labels.argmax(dim=1).long()
    return float(
        torch.nn.functional.cross_entropy(scores.float(), targets).item()
    )


def ranking_metrics(
    ordered_df: pd.DataFrame, scores: Tensor
) -> Dict[str, float]:
    """evaluate_ranking을 그대로 쓴다. 동점 규칙이 기존과 같다."""
    work = ordered_df[[GROUP_COLUMN, "label"]].copy()
    work["label"] = work["label"].astype(int)
    work["candidate_score"] = scores.detach().cpu().numpy().reshape(-1)
    return evaluate_ranking(df=work, group_column=GROUP_COLUMN)


def evaluate_alpha(
    ordered_df: pd.DataFrame,
    levels: Tensor,
    labels: Tensor,
    alpha: Tuple[float, float, float],
) -> Dict[str, float]:
    scores = scores_for_alpha(levels, alpha)
    metrics = ranking_metrics(ordered_df, scores)

    return {
        "alpha1": alpha[0],
        "alpha2": alpha[1],
        "alpha3": alpha[2],
        "alpha_sum": float(sum(alpha)),
        "top1_accuracy": metrics["top1_accuracy"],
        "mrr": metrics["mrr"],
        "ndcg5": metrics["ndcg@5"],
        "auc": metrics["auc"],
        "loss": ce_loss(scores, labels),
        "num_impressions": metrics["num_impressions"],
    }


# ---------------------------------------------------------------- sanity check


def load_priority0_reference(priority0_dir: Path) -> Optional[Dict[str, float]]:
    summary_path = priority0_dir / "summary.csv"

    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary.columns = [c.strip().lstrip("﻿") for c in summary.columns]

    row = summary[summary["score_type"] == "S123"]

    if row.empty:
        return None

    row = row.iloc[0]

    return {
        "top1_accuracy": float(row["top1_accuracy"]),
        "mrr": float(row["mrr"]),
        "ndcg5": float(row["ndcg5"]),
        "auc": float(row["auc"]),
        "preference_loss": float(row["preference_loss"]),
        "num_impressions": int(row["num_impressions"]),
    }


def sanity_check_epoch0(
    weights: LevelWeights,
    epoch0: Dict[str, float],
    reference: Optional[Dict[str, float]],
    s123_column: Optional[np.ndarray],
    epoch0_scores: Tensor,
) -> Dict[str, Any]:
    section("Sanity check — epoch 0의 alpha = [1,1,1]이 Priority 0 S123를 재현하는가")

    result: Dict[str, Any] = {"checks": [], "all_passed": True}

    def record(name: str, detail: str, passed: bool) -> None:
        result["checks"].append({"name": name, "detail": detail, "passed": passed})
        result["all_passed"] = result["all_passed"] and passed
        print(f"  {'OK  ' if passed else 'FAIL'}  {name:<34} {detail}")

    # 1. alpha가 정확히 [1,1,1]인가
    alpha = weights.alpha().detach().cpu().numpy()
    alpha_dev = float(np.max(np.abs(alpha - 1.0)))
    record(
        "alpha == [1, 1, 1]",
        f"alpha = [{alpha[0]:.17g}, {alpha[1]:.17g}, {alpha[2]:.17g}]  "
        f"최대편차 {alpha_dev:.3e}",
        alpha_dev <= ALPHA_EXACT_TOLERANCE,
    )

    # 2. score가 Priority 0의 S123 컬럼과 같은가
    if s123_column is not None:
        got = epoch0_scores.detach().cpu().numpy().reshape(-1)
        score_dev = float(np.max(np.abs(got - s123_column)))
        record(
            "score == Priority 0 S123",
            f"최대편차 {score_dev:.3e}",
            score_dev <= SANITY_SCORE_TOLERANCE,
        )
    else:
        record(
            "score == Priority 0 S123",
            "S123 컬럼이 없어 확인하지 못했습니다",
            False,
        )

    # 3. metric이 Priority 0 summary.csv의 S123와 같은가
    if reference is None:
        record(
            "metric == Priority 0 S123",
            "summary.csv를 찾지 못했습니다",
            False,
        )
    else:
        pairs = [
            ("Top-1", "top1_accuracy"),
            ("MRR", "mrr"),
            ("nDCG@5", "ndcg5"),
            ("AUC", "auc"),
        ]

        deltas = []

        for display, key in pairs:
            delta = abs(epoch0[key] - reference[key])
            deltas.append((display, delta, epoch0[key], reference[key]))

        worst = max(deltas, key=lambda d: d[1])

        record(
            "metric == Priority 0 S123",
            f"최대편차 {worst[1]:.3e} ({worst[0]})",
            worst[1] <= SANITY_METRIC_TOLERANCE,
        )

        print()
        print(f"    {'지표':<10} {'epoch 0':>14} {'Priority 0':>14} {'차이':>12}")
        print("    " + "-" * 54)

        for display, delta, got, want in deltas:
            print(f"    {display:<10} {got:>14.10f} {want:>14.10f} {delta:>12.3e}")

        # impression 수
        record(
            "impression 수 일치",
            f"{int(epoch0['num_impressions']):,} vs {reference['num_impressions']:,}",
            int(epoch0["num_impressions"]) == reference["num_impressions"],
        )

    print()

    if result["all_passed"]:
        print("  => epoch 0이 Priority 0 S123를 재현합니다. 학습을 시작합니다.")
    else:
        print("  => 재현되지 않습니다. 학습을 시작하지 않습니다.")
        print("     checkpoint, validation 파일, gin binding을 먼저 확인하세요.")

    return result


# ---------------------------------------------------------------- 학습


def alpha_tuple(weights: LevelWeights) -> Tuple[float, float, float]:
    a = weights.alpha().detach().cpu().numpy()
    return (float(a[0]), float(a[1]), float(a[2]))


def theta_list(weights: LevelWeights) -> List[float]:
    return [float(t) for t in weights.theta.detach().cpu().numpy()]


def train_alpha(
    weights: LevelWeights,
    train_levels: Tensor,
    train_labels: Tensor,
    val_ordered: pd.DataFrame,
    val_levels: Tensor,
    val_labels: Tensor,
    epoch0: Dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[str]]:
    section("학습")

    num_impressions = train_levels.shape[0]
    steps_per_epoch = math.ceil(num_impressions / args.batch_size)

    print(f"  train impression   : {num_impressions:,}")
    print(f"  batch size         : {args.batch_size}")
    print(f"  epoch당 step       : {steps_per_epoch:,}")
    print(f"  learning rate      : {args.lr}")
    print(f"  weight decay       : 0")
    print(f"  max epoch          : {args.max_epochs}")
    print(f"  patience           : {args.patience}  (Validation Top-1 기준)")
    print(f"  첫 epoch step log  : {args.step_log_interval} step마다")
    print()

    optimizer = torch.optim.AdamW(
        weights.parameters(), lr=args.lr, weight_decay=0.0
    )

    loss_fn = TransformerLoss().to(device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    history: List[Dict[str, Any]] = []
    step_log: List[Dict[str, Any]] = []
    warnings: List[str] = []

    a0 = alpha_tuple(weights)

    history.append({
        "epoch": 0,
        "alpha1": a0[0], "alpha2": a0[1], "alpha3": a0[2],
        "theta1": theta_list(weights)[0],
        "theta2": theta_list(weights)[1],
        "theta3": theta_list(weights)[2],
        "train_loss": ce_loss(scores_for_alpha(train_levels, a0), train_labels),
        "validation_loss": epoch0["loss"],
        "validation_top1": epoch0["top1_accuracy"],
        "mrr": epoch0["mrr"],
        "ndcg5": epoch0["ndcg5"],
        "auc": epoch0["auc"],
        "is_best": True,
        "note": "학습 전 (V1과 동일)",
    })

    best = {
        "epoch": 0,
        "alpha": a0,
        "theta": theta_list(weights),
        "metrics": epoch0,
        "train_loss": history[0]["train_loss"],
    }

    best_top1 = epoch0["top1_accuracy"]
    best_loss = epoch0["loss"]
    epochs_without_improvement = 0

    header = (
        f"  {'epoch':>5} {'alpha1':>8} {'alpha2':>8} {'alpha3':>8} "
        f"{'train loss':>11} {'val loss':>10} {'val Top-1':>10} "
        f"{'MRR':>8} {'nDCG@5':>8} {'AUC':>8}   "
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    print(
        f"  {0:>5} {a0[0]:>8.4f} {a0[1]:>8.4f} {a0[2]:>8.4f} "
        f"{history[0]['train_loss']:>11.6f} {epoch0['loss']:>10.6f} "
        f"{epoch0['top1_accuracy']:>10.6f} {epoch0['mrr']:>8.4f} "
        f"{epoch0['ndcg5']:>8.4f} {epoch0['auc']:>8.4f}   시작"
    )

    previous_alpha = a0
    stopped_early = False
    global_step = 0

    for epoch in range(1, args.max_epochs + 1):
        weights.train()

        permutation = torch.randperm(
            num_impressions, generator=generator
        ).to(device)

        train_loss_sum = 0.0
        train_seen = 0

        for start in range(0, num_impressions, args.batch_size):
            batch_idx = permutation[start:start + args.batch_size]

            batch_levels = train_levels[batch_idx]
            batch_labels = train_labels[batch_idx]

            scores = weights(batch_levels)

            loss_output = loss_fn(
                candidate_scores=scores.float(),
                candidate_labels=batch_labels.float(),
            )

            optimizer.zero_grad(set_to_none=True)
            loss_output.total_loss.backward()
            optimizer.step()

            batch_size = batch_idx.shape[0]
            loss_value = float(loss_output.total_loss.detach())

            if not math.isfinite(loss_value):
                message = (
                    f"epoch {epoch} step {global_step + 1}에서 "
                    f"loss가 유한하지 않습니다: {loss_value}"
                )
                warnings.append(message)
                print(f"\n  !! {message}")
                stopped_early = True
                break

            train_loss_sum += loss_value * batch_size
            train_seen += batch_size
            global_step += 1

            # 첫 epoch은 step 단위로도 alpha를 남긴다.
            # alpha가 3개뿐이라 1 epoch 안에 수렴할 수 있다.
            if epoch == 1 and (
                global_step == 1
                or global_step % args.step_log_interval == 0
                or start + args.batch_size >= num_impressions
            ):
                a = alpha_tuple(weights)
                step_log.append({
                    "epoch": epoch,
                    "step": global_step,
                    "alpha1": a[0], "alpha2": a[1], "alpha3": a[2],
                    "theta1": theta_list(weights)[0],
                    "theta2": theta_list(weights)[1],
                    "theta3": theta_list(weights)[2],
                    "batch_loss": loss_value,
                })

        if stopped_early:
            break

        alpha = alpha_tuple(weights)

        if not all(math.isfinite(a) for a in alpha):
            message = f"epoch {epoch}에서 alpha가 유한하지 않습니다: {alpha}"
            warnings.append(message)
            print(f"\n  !! {message}")
            break

        jump = max(abs(alpha[i] - previous_alpha[i]) for i in range(NUM_LEVELS))

        if jump > ALPHA_JUMP_WARN:
            message = (
                f"epoch {epoch}에서 alpha가 한 번에 {jump:.4f} 움직였습니다 "
                f"(경고 기준 {ALPHA_JUMP_WARN})"
            )
            warnings.append(message)

        previous_alpha = alpha

        weights.eval()

        val_scores = weights(val_levels)
        metrics = ranking_metrics(val_ordered, val_scores)

        current = {
            "top1_accuracy": metrics["top1_accuracy"],
            "mrr": metrics["mrr"],
            "ndcg5": metrics["ndcg@5"],
            "auc": metrics["auc"],
            "loss": ce_loss(val_scores, val_labels),
            "num_impressions": metrics["num_impressions"],
            "alpha1": alpha[0], "alpha2": alpha[1], "alpha3": alpha[2],
            "alpha_sum": float(sum(alpha)),
        }

        train_loss = train_loss_sum / train_seen if train_seen else float("nan")

        # best 선택 기준은 기존 V1과 같은 Validation Top-1.
        # 동률이면 Validation loss가 낮은 쪽을 쓴다.
        improved = (
            current["top1_accuracy"] > best_top1
            or (
                current["top1_accuracy"] == best_top1
                and current["loss"] < best_loss
            )
        )

        if improved:
            best_top1 = current["top1_accuracy"]
            best_loss = current["loss"]
            epochs_without_improvement = 0
            best = {
                "epoch": epoch,
                "alpha": alpha,
                "theta": theta_list(weights),
                "metrics": current,
                "train_loss": train_loss,
            }
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "alpha1": alpha[0], "alpha2": alpha[1], "alpha3": alpha[2],
            "theta1": theta_list(weights)[0],
            "theta2": theta_list(weights)[1],
            "theta3": theta_list(weights)[2],
            "train_loss": train_loss,
            "validation_loss": current["loss"],
            "validation_top1": current["top1_accuracy"],
            "mrr": current["mrr"],
            "ndcg5": current["ndcg5"],
            "auc": current["auc"],
            "is_best": improved,
            "note": "",
        })

        marker = "best" if improved else f"{epochs_without_improvement}/{args.patience}"

        print(
            f"  {epoch:>5} {alpha[0]:>8.4f} {alpha[1]:>8.4f} {alpha[2]:>8.4f} "
            f"{train_loss:>11.6f} {current['loss']:>10.6f} "
            f"{current['top1_accuracy']:>10.6f} {current['mrr']:>8.4f} "
            f"{current['ndcg5']:>8.4f} {current['auc']:>8.4f}   {marker}",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            print()
            print(
                f"  Validation Top-1이 {args.patience} epoch 동안 "
                f"좋아지지 않아 멈춥니다."
            )
            stopped_early = True
            break

    if not stopped_early:
        print()
        print(f"  max epoch {args.max_epochs}에 도달했습니다.")

    print()
    print(f"  best epoch : {best['epoch']}")
    print(
        f"  best alpha : [{best['alpha'][0]:.6f}, "
        f"{best['alpha'][1]:.6f}, {best['alpha'][2]:.6f}]  "
        f"(합 {sum(best['alpha']):.6f})"
    )

    return history, step_log, best, warnings


# ---------------------------------------------------------------- 보고


def print_step_log(step_log: List[Dict[str, Any]]) -> None:
    if not step_log:
        return

    section("첫 epoch의 step별 alpha 변화")

    print(
        f"  {'step':>7} {'alpha1':>9} {'alpha2':>9} {'alpha3':>9} "
        f"{'batch loss':>11}"
    )
    print("  " + "-" * 50)

    for row in step_log:
        print(
            f"  {row['step']:>7,} {row['alpha1']:>9.5f} {row['alpha2']:>9.5f} "
            f"{row['alpha3']:>9.5f} {row['batch_loss']:>11.6f}"
        )

    first, last = step_log[0], step_log[-1]

    print()
    print(
        f"  첫 epoch 동안 alpha1 {first['alpha1']:.5f} -> {last['alpha1']:.5f}, "
        f"alpha2 {first['alpha2']:.5f} -> {last['alpha2']:.5f}, "
        f"alpha3 {first['alpha3']:.5f} -> {last['alpha3']:.5f}"
    )


def comparison_table(
    val_ordered: pd.DataFrame,
    val_levels: Tensor,
    val_labels: Tensor,
    best_alpha: Tuple[float, float, float],
) -> List[Dict[str, Any]]:
    section("비교 — Validation")

    configs = [
        ("기존 V1", V1_ALPHA, True),
        ("coarse grid best", COARSE_BEST, False),
        ("coarse best rescaled", COARSE_BEST_RESCALED, True),
        ("learnable alpha", best_alpha, True),
    ]

    rows: List[Dict[str, Any]] = []

    for name, alpha, loss_comparable in configs:
        row = evaluate_alpha(val_ordered, val_levels, val_labels, alpha)
        row["config"] = name
        row["loss_comparable"] = loss_comparable
        rows.append(row)

    print(
        f"  {'구분':<22} {'alpha':<26} {'합':>6} {'Top-1':>10} "
        f"{'MRR':>9} {'nDCG@5':>9} {'AUC':>9} {'val loss':>11}"
    )
    print("  " + "-" * 116)

    for row in rows:
        alpha_text = (
            f"[{row['alpha1']:.4f}, {row['alpha2']:.4f}, {row['alpha3']:.4f}]"
        )
        marker = "" if row["loss_comparable"] else "  *"
        print(
            f"  {row['config']:<22} {alpha_text:<26} {row['alpha_sum']:>6.2f} "
            f"{row['top1_accuracy']:>10.6f} {row['mrr']:>9.4f} "
            f"{row['ndcg5']:>9.4f} {row['auc']:>9.4f} {row['loss']:>11.6f}{marker}"
        )

    print()
    print("  * coarse grid best는 alpha 합이 1.6이라 CE loss를 합 3짜리와")
    print("    직접 비교할 수 없습니다. 전체 scale이 softmax temperature로")
    print("    들어가기 때문입니다. loss 비교는 합 3인 세 행끼리 하세요.")

    # 순위 지표는 rescale에 영향을 받지 않아야 한다
    original = rows[1]
    rescaled = rows[2]

    keys = [("Top-1", "top1_accuracy"), ("MRR", "mrr"),
            ("nDCG@5", "ndcg5"), ("AUC", "auc")]

    deltas = [(name, abs(original[key] - rescaled[key])) for name, key in keys]
    worst = max(deltas, key=lambda d: d[1])

    print()

    if worst[1] == 0.0:
        print("  scale invariance 확인: coarse best와 rescaled의 순위 지표가")
        print("  완전히 같습니다. 양수배가 순위를 바꾸지 않는다는 뜻입니다.")
    else:
        print(f"  !! coarse best와 rescaled의 순위 지표가 다릅니다 "
              f"(최대 {worst[1]:.3e}, {worst[0]}).")
        print("     양수배는 순위를 바꾸지 않아야 하므로 원인을 확인하세요.")

    print()
    print("  loss 비교 (합 = 3인 것끼리):")

    comparable = [r for r in rows if r["loss_comparable"]]
    baseline = comparable[0]

    for row in comparable:
        delta = row["loss"] - baseline["loss"]
        print(
            f"    {row['config']:<22} loss {row['loss']:.6f}  "
            f"V1 대비 {delta:+.6f}"
        )

    return rows


def write_readme(
    path: Path,
    history: List[Dict[str, Any]],
    step_log: List[Dict[str, Any]],
    best: Dict[str, Any],
    comparison: List[Dict[str, Any]],
    sanity: Dict[str, Any],
    warnings: List[str],
    metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines: List[str] = []
    add = lines.append

    add("# learnable level weights")
    add("")
    add("기존 V1의 `S = L1 + L2 + L3`를 `S = a1*L1 + a2*L2 + a3*L3`로 바꾸고")
    add("alpha 3개만 Train에서 학습했다. Transformer는 전부 freeze했다.")
    add("")
    add("    alpha = 3.0 * softmax(theta),  theta 초기값 [0, 0, 0] -> alpha [1, 1, 1]")
    add("")
    add("따라서 alpha는 항상 양수이고 합이 3이다.")
    add("")

    add("## 실행 정보")
    add("")
    add(f"- checkpoint    : `{metadata['checkpoint']}`")
    add(f"- train         : `{metadata['train_path']}`")
    add(f"- validation    : `{metadata['validation_path']}`")
    add(f"- train cache   : `{metadata['train_cache']}`")
    add(f"- val cache     : `{metadata['validation_cache']}`")
    add(f"- batch size    : {args.batch_size}")
    add(f"- alpha lr      : {args.lr}  (AdamW, weight decay 0)")
    add(f"- max epoch     : {args.max_epochs}, patience {args.patience}")
    add(f"- best 선택     : Validation Top-1 (기존 V1과 동일)")
    add(f"- git commit    : {metadata.get('git_commit')}")
    add("")
    add("L1/L2/L3는 `model.eval()` + `no_grad` + fp32로 한 번만 뽑아 cache로 뒀다.")
    add("dropout은 꺼져 있다. Transformer가 freeze이고 loss가 candidate score")
    add("5개만의 함수이므로 이 방식은 근사가 아니라 동일하다.")
    add("")

    add("## Sanity check — epoch 0")
    add("")
    add("| 확인 | 내용 | 판정 |")
    add("|---|---|---|")

    for check in sanity["checks"]:
        verdict = "OK" if check["passed"] else "FAIL"
        add(f"| {check['name']} | {check['detail']} | {verdict} |")

    add("")

    if sanity["all_passed"]:
        add("epoch 0의 alpha = [1,1,1]이 Priority 0의 S123를 재현했다.")
    else:
        add("**재현되지 않았다.** 아래 결과를 해석하면 안 된다.")

    add("")

    add("## 학습된 alpha")
    add("")
    add(f"- best epoch : {best['epoch']}")
    add(f"- alpha1 : {best['alpha'][0]:.6f}")
    add(f"- alpha2 : {best['alpha'][1]:.6f}")
    add(f"- alpha3 : {best['alpha'][2]:.6f}")
    add(f"- 합     : {sum(best['alpha']):.6f}")
    add(f"- theta  : [{best['theta'][0]:.6f}, {best['theta'][1]:.6f}, "
        f"{best['theta'][2]:.6f}]")
    add("")
    add("theta는 전체에 상수를 더해도 alpha가 같다. 해석은 alpha로 한다.")
    add("")

    add("## 비교 — Validation")
    add("")
    add("| 구분 | alpha | 합 | Top-1 | MRR | nDCG@5 | AUC | val loss |")
    add("|---|---|---|---|---|---|---|---|")

    for row in comparison:
        alpha_text = (
            f"[{row['alpha1']:.4f}, {row['alpha2']:.4f}, {row['alpha3']:.4f}]"
        )
        mark = "" if row["loss_comparable"] else " *"
        add(
            f"| {row['config']} | {alpha_text} | {row['alpha_sum']:.2f} | "
            f"{row['top1_accuracy']:.6f} | {row['mrr']:.4f} | "
            f"{row['ndcg5']:.4f} | {row['auc']:.4f} | {row['loss']:.6f}{mark} |"
        )

    add("")
    add("`*` coarse grid best는 alpha 합이 1.6이라 CE loss를 합 3짜리와 직접")
    add("비교할 수 없다. 전체 scale이 softmax temperature로 들어가기 때문이다.")
    add("순위 지표는 scale에 영향을 받지 않으므로 coarse best와 rescaled가")
    add("같은 값이어야 한다.")
    add("")

    add("## epoch별 alpha")
    add("")
    add("| epoch | alpha1 | alpha2 | alpha3 | train loss | val loss | val Top-1 |")
    add("|---|---|---|---|---|---|---|")

    shown = history if len(history) <= 25 else history[:12] + history[-12:]

    for i, row in enumerate(shown):
        if len(history) > 25 and i == 12:
            add("| ... | ... | ... | ... | ... | ... | ... |")

        add(
            f"| {row['epoch']} | {row['alpha1']:.5f} | {row['alpha2']:.5f} | "
            f"{row['alpha3']:.5f} | {row['train_loss']:.6f} | "
            f"{row['validation_loss']:.6f} | {row['validation_top1']:.6f} |"
        )

    add("")
    add("전체는 `training_history.csv`에 있다.")
    add("")

    if step_log:
        first, last = step_log[0], step_log[-1]
        add("## 첫 epoch의 step별 변화")
        add("")
        add(f"- alpha1 {first['alpha1']:.5f} -> {last['alpha1']:.5f}")
        add(f"- alpha2 {first['alpha2']:.5f} -> {last['alpha2']:.5f}")
        add(f"- alpha3 {first['alpha3']:.5f} -> {last['alpha3']:.5f}")
        add("")
        add(f"{args.step_log_interval} step 간격으로 기록했다. "
            f"전체는 `step_log.csv`에 있다.")
        add("")

    add("## NaN / 급변")
    add("")

    if warnings:
        for message in warnings:
            add(f"- {message}")
    else:
        add(f"없음. loss와 alpha가 전 구간에서 유한했고, alpha가 한 epoch에")
        add(f"{ALPHA_JUMP_WARN} 넘게 움직인 적도 없다.")

    add("")

    add("## 하지 않은 것")
    add("")
    add("- Transformer / RQ-VAE 재학습, Transformer parameter 갱신")
    add("- checkpoint 생성, SID 재생성")
    add("- Test 데이터 사용")
    add("- Priority 2")
    add("- fine grid")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="frozen Transformer 위에서 level weight alpha만 학습한다."
    )
    parser.add_argument(
        "--priority0-dir",
        default="sweep_out/ebnerd_v2/priority0_shuffled/seed42",
        help="Priority 0 결과 폴더. checkpoint / validation 경로 / S123 기준값을 여기서 읽는다.",
    )
    parser.add_argument(
        "--out",
        default="sweep_out/ebnerd_v2/learnable_level_weights/seed42",
    )
    parser.add_argument("--config", default="configs/transformer_ebnerd.gin")
    parser.add_argument(
        "--train-path",
        default="datasets/ebnerd/train_sequences_1pos4neg.parquet",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="생략하면 Priority 0 run_metadata.json의 checkpoint를 쓴다.",
    )
    parser.add_argument(
        "--train-cache", default=None,
        help="이미 뽑아둔 train L1/L2/L3 parquet. 있으면 추론을 건너뛴다.",
    )
    parser.add_argument(
        "--gin-binding", action="append", default=[], metavar="BINDING",
        help=(
            "gin override를 직접 지정한다. 주면 Priority 0 metadata의 "
            "설정 대신 이 값만 쓴다."
        ),
    )
    parser.add_argument(
        "--skip-extractor-check", action="store_true",
        help="validation을 다시 추론해 Priority 0과 대조하는 단계를 건너뛴다.",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--step-log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    started_at = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    priority0_dir = resolve_path(args.priority0_dir)
    out_dir = resolve_path(args.out)
    config_path = resolve_path(args.config)
    train_path = resolve_path(args.train_path)

    section("learnable level weights")

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"출력 폴더가 비어 있지 않습니다. 덮어쓰지 않습니다: {out_dir}")
        return 1

    if not priority0_dir.exists():
        print(f"Priority 0 폴더가 없습니다: {priority0_dir}")
        return 1

    meta_path = priority0_dir / "run_metadata.json"

    if not meta_path.exists():
        print(f"Priority 0 run_metadata.json이 없습니다: {meta_path}")
        return 1

    p0_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    checkpoint_path = resolve_path(
        args.checkpoint if args.checkpoint else p0_meta["checkpoint"]
    )
    validation_path = resolve_path(p0_meta["dataset"])
    bindings, binding_source = bindings_from_priority0(p0_meta, args.gin_binding)

    scores_path = priority0_dir / "candidate_scores.parquet"

    if not scores_path.exists():
        print(f"Priority 0 candidate_scores.parquet이 없습니다: {scores_path}")
        return 1

    if not checkpoint_path.exists():
        print(f"checkpoint가 없습니다: {checkpoint_path}")
        return 1

    if not train_path.exists():
        print(f"train parquet이 없습니다: {train_path}")
        return 1

    device = get_device()

    print(f"  checkpoint  : {checkpoint_path}")
    print(f"  train       : {train_path}")
    print(f"  validation  : {validation_path}")
    print(f"  Priority 0  : {priority0_dir}")
    print(f"  출력        : {out_dir}")
    print(f"  device      : {device}")
    print(f"  gin binding : {len(bindings)}개  (출처: {binding_source})")

    for binding in bindings:
        print(f"      {binding}")
    print()
    print("  Transformer를 재학습하지 않습니다. Test를 쓰지 않습니다.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- validation cache: Priority 0 결과를 그대로 쓴다
    section("cache 준비")

    val_cache = pd.read_parquet(scores_path)

    required = [GROUP_COLUMN, "candidate_index", "label", "L1", "L2", "L3"]
    missing = [c for c in required if c not in val_cache.columns]

    if missing:
        print(f"Priority 0 candidate_scores.parquet에 컬럼이 없습니다: {missing}")
        return 1

    print(f"  validation cache : Priority 0 결과 재사용  "
          f"({len(val_cache):,} row)")

    val_ordered = val_cache.sort_values(
        [GROUP_COLUMN, "candidate_index"], kind="stable"
    ).reset_index(drop=True)

    s123_column = (
        val_ordered["S123"].to_numpy(dtype=np.float64)
        if "S123" in val_ordered.columns else None
    )

    val_levels, val_labels = as_tensors(val_ordered, device)

    # ---- extractor 검증: 같은 방식으로 validation을 다시 뽑아 대조한다
    extractor_check: Dict[str, Any] = {"ran": False}

    if not args.skip_extractor_check:
        print()
        print("  train cache를 만들 추출기가 Priority 0과 같은 값을 내는지")
        print("  validation으로 먼저 확인합니다.")

        recomputed = make_cache(
            config_path, bindings, checkpoint_path, validation_path,
            device, args.predict_batch_size, args.num_workers, "validation",
        )

        recomputed = recomputed.sort_values(
            [GROUP_COLUMN, "candidate_index"], kind="stable"
        ).reset_index(drop=True)

        if len(recomputed) != len(val_ordered):
            print(
                f"  row 수가 다릅니다: 재추출 {len(recomputed):,} vs "
                f"Priority 0 {len(val_ordered):,}"
            )
            return 1

        deviations = {
            level: float(np.max(np.abs(
                recomputed[level].to_numpy() - val_ordered[level].to_numpy()
            )))
            for level in ("L1", "L2", "L3")
        }

        worst = max(deviations.values())

        print()
        for level, dev in deviations.items():
            print(f"    {level} 최대편차 {dev:.3e}")

        extractor_check = {
            "ran": True,
            "deviations": deviations,
            "passed": worst <= SANITY_SCORE_TOLERANCE,
        }

        if worst > SANITY_SCORE_TOLERANCE:
            print()
            print("  추출기가 Priority 0과 다른 값을 냅니다. 진행하지 않습니다.")
            print("  checkpoint, gin binding, validation 파일을 확인하세요.")
            return 1

        print()
        print("  일치합니다. 같은 방식으로 train을 뽑습니다.")

    # ---- train cache
    print()

    train_cache_path = out_dir / "train_level_scores.parquet"

    if args.train_cache:
        given = resolve_path(args.train_cache)

        if not given.exists():
            print(f"train cache가 없습니다: {given}")
            return 1

        print(f"  train cache 재사용: {given}")
        train_cache = pd.read_parquet(given)
        train_cache_path = given
    else:
        print("  train L1/L2/L3 추출 중...")
        train_cache = make_cache(
            config_path, bindings, checkpoint_path, train_path,
            device, args.predict_batch_size, args.num_workers, "train",
        )
        train_cache.to_parquet(train_cache_path, index=False)
        print(f"  저장: {train_cache_path}  ({len(train_cache):,} row)")

    train_ordered = train_cache.sort_values(
        [GROUP_COLUMN, "candidate_index"], kind="stable"
    ).reset_index(drop=True)

    train_levels, train_labels = as_tensors(train_ordered, device)

    # ---- epoch 0 sanity check
    weights = LevelWeights().to(device)

    weights.eval()
    epoch0_scores = weights(val_levels)
    epoch0_metrics = ranking_metrics(val_ordered, epoch0_scores)

    epoch0 = {
        "top1_accuracy": epoch0_metrics["top1_accuracy"],
        "mrr": epoch0_metrics["mrr"],
        "ndcg5": epoch0_metrics["ndcg@5"],
        "auc": epoch0_metrics["auc"],
        "loss": ce_loss(epoch0_scores, val_labels),
        "num_impressions": epoch0_metrics["num_impressions"],
    }

    reference = load_priority0_reference(priority0_dir)

    sanity = sanity_check_epoch0(
        weights, epoch0, reference, s123_column, epoch0_scores
    )

    if not sanity["all_passed"]:
        (out_dir / "sanity_check_failed.json").write_text(
            json.dumps(
                {"sanity_check": sanity, "epoch0": epoch0,
                 "reference": reference, "extractor_check": extractor_check},
                ensure_ascii=False, indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print()
        print(f"  기록: {out_dir / 'sanity_check_failed.json'}")
        return 1

    # ---- 학습
    history, step_log, best, warnings = train_alpha(
        weights, train_levels, train_labels,
        val_ordered, val_levels, val_labels,
        epoch0, args, device,
    )

    print_step_log(step_log)

    comparison = comparison_table(
        val_ordered, val_levels, val_labels, best["alpha"]
    )

    if warnings:
        section("경고")
        for message in warnings:
            print(f"  - {message}")

    # ---- 저장
    section("저장")

    write_csv(out_dir / "training_history.csv", history)

    if step_log:
        write_csv(out_dir / "step_log.csv", step_log)

    write_csv(
        out_dir / "comparison.csv",
        [{k: v for k, v in row.items()} for row in comparison],
    )

    (out_dir / "best_alpha.json").write_text(
        json.dumps({
            "alpha1": best["alpha"][0],
            "alpha2": best["alpha"][1],
            "alpha3": best["alpha"][2],
            "alpha_sum": float(sum(best["alpha"])),
            "theta": best["theta"],
            "best_epoch": best["epoch"],
            "parameterization": f"alpha = {ALPHA_SUM} * softmax(theta)",
            "theta_note": (
                "theta는 전체에 상수를 더해도 alpha가 같다. "
                "해석은 alpha로 한다."
            ),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_dir / "best_metrics.json").write_text(
        json.dumps({
            "best_epoch": best["epoch"],
            "alpha": list(best["alpha"]),
            "train_loss": best["train_loss"],
            "validation": best["metrics"],
            "selection_criterion": "validation top1_accuracy",
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    (out_dir / "final_metrics.json").write_text(
        json.dumps({
            "final_epoch": history[-1]["epoch"],
            "final_alpha": [
                history[-1]["alpha1"], history[-1]["alpha2"], history[-1]["alpha3"],
            ],
            "final_train_loss": history[-1]["train_loss"],
            "final_validation_loss": history[-1]["validation_loss"],
            "final_validation_top1": history[-1]["validation_top1"],
            "final_mrr": history[-1]["mrr"],
            "final_ndcg5": history[-1]["ndcg5"],
            "final_auc": history[-1]["auc"],
            "num_epochs_run": len(history) - 1,
            "comparison": comparison,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    metadata = {
        "experiment": "learnable_level_weights",
        "checkpoint": str(checkpoint_path),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "priority0_dir": str(priority0_dir),
        "train_cache": str(train_cache_path),
        "validation_cache": str(scores_path),
        "config": str(config_path),
        "gin_bindings": bindings,
        "gin_binding_source": binding_source,
        "parameterization": f"alpha = {ALPHA_SUM} * softmax(theta), theta0 = [0,0,0]",
        "constraint": "alpha > 0, alpha1 + alpha2 + alpha3 = 3",
        "transformer_frozen": True,
        "trainable_parameters": int(
            sum(p.numel() for p in weights.parameters() if p.requires_grad)
        ),
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "selection_criterion": "validation top1_accuracy",
        "seed": args.seed,
        "level_extraction": "model.eval() + no_grad + fp32, dropout off",
        "extractor_check": extractor_check,
        "sanity_check": sanity,
        "warnings": warnings,
        "train_impressions": int(train_levels.shape[0]),
        "validation_impressions": int(val_levels.shape[0]),
        "did_not_do": [
            "Transformer 재학습",
            "Transformer parameter 갱신",
            "RQ-VAE 재학습",
            "SID 재생성",
            "Test 사용",
            "Priority 2",
            "fine grid",
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
        history, step_log, best, comparison, sanity, warnings, metadata, args,
    )

    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {out_dir / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
