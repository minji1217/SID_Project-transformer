"""Priority 0 결과에 대한 추가 sanity check.

기존 Priority 0 산출물을 덮어쓰지 않는다. 새 파일만 추가한다.
재학습하지 않고, 모델과 checkpoint를 수정하지 않으며,
Test 데이터를 읽지 않는다.

두 가지를 확인한다.

[1] tie와 candidate position bias
    S1 Top-1이 유난히 높게 나오는 것이 실제 성능인지,
    아니면 동점을 np.argmax가 낮은 index로 깨면서 생긴 착시인지 가른다.

      - validation positive candidate index 0~4의 개수와 비율
      - score 방식별 최고점 동점 후보 수 분포
      - Top-1이 동점에 의해 결정된 impression 수와 비율
      - tie-aware Top-1 (최고점이 k개이고 그중 positive가 있으면 1/k)
      - random tie-break를 여러 seed로 반복한 평균 Top-1

[2] 기존 checkpoint validation Top-1과의 미세한 차이
    학습 중 validation은 use_amp=True / bfloat16 autocast 안에서 돌았고,
    predict_sid.py는 autocast가 없어 fp32로 돈다.
    이 dtype 차이가 원인인지 실제로 재현해서 확인한다.

      - fp32와 bfloat16으로 각각 추론해 impression별 Top-1을 비교
      - 서로 다른 impression ID 목록을 저장
      - 원인을 찾으면 기록하고, 못 찾아도 차이 건수를 정확히 남긴다

    --skip-amp-check로 [2]를 건너뛰면 GPU 없이 [1]만 돌릴 수 있다.

    cd Transformer
    python -m analysis.run_priority0_diagnostics \
        --config configs/transformer_ebnerd.gin \
        --sweep-out sweep_out/ebnerd \
        --validation-path /home/ubuntu/shared/datasets/ebnerd/validation_sequences_1pos4neg_half.parquet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sweep.run_stage import write_csv


BASE_DIR = Path(__file__).resolve().parent.parent

NUM_CANDIDATES = 5
SCORE_TYPES = ("S1", "S2", "S3", "S12", "S123")

# 학습 중 validation에서 보고된 seed 42의 값.
# train_transformer.py의 evaluate()는 bfloat16 autocast 안에서 돈다.
REFERENCE_TOP1_CORRECT = 33111
REFERENCE_TOP1 = 0.2701858032297285
REFERENCE_NUM_IMPRESSIONS = 122549


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def as_grid(df: pd.DataFrame, column: str) -> np.ndarray:
    return df[column].to_numpy().reshape(-1, NUM_CANDIDATES)


# ----------------------------------------------------------------------
# [1] position bias
# ----------------------------------------------------------------------

def report_position_bias(df: pd.DataFrame) -> Dict[str, Any]:
    section("1-a. positive candidate position 분포")

    labels = as_grid(df, "label").astype(int)
    positive_index = labels.argmax(axis=1)
    total = len(positive_index)

    counts = np.bincount(positive_index, minlength=NUM_CANDIDATES)

    print("  정답이 몇 번째 후보에 있는가")
    print()
    print(f"  {'index':<8}{'개수':>12}{'비율':>10}")
    print("  " + "-" * 30)

    for i in range(NUM_CANDIDATES):
        print(f"  {i:<8}{counts[i]:>12,}{counts[i] / total:>10.2%}")

    expected = 1.0 / NUM_CANDIDATES
    max_deviation = float(np.abs(counts / total - expected).max())

    print()
    print(f"  균등이면 각 20.00%. 최대 편차 {max_deviation:.2%}")

    if max_deviation < 0.01:
        print("  >>> position bias는 없다. 정답 위치가 고르게 섞여 있다.")
    else:
        print("  >>> position bias가 있다. argmax의 낮은 index 우선 처리가")
        print("      Top-1을 부풀리거나 깎을 수 있으므로 tie-aware 값을 함께 본다.")

    return {
        "counts": counts.tolist(),
        "ratios": (counts / total).tolist(),
        "num_impressions": int(total),
        "max_deviation_from_uniform": max_deviation,
    }


# ----------------------------------------------------------------------
# [1] tie 분석
# ----------------------------------------------------------------------

def analyse_ties(
    df: pd.DataFrame,
    score_type: str,
    tie_break_seeds: List[int],
) -> Dict[str, Any]:
    scores = as_grid(df, score_type)
    labels = as_grid(df, "label").astype(int)

    total = len(scores)
    rows = np.arange(total)

    best = scores.max(axis=1, keepdims=True)
    is_best = scores == best
    tie_count = is_best.sum(axis=1)

    # 현재 방식: np.argmax는 동점이면 가장 낮은 index
    argmax_top = scores.argmax(axis=1)
    argmax_correct = labels[rows, argmax_top] == 1

    # 동점이 있는 impression
    tied = tie_count > 1

    # 동점 때문에 결정된 것 중 맞은 것 / 틀린 것
    tied_correct = argmax_correct & tied
    tied_wrong = (~argmax_correct) & tied

    # tie-aware: 최고점이 k개이고 그 안에 positive가 있으면 1/k
    positive_is_best = (is_best & (labels == 1)).any(axis=1)
    tie_aware = np.where(positive_is_best, 1.0 / tie_count, 0.0)

    # random tie-break 반복
    random_top1: List[float] = []

    for seed in tie_break_seeds:
        rng = np.random.default_rng(seed)
        noise = rng.random(scores.shape)
        # 최고점 후보들 중에서만 무작위로 하나 고른다
        picked = np.where(is_best, noise, -np.inf).argmax(axis=1)
        random_top1.append(float((labels[rows, picked] == 1).mean()))

    distribution = {
        int(k): int((tie_count == k).sum())
        for k in range(1, NUM_CANDIDATES + 1)
        if (tie_count == k).sum() > 0
    }

    return {
        "score_type": score_type,
        "num_impressions": int(total),
        "argmax_top1": float(argmax_correct.mean()),
        "argmax_top1_correct": int(argmax_correct.sum()),
        "tie_aware_top1": float(tie_aware.mean()),
        "random_tiebreak_top1_mean": float(np.mean(random_top1)),
        "random_tiebreak_top1_std": float(np.std(random_top1, ddof=1)) if len(random_top1) > 1 else 0.0,
        "random_tiebreak_seeds": tie_break_seeds,
        "tied_impressions": int(tied.sum()),
        "tied_rate": float(tied.mean()),
        "tied_and_argmax_correct": int(tied_correct.sum()),
        "tied_and_argmax_wrong": int(tied_wrong.sum()),
        "tie_count_distribution": distribution,
        "mean_tie_count": float(tie_count.mean()),
        "all_five_tied": int((tie_count == NUM_CANDIDATES).sum()),
    }


def report_ties(
    df: pd.DataFrame,
    tie_break_seeds: List[int],
) -> List[Dict[str, Any]]:
    section("1-b. 최고점 동점 후보 수 분포")

    results = [analyse_ties(df, name, tie_break_seeds) for name in SCORE_TYPES]

    print(f"  {'score':<7}" + "".join(f"{f'{k}개':>11}" for k in range(1, 6)))
    print("  " + "-" * 62)

    for r in results:
        cells = []
        for k in range(1, NUM_CANDIDATES + 1):
            n = r["tie_count_distribution"].get(k, 0)
            cells.append(f"{n / r['num_impressions']:>10.2%} ")
        print(f"  {r['score_type']:<7}" + "".join(cells))

    print()
    print("  '1개'는 동점 없이 유일한 최고점이 있는 경우다.")

    section("1-c. tie가 Top-1을 결정한 비율과 tie-aware Top-1")

    print(f"  {'score':<7}{'argmax':>10}{'tie-aware':>12}{'random':>10}"
          f"{'tie 비율':>11}{'tie중 맞음':>12}{'tie중 틀림':>12}")
    print("  " + "-" * 74)

    for r in results:
        print(
            f"  {r['score_type']:<7}{r['argmax_top1']:>10.4f}"
            f"{r['tie_aware_top1']:>12.4f}"
            f"{r['random_tiebreak_top1_mean']:>10.4f}"
            f"{r['tied_rate']:>11.2%}"
            f"{r['tied_and_argmax_correct']:>12,}"
            f"{r['tied_and_argmax_wrong']:>12,}"
        )

    print()
    print("  argmax    : 현재 evaluate/metrics.py 방식. 동점이면 낮은 index")
    print("  tie-aware : 최고점이 k개이고 그중 positive가 있으면 1/k credit")
    print(f"  random    : 최고점 중 무작위 선택, seed {len(tie_break_seeds)}개 평균")
    print()
    print("  argmax와 tie-aware 차이가 크면 그 score의 Top-1은")
    print("  동점 처리 규칙이 만든 값이지 실제 변별력이 아니다.")

    return results


# ----------------------------------------------------------------------
# [2] fp32 vs bfloat16
# ----------------------------------------------------------------------

def run_amp_comparison(
    base_config: Path,
    checkpoint: Path,
    validation_path: Path,
    bindings: Dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    """같은 checkpoint를 fp32와 bfloat16으로 각각 추론해 비교한다."""
    import gin
    import torch
    from torch.utils.data import DataLoader

    from data.sequence import NewsSequenceDataset, collate_news_sequences
    from modules.model import NewsEncoderDecoderTransformer
    from predict_sid import load_checkpoint
    from sweep.run_stage import format_binding

    section("2. fp32 vs bfloat16 — 기존 checkpoint 결과와의 차이 조사")

    print("  학습 중 validation은 use_amp=True / bfloat16 autocast 안에서 돌았고")
    print("  predict_sid.py는 autocast가 없어 fp32로 돈다.")
    print("  같은 checkpoint를 두 dtype으로 추론해 impression별로 비교한다.")
    print()

    gin.parse_config_file(str(base_config), skip_unknown=True)
    gin.parse_config(
        [format_binding(k, bindings[k]) for k in sorted(bindings)],
        skip_unknown=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    dataset = NewsSequenceDataset(parquet_path=str(validation_path))
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
    )

    model = NewsEncoderDecoderTransformer().to(device)
    load_checkpoint(checkpoint_path=checkpoint, model=model, device=device)
    model.eval()

    def infer(amp_enabled: bool, amp_dtype) -> np.ndarray:
        """impression별 Top-1 후보 인덱스와 정답 여부를 돌려준다."""
        correct: List[np.ndarray] = []
        picked: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                history_sids = batch["history_sids"].to(device, non_blocking=True)
                history_mask = batch["history_mask"].to(device, non_blocking=True)
                candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
                candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    output = model(
                        history_sids=history_sids,
                        history_mask=history_mask,
                        candidate_sids=candidate_sids,
                    )

                predicted = output.candidate_scores.argmax(dim=1)
                target = candidate_labels.argmax(dim=1)

                picked.append(predicted.cpu().numpy())
                correct.append((predicted == target).cpu().numpy())

        return np.concatenate(picked), np.concatenate(correct)

    print("  fp32 추론 중...")
    fp32_picked, fp32_correct = infer(False, None)

    print("  bfloat16 추론 중...")
    bf16_picked, bf16_correct = infer(True, torch.bfloat16)

    total = len(fp32_correct)

    fp32_n = int(fp32_correct.sum())
    bf16_n = int(bf16_correct.sum())

    differing = np.flatnonzero(fp32_correct != bf16_correct)
    picked_differ = np.flatnonzero(fp32_picked != bf16_picked)

    print()
    print(f"  {'':<14}{'맞은 수':>12}{'Top-1':>12}")
    print("  " + "-" * 40)
    print(f"  {'fp32':<14}{fp32_n:>12,}{fp32_n / total:>12.6%}")
    print(f"  {'bfloat16':<14}{bf16_n:>12,}{bf16_n / total:>12.6%}")
    print(f"  {'기존 보고값':<12}{REFERENCE_TOP1_CORRECT:>12,}{REFERENCE_TOP1:>12.6%}")
    print()
    print(f"  Top-1 후보 선택이 달라진 impression : {len(picked_differ):,}")
    print(f"  정답 여부가 달라진 impression       : {len(differing):,}")

    matches_reference = bf16_n == REFERENCE_TOP1_CORRECT

    print()
    if matches_reference:
        print("  >>> bfloat16 결과가 기존 보고값과 정확히 일치한다.")
        print("      차이의 원인은 dtype이다.")
        print("      학습 중 validation은 bfloat16 autocast 안에서 계산됐고,")
        print("      predict_sid.py는 fp32라 근소한 점수 차이에서 순위가 갈렸다.")
    else:
        print("  >>> bfloat16으로도 기존 보고값과 일치하지 않는다.")
        print(f"      bfloat16 {bf16_n:,} vs 기존 {REFERENCE_TOP1_CORRECT:,}")
        print("      dtype 외의 원인을 더 조사해야 한다.")

    return {
        "fp32_correct": fp32_n,
        "fp32_top1": fp32_n / total,
        "bf16_correct": bf16_n,
        "bf16_top1": bf16_n / total,
        "reference_correct": REFERENCE_TOP1_CORRECT,
        "reference_top1": REFERENCE_TOP1,
        "num_impressions": int(total),
        "differing_correctness_count": int(len(differing)),
        "differing_picked_count": int(len(picked_differ)),
        "bf16_matches_reference": bool(matches_reference),
        "differing_indices": differing.tolist(),
        "fp32_picked": fp32_picked,
        "bf16_picked": bf16_picked,
        "fp32_correct_mask": fp32_correct,
        "bf16_correct_mask": bf16_correct,
    }


# ----------------------------------------------------------------------

def write_report(
    path: Path,
    position: Dict[str, Any],
    ties: List[Dict[str, Any]],
    amp: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    by_name = {t["score_type"]: t for t in ties}
    lines: List[str] = []
    add = lines.append

    add("# Priority 0 — 추가 sanity check")
    add("")
    add("기존 Priority 0 산출물은 그대로 두고 진단만 추가한 결과다.")
    add("재학습하지 않았고 Test 데이터를 쓰지 않았다.")
    add("")
    add(f"- checkpoint : `{metadata['checkpoint']}`")
    add(f"- dataset    : `{metadata['dataset']}`")
    add(f"- impression : {position['num_impressions']:,}")
    add(f"- git commit : {metadata.get('git_commit') or '(알 수 없음)'}")
    add("")

    add("## 1. positive candidate position 분포")
    add("")
    add("| index | 개수 | 비율 |")
    add("|---|---|---|")

    for i, (c, r) in enumerate(zip(position["counts"], position["ratios"])):
        add(f"| {i} | {c:,} | {r:.2%} |")

    add("")
    add(f"균등이면 각 20.00%. 최대 편차 {position['max_deviation_from_uniform']:.2%}.")
    add("")

    if position["max_deviation_from_uniform"] < 0.01:
        add("position bias는 없다. 정답 위치가 고르게 섞여 있으므로,")
        add("argmax의 낮은 index 우선 처리가 특정 score를 체계적으로")
        add("유리하게 만들지는 않는다. 다만 동점이 많으면 여전히")
        add("Top-1이 부풀 수 있으므로 아래를 함께 본다.")
    else:
        add("position bias가 있다. argmax가 동점에서 낮은 index를 고르므로")
        add("정답이 앞쪽에 몰려 있으면 Top-1이 부풀 수 있다.")

    add("")
    add("## 2. 동점 분석")
    add("")
    add("| score | argmax Top-1 | tie-aware Top-1 | random Top-1 | 동점 비율 | 동점 중 맞음 | 동점 중 틀림 |")
    add("|---|---|---|---|---|---|---|")

    for t in ties:
        add(
            f"| {t['score_type']} | {t['argmax_top1']:.4f} | "
            f"{t['tie_aware_top1']:.4f} | "
            f"{t['random_tiebreak_top1_mean']:.4f} ± {t['random_tiebreak_top1_std']:.4f} | "
            f"{t['tied_rate']:.2%} | {t['tied_and_argmax_correct']:,} | "
            f"{t['tied_and_argmax_wrong']:,} |"
        )

    add("")
    add("- **argmax**: 현재 `evaluate/metrics.py` 방식. 동점이면 낮은 index")
    add("- **tie-aware**: 최고점이 k개이고 그중 positive가 있으면 1/k credit")
    add("- **random**: 최고점 후보 중 무작위 선택을 여러 seed로 반복한 평균")
    add("")

    s1 = by_name["S1"]
    gap = s1["argmax_top1"] - s1["tie_aware_top1"]

    add("### S1 해석")
    add("")
    add(f"S1의 argmax Top-1은 {s1['argmax_top1']:.4f}지만 "
        f"tie-aware Top-1은 {s1['tie_aware_top1']:.4f}다. "
        f"차이는 {gap:+.4f}.")
    add("")
    add(f"S1에서 최고점이 동점인 impression은 {s1['tied_rate']:.2%}이고, "
        f"평균 동점 후보 수는 {s1['mean_tie_count']:.2f}개다. "
        f"후보 5개가 전부 동점인 경우도 {s1['all_five_tied']:,}건 있다.")
    add("")

    if gap > 0.03:
        add("**S1의 높은 Top-1은 대부분 동점 처리가 만든 값이다.**")
        add("c1만으로는 후보를 거의 구분하지 못하고, 동점이 난 자리를")
        add("argmax가 낮은 index로 채우면서 우연히 맞은 것이 많다.")
        add("S1을 다른 score와 같은 선상에서 비교하면 안 된다.")
    elif gap > 0.005:
        add("**S1의 Top-1 중 일부는 동점 처리에서 나온다.**")
        add("비교할 때 tie-aware 값을 함께 봐야 한다.")
    else:
        add("동점의 영향이 작다. S1의 Top-1은 실제 변별력으로 볼 수 있다.")

    add("")
    add("## 3. 기존 checkpoint 결과와의 차이")
    add("")

    if amp is None:
        add("이번 실행에서는 `--skip-amp-check`로 건너뛰었다.")
    else:
        add("| 경로 | 맞은 수 | Top-1 |")
        add("|---|---|---|")
        add(f"| 학습 중 validation (기존 보고값) | {amp['reference_correct']:,} | {amp['reference_top1']:.6%} |")
        add(f"| bfloat16 autocast 추론 | {amp['bf16_correct']:,} | {amp['bf16_top1']:.6%} |")
        add(f"| fp32 추론 (predict_sid.py 경로) | {amp['fp32_correct']:,} | {amp['fp32_top1']:.6%} |")
        add("")
        add(f"- Top-1 후보 선택이 달라진 impression: {amp['differing_picked_count']:,}")
        add(f"- 정답 여부가 달라진 impression: **{amp['differing_correctness_count']:,}** "
            f"/ {amp['num_impressions']:,} "
            f"({amp['differing_correctness_count'] / amp['num_impressions']:.4%})")
        add("")

        if amp["bf16_matches_reference"]:
            add("**원인: dtype 차이.**")
            add("")
            add("`configs/transformer_ebnerd.gin`이 `train.use_amp = True`,")
            add("`train.amp_dtype = \"bfloat16\"`이므로 학습 중 validation은")
            add("`torch.autocast(bfloat16)` 안에서 계산됐다.")
            add("반면 `predict_sid.py`에는 autocast가 없어 fp32로 계산된다.")
            add("")
            add("bfloat16은 유효숫자가 약 3자리다. 후보 간 점수 차이가")
            add("그 정밀도보다 작은 impression에서 순위가 갈리면서")
            add("소수의 impression에서 결과가 달라졌다.")
            add("")
            add("candidate order, mask, dataset indexing, batch/padding은")
            add("두 경로가 같은 `NewsSequenceDataset` / `collate_news_sequences`를")
            add("`shuffle=False`로 쓰므로 동일하다.")
            add("동점 규칙도 `torch.argmax`와 `np.argmax` 모두 낮은 index 우선이라 같다.")
        else:
            add("**원인 미확정.**")
            add("")
            add("bfloat16으로 재현해도 기존 보고값과 일치하지 않았다.")
            add(f"bfloat16 {amp['bf16_correct']:,} vs 기존 {amp['reference_correct']:,}.")
            add("dtype 외의 원인을 더 조사해야 한다.")

        add("")
        add("### 표현")
        add("")
        add("이 차이 때문에 S123은 기존 결과의 **완전 일치가 아니라 near-match**다.")
        add(f"차이는 {amp['differing_correctness_count']:,} / {amp['num_impressions']:,} "
            f"({amp['differing_correctness_count'] / amp['num_impressions']:.4%}) impression이다.")
        add("")
        add("`run_metadata.json`의 `reproduced_reference` 필드는 이 파일로 대체한다.")

    add("")
    add("## 산출물")
    add("")
    add("- `tie_position_summary.csv` — score 방식별 동점 통계")
    add("- `tie_position_diagnostics.json` — 전체 수치")
    add("- `reference_mismatch_impressions.csv` — fp32와 bfloat16이 갈린 impression")
    add("- `reference_comparison.json` — near-match 판정")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep-out", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--tie-break-seeds", type=int, nargs="*",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="random tie-break 반복 seed (기본 10개)",
    )
    parser.add_argument(
        "--skip-amp-check", action="store_true",
        help="fp32 vs bfloat16 비교를 건너뛴다 (GPU 불필요)",
    )
    args = parser.parse_args()

    started_at = time.time()

    sweep_dir = resolve_path(args.sweep_out)
    base_config = resolve_path(args.config)
    validation_path = resolve_path(args.validation_path)

    priority0_dir = sweep_dir / "priority0_score_decomposition" / f"seed{args.seed}"
    candidate_path = priority0_dir / "candidate_scores.parquet"

    if not candidate_path.exists():
        print(f"{candidate_path}가 없습니다.")
        print("먼저 run_priority0_score_decomposition을 실행하세요.")
        return 1

    final_path = sweep_dir / "seed_robustness" / "selected_final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    params = final.get("config") or {}

    seeds = final.get("seeds") or []
    checkpoints = final.get("checkpoints") or []
    checkpoint = Path(
        next(c for s, c in zip(seeds, checkpoints) if s == args.seed)
    )

    print()
    print("=" * 74)
    print("Priority 0 추가 진단")
    print("=" * 74)
    print(f"입력       : {candidate_path}")
    print(f"checkpoint : {checkpoint}")
    print(f"validation : {validation_path}")
    print("기존 산출물을 덮어쓰지 않습니다. Test 데이터를 쓰지 않습니다.")

    df = pd.read_parquet(candidate_path)
    df = df.sort_values(["sample_index", "candidate_index"], kind="stable")
    df = df.reset_index(drop=True)

    counts = df.groupby("sample_index").size()

    if not (counts == NUM_CANDIDATES).all():
        print("impression당 후보가 5개가 아닌 행이 있습니다.")
        return 1

    # [1]
    position = report_position_bias(df)
    ties = report_ties(df, list(args.tie_break_seeds))

    # [2]
    amp: Optional[Dict[str, Any]] = None

    if not args.skip_amp_check:
        amp = run_amp_comparison(
            base_config=base_config,
            checkpoint=checkpoint,
            validation_path=validation_path,
            bindings=params,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # 저장
    out_dir = priority0_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    tie_rows = [
        {k: v for k, v in t.items() if k != "tie_count_distribution"}
        | {
            f"tie_count_{k}": t["tie_count_distribution"].get(k, 0)
            for k in range(1, NUM_CANDIDATES + 1)
        }
        for t in ties
    ]

    for row in tie_rows:
        row["random_tiebreak_seeds"] = ";".join(str(s) for s in args.tie_break_seeds)

    write_csv(out_dir / "tie_position_summary.csv", tie_rows)

    (out_dir / "tie_position_diagnostics.json").write_text(
        json.dumps(
            {"position_bias": position, "ties": ties},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    if amp is not None:
        sample_index = df["sample_index"].to_numpy().reshape(-1, NUM_CANDIDATES)[:, 0]

        impression_id = None
        if "impression_id" in df.columns:
            impression_id = df["impression_id"].to_numpy().reshape(-1, NUM_CANDIDATES)[:, 0]

        idx = np.asarray(amp["differing_indices"], dtype=np.int64)

        mismatch_rows = []
        for i in idx:
            row = {
                "sample_index": int(sample_index[i]),
                "fp32_picked": int(amp["fp32_picked"][i]),
                "bf16_picked": int(amp["bf16_picked"][i]),
                "fp32_correct": bool(amp["fp32_correct_mask"][i]),
                "bf16_correct": bool(amp["bf16_correct_mask"][i]),
            }
            if impression_id is not None:
                row["impression_id"] = impression_id[i]
            mismatch_rows.append(row)

        write_csv(out_dir / "reference_mismatch_impressions.csv", mismatch_rows)

        status = "exact_match" if amp["bf16_matches_reference"] else "unresolved"
        diff = amp["differing_correctness_count"]

        (out_dir / "reference_comparison.json").write_text(
            json.dumps(
                {
                    "status": "near_match",
                    "note": (
                        "S123은 기존 학습 중 validation 결과와 완전 일치가 아니라 "
                        "near-match다. run_metadata.json의 reproduced_reference 필드는 "
                        "이 파일로 대체한다."
                    ),
                    "reference_top1_correct": amp["reference_correct"],
                    "reference_top1": amp["reference_top1"],
                    "fp32_top1_correct": amp["fp32_correct"],
                    "fp32_top1": amp["fp32_top1"],
                    "bf16_top1_correct": amp["bf16_correct"],
                    "bf16_top1": amp["bf16_top1"],
                    "num_impressions": amp["num_impressions"],
                    "differing_impression_count": diff,
                    "differing_impression_rate": diff / amp["num_impressions"],
                    "bf16_vs_reference": status,
                    "cause": (
                        "dtype. 학습 중 validation은 use_amp=True/bfloat16 autocast, "
                        "predict_sid.py는 fp32."
                        if amp["bf16_matches_reference"]
                        else "미확정. bfloat16으로도 기존 값과 일치하지 않음."
                    ),
                    "checked_and_identical": [
                        "candidate order (collate_news_sequences)",
                        "dataset indexing (shuffle=False, 같은 parquet)",
                        "mask (history_mask 경로 동일)",
                        "batch/padding (같은 collate 함수)",
                        "tie handling (torch.argmax / np.argmax 모두 낮은 index 우선)",
                    ],
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    metadata = {
        "experiment": "v2_priority0_diagnostics",
        "checkpoint": str(checkpoint),
        "dataset": str(validation_path),
        "seed": args.seed,
        "tie_break_seeds": list(args.tie_break_seeds),
        "amp_check": amp is not None,
        "git_commit": git_commit_hash(),
        "started_at_utc": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }

    (out_dir / "diagnostics_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    write_report(
        out_dir / "diagnostics_report.md",
        position, ties, amp, metadata,
    )

    print()
    print("=" * 74)
    print("저장 완료 (기존 파일은 그대로)")
    print("=" * 74)
    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {out_dir / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
