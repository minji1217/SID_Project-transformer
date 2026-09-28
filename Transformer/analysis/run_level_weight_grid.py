"""level별 score weight를 탐색한다. 모델을 다시 돌리지 않는다.

    S(alpha2, alpha3) = L1 + alpha2 * L2 + alpha3 * L3

alpha1은 1.0으로 고정한다. 세 alpha를 동시에 배율만 바꾸면
ranking이 같아지는 중복 조합이 생기므로, alpha1을 기준으로 두고
c2 / c3의 상대적인 영향만 본다.

이 스크립트가 하지 않는 것
  Transformer 재학습, checkpoint 생성, RQ-VAE 재학습, SID 재생성,
  candidate 재추론, Test 사용

Priority 0이 만든 candidate_scores.parquet의 L1 / L2 / L3를 그대로 쓴다.
z-score, normalize, standardize를 하지 않는다.
순수하게 기존 log-prob에 level weight만 적용했을 때의 효과를 본다.

metric은 evaluate/metrics.py의 evaluate_ranking을 그대로 쓴다.
candidate_score 컬럼만 갈아끼워 호출하므로 동점 규칙이 기존과 동일하다.

    cd Transformer
    python -m analysis.run_level_weight_grid \
        --scores sweep_out/ebnerd_v2/priority0_shuffled/seed42/candidate_scores.parquet \
        --out sweep_out/ebnerd_v2/priority0_level_weight_grid/seed42_shuffled
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluate.metrics import evaluate_ranking
from sweep.run_stage import write_csv


BASE_DIR = Path(__file__).resolve().parent.parent

NUM_CANDIDATES = 5
GROUP_COLUMN = "sample_index"

ALPHA1 = 1.0

ALPHA2_GRID = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
ALPHA3_GRID = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]

# (alpha2, alpha3) -> 기존 Priority 0의 어느 score와 같아야 하는가
BASELINES: Dict[Tuple[float, float], str] = {
    (0.0, 0.0): "S1",
    (1.0, 0.0): "S12",
    (1.0, 1.0): "S123",
}

REFERENCE_KEY = (1.0, 1.0)  # 모든 delta의 기준

# 재구성한 score가 기존 컬럼과 이만큼 안이면 같다고 본다.
# float32로 저장된 값을 float64로 다시 더하는 반올림만 허용한다.
SCORE_TOLERANCE = 1e-5
METRIC_TOLERANCE = 1e-6


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
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()


def weighted_score(df: pd.DataFrame, alpha2: float, alpha3: float) -> np.ndarray:
    return (
        df["L1"].to_numpy()
        + alpha2 * df["L2"].to_numpy()
        + alpha3 * df["L3"].to_numpy()
    ).astype(np.float64)


def group_correct(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """impression별로 Top-1이 정답이었는지.

    evaluate/metrics.py의 top1_accuracy와 같은 규칙(np.argmax)을 쓴다.
    """
    grid = scores.reshape(-1, NUM_CANDIDATES)
    label_grid = labels.reshape(-1, NUM_CANDIDATES)
    rows = np.arange(len(grid))
    return label_grid[rows, grid.argmax(axis=1)] == 1


def evaluate_weights(
    df: pd.DataFrame,
    base: pd.DataFrame,
    alpha2: float,
    alpha3: float,
) -> Dict[str, Any]:
    scores = weighted_score(df, alpha2, alpha3)
    labels = df["label"].to_numpy().astype(int)

    work = base.copy()
    work["candidate_score"] = scores

    metrics = evaluate_ranking(df=work, group_column=GROUP_COLUMN)

    grid = scores.reshape(-1, NUM_CANDIDATES)
    label_grid = labels.reshape(-1, NUM_CANDIDATES)

    best = grid.max(axis=1, keepdims=True)
    is_best = grid == best
    tie_count = is_best.sum(axis=1)
    tied = tie_count > 1

    positive_is_best = (is_best & (label_grid == 1)).any(axis=1)
    tie_aware = float(np.mean(np.where(positive_is_best, 1.0 / tie_count, 0.0)))

    positive = scores[labels == 1]
    negative = scores[labels == 0]

    # softmax 확률과 preference loss.
    # alpha를 바꾸면 score scale도 바뀌어 ranking이 같아도 값이 달라진다.
    # 보조 지표로만 본다.
    shifted = grid - grid.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)

    rows = np.arange(len(probs))
    positive_prob = probs[rows, label_grid.argmax(axis=1)]

    return {
        "alpha1": ALPHA1,
        "alpha2": alpha2,
        "alpha3": alpha3,
        "top1_accuracy": metrics["top1_accuracy"],
        "tie_aware_top1": tie_aware,
        "tie_impression_count": int(tied.sum()),
        "tie_rate": float(tied.mean()),
        "mrr": metrics["mrr"],
        "ndcg5": metrics["ndcg@5"],
        "auc": metrics["auc"],
        "positive_score_mean": float(positive.mean()),
        "negative_score_mean": float(negative.mean()),
        "score_gap": float(positive.mean() - negative.mean()),
        "positive_prob_mean": float(positive_prob.mean()),
        "preference_loss": float(
            -np.log(np.clip(positive_prob, 1e-12, None)).mean()
        ),
        "num_impressions": int(metrics["num_impressions"]),
        "_scores": scores,
    }


def sanity_check(df: pd.DataFrame, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """baseline 조합이 기존 Priority 0 결과를 재현하는지 확인한다."""
    section("Sanity check — baseline 3개가 기존 Priority 0와 같은가")

    by_key = {(r["alpha2"], r["alpha3"]): r for r in rows}
    checks: List[Dict[str, Any]] = []
    all_ok = True

    print()
    print(f"  {'조합':<22}{'대조':<7}{'score max diff':>17}"
          f"{'Top-1 diff':>13}{'MRR diff':>12}{'nDCG diff':>12}{'AUC diff':>12}{'':>6}")
    print("  " + "-" * 92)

    for key, column in BASELINES.items():
        row = by_key[key]
        label = f"(1, {key[0]:g}, {key[1]:g})"

        if column not in df.columns:
            print(f"  {label:<22}{column:<7}  파일에 {column} 컬럼이 없어 건너뜁니다.")
            continue

        # 1) score 값 자체 비교
        existing = df[column].to_numpy().astype(np.float64)
        score_diff = float(np.abs(row["_scores"] - existing).max())

        # 2) metric 비교 — 기존 컬럼으로 직접 평가해서 맞춘다
        work = df[[GROUP_COLUMN, "label"]].copy()
        work["candidate_score"] = existing
        work["label"] = work["label"].astype(int)
        reference = evaluate_ranking(df=work, group_column=GROUP_COLUMN)

        deltas = {
            "top1": row["top1_accuracy"] - reference["top1_accuracy"],
            "mrr": row["mrr"] - reference["mrr"],
            "ndcg5": row["ndcg5"] - reference["ndcg@5"],
            "auc": row["auc"] - reference["auc"],
        }

        ok = (
            score_diff <= SCORE_TOLERANCE
            and all(abs(v) <= METRIC_TOLERANCE for v in deltas.values())
        )
        all_ok = all_ok and ok

        print(f"  {label:<22}{column:<7}{score_diff:>17.3e}"
              f"{deltas['top1']:>13.2e}{deltas['mrr']:>12.2e}"
              f"{deltas['ndcg5']:>12.2e}{deltas['auc']:>12.2e}"
              f"{'  OK' if ok else '  FAIL':>6}")

        checks.append({
            "alpha2": key[0], "alpha3": key[1],
            "baseline_column": column,
            "score_max_abs_diff": score_diff,
            "metric_deltas": deltas,
            "passed": bool(ok),
            "reference_metrics": {
                "top1_accuracy": reference["top1_accuracy"],
                "mrr": reference["mrr"],
                "ndcg@5": reference["ndcg@5"],
                "auc": reference["auc"],
            },
        })

    print()

    if all_ok:
        print("  => 세 baseline이 모두 재현됩니다. grid 결과를 해석해도 됩니다.")
    else:
        print("  => 재현되지 않습니다. grid 결과를 해석하지 마세요.")
        print("     컬럼명, group 컬럼, 동점 규칙, 파일 경로를 먼저 확인해야 합니다.")

    return {"all_passed": all_ok, "checks": checks}


def print_grid(rows: List[Dict[str, Any]], reference: Dict[str, Any]) -> None:
    section("전체 42개 조합  (Top-1 내림차순)")

    ordered = sorted(rows, key=lambda r: -r["top1_accuracy"])

    print()
    print(f"  {'a2':>5}{'a3':>6}  {'Top-1':>8}{'tie-aware':>11}{'tie%':>8}"
          f"{'MRR':>9}{'nDCG@5':>9}{'AUC':>9}"
          f"{'dTop1':>9}{'dMRR':>9}{'dnDCG':>9}{'dAUC':>9}  {'':<6}")
    print("  " + "-" * 94)

    for row in ordered:
        tag = BASELINES.get((row["alpha2"], row["alpha3"]), "")

        print(
            f"  {row['alpha2']:>5.2f}{row['alpha3']:>6.2f}  "
            f"{row['top1_accuracy']:>8.4f}{row['tie_aware_top1']:>11.4f}"
            f"{row['tie_rate']:>8.2%}"
            f"{row['mrr']:>9.4f}{row['ndcg5']:>9.4f}{row['auc']:>9.4f}"
            f"{row['delta_top1']:>+9.4f}{row['delta_mrr']:>+9.4f}"
            f"{row['delta_ndcg5']:>+9.4f}{row['delta_auc']:>+9.4f}  {tag:<6}"
        )

    print()
    print(f"  delta는 모두 S123 = (1, 1, 1) 대비입니다.")
    print(f"  S123 : Top-1 {reference['top1_accuracy']:.4f} | "
          f"MRR {reference['mrr']:.4f} | nDCG@5 {reference['ndcg5']:.4f} | "
          f"AUC {reference['auc']:.4f}")


def print_top10(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    section("Top-10")

    ordered = sorted(rows, key=lambda r: -r["top1_accuracy"])[:10]

    print()
    print(f"  {'#':>3}{'a2':>7}{'a3':>6}  {'Top-1':>8}{'dTop1':>9}"
          f"{'MRR':>9}{'dMRR':>9}{'nDCG@5':>9}{'dnDCG':>9}"
          f"{'AUC':>9}{'dAUC':>9}{'같은 방향':>11}")
    print("  " + "-" * 92)

    for index, row in enumerate(ordered, 1):
        # Top-1이 올랐을 때 다른 지표도 같이 올랐는가
        same = (
            row["delta_top1"] > 0
            and row["delta_mrr"] > 0
            and row["delta_ndcg5"] > 0
            and row["delta_auc"] > 0
        )
        mark = "예" if same else ("기준" if row["delta_top1"] == 0 else "아니오")

        print(
            f"  {index:>3}{row['alpha2']:>7.2f}{row['alpha3']:>6.2f}  "
            f"{row['top1_accuracy']:>8.4f}{row['delta_top1']:>+9.4f}"
            f"{row['mrr']:>9.4f}{row['delta_mrr']:>+9.4f}"
            f"{row['ndcg5']:>9.4f}{row['delta_ndcg5']:>+9.4f}"
            f"{row['auc']:>9.4f}{row['delta_auc']:>+9.4f}{mark:>11}"
        )

    print()
    print("  '같은 방향'은 Top-1 / MRR / nDCG@5 / AUC가 모두 S123보다 나은 경우다.")
    print("  Top-1 하나만 보고 alpha를 확정하지 않는다.")

    return ordered


def print_alpha3_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    section("alpha3 값별 최고 조합")

    summary: List[Dict[str, Any]] = []

    print()
    print(f"  {'alpha3':>8}{'best alpha2':>13}{'Top-1':>10}{'dTop1':>10}"
          f"{'MRR':>10}{'nDCG@5':>10}{'AUC':>10}")
    print("  " + "-" * 71)

    for alpha3 in ALPHA3_GRID:
        candidates = [r for r in rows if r["alpha3"] == alpha3]
        best = max(candidates, key=lambda r: r["top1_accuracy"])

        print(
            f"  {alpha3:>8.2f}{best['alpha2']:>13.2f}"
            f"{best['top1_accuracy']:>10.4f}{best['delta_top1']:>+10.4f}"
            f"{best['mrr']:>10.4f}{best['ndcg5']:>10.4f}{best['auc']:>10.4f}"
        )

        summary.append({
            "alpha3": alpha3,
            "best_alpha2": best["alpha2"],
            "top1_accuracy": best["top1_accuracy"],
            "delta_top1": best["delta_top1"],
            "mrr": best["mrr"],
            "ndcg5": best["ndcg5"],
            "auc": best["auc"],
        })

    best_alpha3 = max(summary, key=lambda s: s["top1_accuracy"])["alpha3"]

    print()
    print(f"  전체 최고는 alpha3 = {best_alpha3:g} 에서 나왔습니다.")
    print()

    if best_alpha3 == 0.0:
        print("  >>> alpha3 = 0이 최적입니다.")
        print("      현재 L3의 ranking 기여가 거의 없거나 방해가 됩니다.")
    elif best_alpha3 < 0.5:
        print(f"  >>> 작은 alpha3({best_alpha3:g})에서 최적입니다.")
        print("      c3 정보는 쓸모가 있지만 기존 weight 1은 너무 큽니다.")
    elif best_alpha3 >= 1.0:
        print(f"  >>> alpha3 = {best_alpha3:g}에서 최적입니다.")
        print("      c3가 방해한다는 Priority 0 해석을 다시 봐야 합니다.")
    else:
        print(f"  >>> 중간 alpha3({best_alpha3:g})에서 최적입니다.")

    return summary


def boundary_flags(best: Dict[str, Any]) -> Dict[str, Any]:
    """best 조합이 grid 경계에 붙어 있는지 본다.

    경계에 붙어 있으면 최적점이 grid 밖에 있을 수 있다.
    fine grid 범위를 정할 때 이쪽으로 넓혀야 한다.
    """
    a2, a3 = best["alpha2"], best["alpha3"]

    flags = {
        "alpha2": a2,
        "alpha3": a3,
        "alpha2_at_min": a2 == ALPHA2_GRID[0],
        "alpha2_at_max": a2 == ALPHA2_GRID[-1],
        "alpha3_at_min": a3 == ALPHA3_GRID[0],
        "alpha3_at_max": a3 == ALPHA3_GRID[-1],
    }

    flags["on_boundary"] = any(
        flags[k] for k in
        ("alpha2_at_min", "alpha2_at_max", "alpha3_at_min", "alpha3_at_max")
    )

    return flags


def print_boundary_notice(flags: Dict[str, Any]) -> None:
    section("grid 경계 확인")

    a2, a3 = flags["alpha2"], flags["alpha3"]

    print(f"  best = (alpha2 {a2:g}, alpha3 {a3:g})")
    print(f"  alpha2 범위 [{ALPHA2_GRID[0]:g}, {ALPHA2_GRID[-1]:g}]  "
          f"alpha3 범위 [{ALPHA3_GRID[0]:g}, {ALPHA3_GRID[-1]:g}]")
    print()

    if not flags["on_boundary"]:
        print("  best가 grid 내부에 있습니다.")
        print("  fine grid는 best 주변을 좁게 보면 됩니다.")
        return

    print("  >>> best가 grid 경계에 있습니다.")
    print("      최적점이 현재 grid 밖에 있을 수 있습니다.")
    print()

    if flags["alpha2_at_max"]:
        print(f"      alpha2가 최대값({ALPHA2_GRID[-1]:g})입니다. "
              f"더 큰 alpha2를 봐야 합니다.")
    if flags["alpha2_at_min"]:
        print(f"      alpha2가 최소값({ALPHA2_GRID[0]:g})입니다. "
              f"alpha2 = 0이 최적이면 c2가 ranking에 기여하지 않습니다.")
    if flags["alpha3_at_max"]:
        print(f"      alpha3가 최대값({ALPHA3_GRID[-1]:g})입니다. "
              f"더 큰 alpha3를 봐야 합니다.")
    if flags["alpha3_at_min"]:
        print(f"      alpha3가 최소값({ALPHA3_GRID[0]:g})입니다. "
              f"alpha3 = 0이 최적이면 c3가 ranking에 기여하지 않습니다.")

    print()
    print("      다음 grid 범위는 사용자가 결정합니다. "
          "이 스크립트가 자동으로 확장하지 않습니다.")


def transition_vs_reference(
    best: Dict[str, Any],
    reference: Dict[str, Any],
    labels: np.ndarray,
) -> Dict[str, Any]:
    section(f"transition 분석  —  S123 = (1,1,1)  vs  "
            f"best = (1, {best['alpha2']:g}, {best['alpha3']:g})")

    before = group_correct(reference["_scores"], labels)
    after = group_correct(best["_scores"], labels)

    total = len(before)

    cc = int((before & after).sum())
    cw = int((before & ~after).sum())
    wc = int((~before & after).sum())
    ww = int((~before & ~after).sum())

    print()
    print(f"    S123 correct -> weighted correct : {cc:>9,}  ({cc / total:>6.2%})")
    print(f"    S123 correct -> weighted wrong   : {cw:>9,}  ({cw / total:>6.2%})   <- broken")
    print(f"    S123 wrong   -> weighted correct : {wc:>9,}  ({wc / total:>6.2%})   <- rescued")
    print(f"    S123 wrong   -> weighted wrong   : {ww:>9,}  ({ww / total:>6.2%})")
    print()
    print(f"    rescued  = {wc:,}")
    print(f"    broken   = {cw:,}")
    print(f"    net gain = {wc - cw:+,}")

    if cw:
        print(f"    rescued / broken = {wc / cw:.3f}")

    print()
    print(f"    Top-1 {before.mean():.4f} -> {after.mean():.4f}"
          f"  ({after.mean() - before.mean():+.4f})")

    return {
        "reference": {"alpha1": ALPHA1, "alpha2": 1.0, "alpha3": 1.0},
        "weighted": {
            "alpha1": ALPHA1,
            "alpha2": best["alpha2"],
            "alpha3": best["alpha3"],
        },
        "num_impressions": total,
        "correct_to_correct": cc,
        "correct_to_wrong": cw,
        "wrong_to_correct": wc,
        "wrong_to_wrong": ww,
        "broken": cw,
        "rescued": wc,
        "net_gain": wc - cw,
        "rescued_over_broken": (wc / cw) if cw else None,
        "s123_top1": float(before.mean()),
        "weighted_top1": float(after.mean()),
        "top1_delta": float(after.mean() - before.mean()),
    }


def write_readme(
    path: Path,
    rows: List[Dict[str, Any]],
    top10: List[Dict[str, Any]],
    alpha3_summary: List[Dict[str, Any]],
    transition: Dict[str, Any],
    sanity: Dict[str, Any],
    reference: Dict[str, Any],
    metadata: Dict[str, Any],
    flags: Dict[str, Any],
) -> None:
    lines: List[str] = []
    add = lines.append

    best = top10[0]

    add("# Priority 0 후속 — level weight grid")
    add("")
    add("모델을 다시 돌리지 않았다. Priority 0이 저장한 L1 / L2 / L3에")
    add("level weight만 적용해 ranking을 다시 매겼다.")
    add("")
    add("    S(alpha2, alpha3) = L1 + alpha2 * L2 + alpha3 * L3   (alpha1 = 1 고정)")
    add("")
    add("L1 / L2 / L3를 z-score나 normalize 하지 않았다.")
    add("")
    add("## 실행 정보")
    add("")
    add(f"- candidate_scores : `{metadata['source_candidate_scores']}`")
    add(f"- checkpoint       : `{metadata.get('source_checkpoint') or '(기록 없음)'}`")
    add(f"- validation       : `{metadata.get('validation_parquet') or '(기록 없음)'}`")
    add(f"- row 수           : {metadata['dataset_row_count']:,} "
        f"(impression {reference['num_impressions']:,} x {NUM_CANDIDATES})")
    add(f"- alpha2 후보      : {', '.join(f'{a:g}' for a in ALPHA2_GRID)}")
    add(f"- alpha3 후보      : {', '.join(f'{a:g}' for a in ALPHA3_GRID)}")
    add(f"- 조합 수          : {len(rows)}")
    add(f"- git commit       : {metadata.get('git_commit') or '(알 수 없음)'}")
    add("")

    add("## Sanity check")
    add("")
    add("baseline 3개가 기존 Priority 0 결과를 재현하는지 확인했다.")
    add("")
    add("| 조합 | 대조 | score max diff | Top-1 diff | 판정 |")
    add("|---|---|---|---|---|")

    for check in sanity["checks"]:
        add(
            f"| (1, {check['alpha2']:g}, {check['alpha3']:g}) | "
            f"{check['baseline_column']} | "
            f"{check['score_max_abs_diff']:.3e} | "
            f"{check['metric_deltas']['top1']:.2e} | "
            f"{'OK' if check['passed'] else 'FAIL'} |"
        )

    add("")

    if not sanity["all_passed"]:
        add("**재현되지 않았다. 아래 결과를 해석하면 안 된다.**")
        add("")

    add("## Top-10")
    add("")
    add("| # | a2 | a3 | Top-1 | dTop-1 | MRR | dMRR | nDCG@5 | dnDCG | AUC | dAUC |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")

    for index, row in enumerate(top10, 1):
        add(
            f"| {index} | {row['alpha2']:g} | {row['alpha3']:g} | "
            f"{row['top1_accuracy']:.4f} | {row['delta_top1']:+.4f} | "
            f"{row['mrr']:.4f} | {row['delta_mrr']:+.4f} | "
            f"{row['ndcg5']:.4f} | {row['delta_ndcg5']:+.4f} | "
            f"{row['auc']:.4f} | {row['delta_auc']:+.4f} |"
        )

    add("")
    add("delta는 모두 `S123 = (1, 1, 1)` 대비다.")
    add("")
    add(f"기준: S123 Top-1 {reference['top1_accuracy']:.4f} | "
        f"MRR {reference['mrr']:.4f} | nDCG@5 {reference['ndcg5']:.4f} | "
        f"AUC {reference['auc']:.4f}")
    add("")

    add("## alpha3 값별 최고 조합")
    add("")
    add("| alpha3 | best alpha2 | Top-1 | dTop-1 | MRR | nDCG@5 | AUC |")
    add("|---|---|---|---|---|---|---|")

    for entry in alpha3_summary:
        add(
            f"| {entry['alpha3']:g} | {entry['best_alpha2']:g} | "
            f"{entry['top1_accuracy']:.4f} | {entry['delta_top1']:+.4f} | "
            f"{entry['mrr']:.4f} | {entry['ndcg5']:.4f} | {entry['auc']:.4f} |"
        )

    best_alpha3 = max(alpha3_summary, key=lambda s: s["top1_accuracy"])["alpha3"]

    add("")
    add(f"전체 최고는 `alpha3 = {best_alpha3:g}`에서 나왔다.")
    add("")

    if best_alpha3 == 0.0:
        add("**alpha3 = 0이 최적이다.** 현재 L3의 ranking 기여가 거의 없거나")
        add("오히려 방해가 된다는 뜻이다.")
    elif best_alpha3 < 0.5:
        add(f"**작은 alpha3({best_alpha3:g})에서 최적이다.** c3 정보 자체는")
        add("쓸모가 있지만 기존 weight 1은 너무 크다.")
    elif best_alpha3 >= 1.0:
        add(f"**alpha3 = {best_alpha3:g}에서 최적이다.** c3가 ranking을")
        add("방해한다는 Priority 0 해석을 다시 검토해야 한다.")
    else:
        add(f"**중간 alpha3({best_alpha3:g})에서 최적이다.**")

    add("")
    add("## best 조합의 transition")
    add("")
    add(f"`S123 = (1,1,1)` vs `best = (1, {best['alpha2']:g}, {best['alpha3']:g})`")
    add("")
    add(f"- S123 correct -> weighted correct : {transition['correct_to_correct']:,}")
    add(f"- S123 correct -> weighted wrong   : **{transition['broken']:,}** (broken)")
    add(f"- S123 wrong   -> weighted correct : **{transition['rescued']:,}** (rescued)")
    add(f"- S123 wrong   -> weighted wrong   : {transition['wrong_to_wrong']:,}")
    add("")
    add(f"- net gain = rescued - broken = **{transition['net_gain']:+,}**")

    if transition["rescued_over_broken"] is not None:
        add(f"- rescued / broken = **{transition['rescued_over_broken']:.3f}**")

    add(f"- Top-1 {transition['s123_top1']:.4f} -> "
        f"{transition['weighted_top1']:.4f} ({transition['top1_delta']:+.4f})")
    add("")

    add("## grid 경계")
    add("")
    add(f"best = (alpha2 {flags['alpha2']:g}, alpha3 {flags['alpha3']:g}), "
        f"alpha2 범위 [{ALPHA2_GRID[0]:g}, {ALPHA2_GRID[-1]:g}], "
        f"alpha3 범위 [{ALPHA3_GRID[0]:g}, {ALPHA3_GRID[-1]:g}]")
    add("")

    if flags["on_boundary"]:
        add("**best가 grid 경계에 있다.** 최적점이 현재 grid 밖에 있을 수 있다.")
        add("")
        if flags["alpha2_at_max"]:
            add(f"- alpha2가 최대값({ALPHA2_GRID[-1]:g})이다. 더 큰 alpha2를 봐야 한다.")
        if flags["alpha2_at_min"]:
            add(f"- alpha2가 최소값({ALPHA2_GRID[0]:g})이다. "
                f"alpha2 = 0이 최적이면 c2가 ranking에 기여하지 않는다.")
        if flags["alpha3_at_max"]:
            add(f"- alpha3가 최대값({ALPHA3_GRID[-1]:g})이다. 더 큰 alpha3를 봐야 한다.")
        if flags["alpha3_at_min"]:
            add(f"- alpha3가 최소값({ALPHA3_GRID[0]:g})이다. "
                f"alpha3 = 0이 최적이면 c3가 ranking에 기여하지 않는다.")
        add("")
        add("다음 grid 범위는 사용자가 결정한다. 스크립트가 자동으로 확장하지 않는다.")
    else:
        add("best가 grid 내부에 있다. fine grid는 best 주변을 좁게 보면 된다.")

    add("")
    add("## 주의")
    add("")
    add("- alpha를 바꾸면 score의 scale과 temperature도 같이 바뀐다.")
    add("  ranking이 같아도 softmax 확률과 preference loss는 달라진다.")
    add("  이번 alpha 선택의 기준은 ranking metric이다.")
    add("- positive/negative score 평균과 gap도 scale에 따라 바뀌므로")
    add("  조합 간 우열 판단에 쓰지 않는다.")
    add("- 1차 coarse grid만 돌렸다. best 주변 fine grid는 아직 하지 않았다.")
    add("")
    add("## 하지 않은 것")
    add("")
    add("- Transformer / RQ-VAE 재학습")
    add("- checkpoint 생성, SID 재생성, candidate 재추론")
    add("- Test 데이터 사용")
    add("- Priority 2")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", required=True,
        help="Priority 0이 만든 candidate_scores.parquet",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--source-metadata", default=None,
        help=(
            "Priority 0의 run_metadata.json. "
            "생략하면 --scores와 같은 폴더에서 찾는다."
        ),
    )
    args = parser.parse_args()

    started_at = time.time()

    scores_path = resolve_path(args.scores)
    out_dir = resolve_path(args.out)

    if not scores_path.exists():
        print(f"candidate_scores.parquet이 없습니다: {scores_path}")
        return 1

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"출력 폴더가 비어 있지 않습니다. 덮어쓰지 않습니다: {out_dir}")
        return 1

    section("level weight grid")
    print(f"  입력 : {scores_path}")
    print(f"  출력 : {out_dir}")
    print()
    print("  S(alpha2, alpha3) = L1 + alpha2 * L2 + alpha3 * L3   (alpha1 = 1)")
    print()
    print("  재학습하지 않습니다. 추론하지 않습니다. Test를 쓰지 않습니다.")
    print("  L1/L2/L3를 normalize하지 않습니다.")

    df = pd.read_parquet(scores_path)

    required = ["L1", "L2", "L3", "label", GROUP_COLUMN]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"\n필요한 컬럼이 없습니다: {missing}")
        print(f"파일의 컬럼: {list(df.columns)}")
        return 1

    df = df.sort_values(
        [GROUP_COLUMN, "candidate_index"]
        if "candidate_index" in df.columns else [GROUP_COLUMN],
        kind="stable",
    ).reset_index(drop=True)

    counts = df.groupby(GROUP_COLUMN).size()

    if not (counts == NUM_CANDIDATES).all():
        print(f"\nimpression당 후보가 {NUM_CANDIDATES}개가 아닌 행이 있습니다.")
        return 1

    print()
    print(f"  {len(df):,} rows = impression {len(counts):,} x {NUM_CANDIDATES}")

    # evaluate_ranking에 넘길 최소 프레임
    base = df[[GROUP_COLUMN, "label"]].copy()
    base["label"] = base["label"].astype(int)

    labels = df["label"].to_numpy().astype(int)

    # 42개 조합
    rows: List[Dict[str, Any]] = []
    total = len(ALPHA2_GRID) * len(ALPHA3_GRID)

    section(f"{total}개 조합 평가")
    print()

    for index, alpha2 in enumerate(ALPHA2_GRID):
        for alpha3 in ALPHA3_GRID:
            row = evaluate_weights(df, base, alpha2, alpha3)
            rows.append(row)
            done = len(rows)
            print(f"  [{done:>2}/{total}] a2={alpha2:>4.2f} a3={alpha3:>4.2f}  "
                  f"Top-1 {row['top1_accuracy']:.4f}  "
                  f"MRR {row['mrr']:.4f}  AUC {row['auc']:.4f}", flush=True)

    by_key = {(r["alpha2"], r["alpha3"]): r for r in rows}
    reference = by_key[REFERENCE_KEY]

    for row in rows:
        row["delta_top1"] = row["top1_accuracy"] - reference["top1_accuracy"]
        row["delta_mrr"] = row["mrr"] - reference["mrr"]
        row["delta_ndcg5"] = row["ndcg5"] - reference["ndcg5"]
        row["delta_auc"] = row["auc"] - reference["auc"]
        row["baseline"] = BASELINES.get((row["alpha2"], row["alpha3"]), "")

    sanity = sanity_check(df, rows)

    print_grid(rows, reference)
    top10 = print_top10(rows)
    alpha3_summary = print_alpha3_summary(rows)

    best = top10[0]
    flags = boundary_flags(best)
    print_boundary_notice(flags)

    transition = transition_vs_reference(best, reference, labels)

    # 저장
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in sorted(rows, key=lambda r: -r["top1_accuracy"])
    ]
    write_csv(out_dir / "grid_results.csv", csv_rows)
    write_csv(
        out_dir / "top10.csv",
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in top10],
    )

    (out_dir / "best_config.json").write_text(
        json.dumps(
            {
                "alpha1": ALPHA1,
                "alpha2": best["alpha2"],
                "alpha3": best["alpha3"],
                "metrics": {
                    k: v for k, v in best.items() if not k.startswith("_")
                },
                "reference_s123": {
                    k: v for k, v in reference.items() if not k.startswith("_")
                },
                "alpha3_summary": alpha3_summary,
                "sanity_check": sanity,
                "grid_boundary": flags,
                "note": (
                    "Top-1 기준 1위다. 최종 확정 전에 MRR / nDCG@5 / AUC가 "
                    "같은 방향으로 움직였는지 확인할 것."
                ),
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )

    (out_dir / "best_transition_summary.json").write_text(
        json.dumps(transition, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    keep = [
        c for c in
        ["impression_id", GROUP_COLUMN, "candidate_index", "label",
         "c1", "c2", "c3", "L1", "L2", "L3", "S1", "S12", "S123"]
        if c in df.columns
    ]

    best_df = df[keep].copy()
    best_df["weighted_score"] = best["_scores"]
    best_df["s123_score"] = reference["_scores"]
    best_df.to_parquet(out_dir / "best_candidate_scores.parquet", index=False)

    # Priority 0의 metadata에서 출처를 가져온다
    source_meta_path = (
        resolve_path(args.source_metadata) if args.source_metadata
        else scores_path.parent / "run_metadata.json"
    )

    source_meta: Dict[str, Any] = {}

    if source_meta_path.exists():
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))

    metadata = {
        "experiment": "priority0_level_weight_grid",
        "source_candidate_scores": str(scores_path),
        "source_run_metadata": (
            str(source_meta_path) if source_meta_path.exists() else None
        ),
        "source_checkpoint": source_meta.get("checkpoint"),
        "validation_parquet": source_meta.get("dataset"),
        "dataset_row_count": int(len(df)),
        "num_impressions": int(len(counts)),
        "alpha_search_space": {
            "alpha1": ALPHA1,
            "alpha2": ALPHA2_GRID,
            "alpha3": ALPHA3_GRID,
            "num_combinations": total,
        },
        "normalization": "none (raw log-prob)",
        "group_column": GROUP_COLUMN,
        "metric_function": "evaluate.metrics.evaluate_ranking",
        "sanity_check_passed": sanity["all_passed"],
        "best_on_grid_boundary": flags["on_boundary"],
        "best": {
            "alpha1": ALPHA1,
            "alpha2": best["alpha2"],
            "alpha3": best["alpha3"],
        },
        "did_not_do": [
            "Transformer 재학습", "checkpoint 생성", "RQ-VAE 재학습",
            "SID 재생성", "candidate 재추론", "Test 사용", "Priority 2",
            "fine grid",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "git_commit": git_commit_hash(),
        "python": sys.version.split()[0],
    }

    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    write_readme(
        out_dir / "README.md",
        rows, top10, alpha3_summary, transition, sanity, reference, metadata,
        flags,
    )

    section("저장 완료")
    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {out_dir / name}")

    if not sanity["all_passed"]:
        print()
        print("  sanity check가 실패했습니다. 결과를 해석하기 전에 원인을 확인하세요.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
