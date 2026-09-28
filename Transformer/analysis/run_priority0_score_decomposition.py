"""V2 Priority 0 — V1 candidate score를 c1/c2/c3 level로 분해한다.

핵심 질문
    현재 V1의 S123 = L1 + L2 + L3에서
    c3를 같은 가중치로 더하는 것이 ranking을 실제로 개선하는가?

이 스크립트는 inference-only 진단이다.
  - 모델 구조를 바꾸지 않는다
  - 재학습하지 않는다 (optimizer / backward / early stopping 없음)
  - 기존 checkpoint와 sweep 결과를 덮어쓰지 않는다
  - Test 데이터를 읽지 않는다
  - RQ-VAE / SID를 다시 만들지 않는다

score를 새로 정의하지 않는다.
L1/L2/L3는 modules/model.py의 score_candidates()가 이미 반환하는 값이고,
predict_sid.py가 그것을 c1_log_prob / c2_log_prob / c3_log_prob로 저장한다.
이 스크립트는 그 저장된 값을 조합만 한다.

    L1 = log p(c1 | H)                decoder position 0
    L2 = log p(c2 | H, c1)            decoder position 1
    L3 = log p(c3 | H, c1, c2)        decoder position 2

    S1   = L1
    S2   = L2
    S3   = L3
    S12  = L1 + L2
    S123 = L1 + L2 + L3      <- 기존 V1 candidate_score와 같아야 한다

metric은 evaluate/metrics.py의 evaluate_ranking을 그대로 쓴다.
candidate_score 컬럼만 갈아끼워 호출하므로 동점 처리와 rank 규칙이
기존 평가 경로와 완전히 동일하다.

    cd Transformer
    python -m analysis.run_priority0_score_decomposition \
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

from evaluate.metrics import evaluate_ranking
from sweep.run_stage import format_binding, write_csv


BASE_DIR = Path(__file__).resolve().parent.parent
PREDICT_SCRIPT = BASE_DIR / "predict_sid.py"

NUM_CANDIDATES = 5

# 분해해서 만들 score들. (이름, 더할 level 목록)
SCORE_TYPES = (
    ("S1", ("L1",)),
    ("S2", ("L2",)),
    ("S3", ("L3",)),
    ("S12", ("L1", "L2")),
    ("S123", ("L1", "L2", "L3")),
)

LEVEL_COLUMNS = {"L1": "c1_log_prob", "L2": "c2_log_prob", "L3": "c3_log_prob"}

# V1(shuffle 전) seed 42 validation 성능. 기준을 못 찾을 때만 쓴다.
REFERENCE_SEED42 = {
    "top1_accuracy": 0.2702,
    "mrr": 0.5205,
    "ndcg@5": 0.6389,
    "auc": 0.5858,
}

# run_summary.json의 best_metrics 키 -> 이 스크립트의 metric 이름
SUMMARY_METRIC_KEYS = {
    "val_top1_accuracy": "top1_accuracy",
    "val_mrr": "mrr",
    "val_ndcg@5": "ndcg@5",
    "val_auc": "auc",
}

# 재현 판정 허용 오차. float32 저장 반올림만 허용한다.
REPRODUCTION_TOLERANCE = 0.001


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# checkpoint의 run_summary.json에서 설정을 가져올 때 빼는 항목.
# 모델 구조와 무관하고 추론에 영향을 주지 않는다.
BINDING_PREFIXES_TO_DROP = (
    "train.save_dir",
    "train.num_workers",
    "train.prefetch_factor",
    "train.progress_interval",
    "train.num_epochs",
    "train.early_stopping_patience",
    "train.seed",
    "train.save_every_epoch",
    "train.save_optimizer_state",
    "train.train_path",
    "train.validation_path",
)


def bindings_from_run_summary(checkpoint: Path) -> Optional[List[str]]:
    """checkpoint를 만든 학습의 gin override를 그대로 가져온다.

    selected_final.json이 없을 때 쓴다.
    학습 때 쓴 문자열을 그대로 재사용하므로 값 표기가 달라질 여지가 없다.
    """
    summary_path = checkpoint.parent / "run_summary.json"

    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overrides = summary.get("gin_overrides") or []

    # override가 하나도 없는 경우도 정상이다.
    # base gin 파일만으로 설정이 끝났다는 뜻이고,
    # predict_sid.py도 같은 파일을 읽으므로 추가할 것이 없다.
    return [
        binding for binding in overrides
        if not binding.split("=", 1)[0].strip().startswith(BINDING_PREFIXES_TO_DROP)
    ]


def load_reference(
    checkpoint: Path,
    explicit: Optional[str],
) -> tuple[Dict[str, float], str]:
    """재현 확인에 쓸 기준값을 찾는다.

    checkpoint를 만든 학습의 run_summary.json이 옆에 있으면 그걸 쓴다.
    그래야 shuffle 전/후 어느 데이터로 돌렸든 자기 자신과 비교하게 된다.
    없으면 V1 seed 42 값으로 되돌아간다.
    """
    candidates = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.append(checkpoint.parent / "run_summary.json")

    for path in candidates:
        if not path.exists():
            continue

        summary = json.loads(path.read_text(encoding="utf-8"))
        best = summary.get("best_metrics") or {}

        values = {
            name: best[key]
            for key, name in SUMMARY_METRIC_KEYS.items()
            if best.get(key) is not None
        }

        if len(values) == len(SUMMARY_METRIC_KEYS):
            return values, str(path)

    return dict(REFERENCE_SEED42), "V1 seed 42 (하드코딩)"


def find_checkpoint(final: Dict[str, Any], seed: int) -> Path:
    """selected_final.json에서 해당 seed의 checkpoint를 찾는다."""
    seeds = final.get("seeds") or []
    checkpoints = final.get("checkpoints") or []

    if len(seeds) != len(checkpoints):
        raise ValueError("selected_final.json의 seeds와 checkpoints 길이가 다릅니다.")

    for s, c in zip(seeds, checkpoints):
        if s == seed:
            return Path(c)

    raise ValueError(f"seed {seed}의 checkpoint가 selected_final.json에 없습니다.")


def run_predict(
    base_config: Path,
    checkpoint: Path,
    data_path: Path,
    output_path: Path,
    log_path: Path,
    bindings: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    binding_strings: Optional[List[str]] = None,
) -> int:
    command = [
        sys.executable, "-u", str(PREDICT_SCRIPT),
        "--config", str(base_config),
        "--checkpoint", str(checkpoint),
        "--test_path", str(data_path),
        "--output_path", str(output_path),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
    ]

    if binding_strings:
        for binding in binding_strings:
            command += ["--gin-binding", binding]
    else:
        for key in sorted(bindings):
            command += ["--gin-binding", format_binding(key, bindings[key])]

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command, cwd=str(BASE_DIR),
            stdout=log_file, stderr=subprocess.STDOUT,
        )

    return process.returncode


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    """저장된 level별 log prob에서 S1..S123을 만든다."""
    for level, column in LEVEL_COLUMNS.items():
        if column not in df.columns:
            raise ValueError(
                f"예측 parquet에 {column}이 없습니다. "
                "predict_sid.py가 level별 log prob를 저장하는 버전인지 확인하세요."
            )
        df[level] = df[column].astype(np.float64)

    for name, levels in SCORE_TYPES:
        df[name] = sum(df[level] for level in levels)

    return df


def sanity_check(df: pd.DataFrame, sample_rows: int = 640) -> Dict[str, Any]:
    """재구성한 S123이 기존 candidate_score와 같은지 확인한다."""
    print()
    print("=" * 74)
    print("Sanity check — 재구성한 S123 vs 기존 candidate_score")
    print("=" * 74)

    original = df["candidate_score"].astype(np.float64).to_numpy()
    rebuilt = df["S123"].to_numpy()
    diff = np.abs(rebuilt - original)

    head = df.head(sample_rows)
    head_diff = np.abs(
        head["S123"].to_numpy() - head["candidate_score"].astype(np.float64).to_numpy()
    )

    print(f"  small batch ({len(head):,} rows)")
    print(f"    max  abs diff : {head_diff.max():.3e}")
    print(f"    mean abs diff : {head_diff.mean():.3e}")
    print()
    print(f"  전체 ({len(df):,} rows)")
    print(f"    max  abs diff : {diff.max():.3e}")
    print(f"    mean abs diff : {diff.mean():.3e}")
    print()

    # 순위가 같은지 — impression별 argmax 비교
    original_top = group_argmax(df, "candidate_score")
    rebuilt_top = group_argmax(df, "S123")
    same_top1 = float((original_top == rebuilt_top).mean())

    print(f"  impression별 Top-1 후보가 같은 비율 : {same_top1:.6%}")
    print()
    print("  참고: predict_sid.py는 AMP를 쓰지 않아 추론이 fp32다.")
    print("        parquet에 float32로 저장된 값을 float64로 다시 더하므로")
    print("        1e-6 수준의 반올림 차이는 정상이다.")

    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "small_batch_rows": int(len(head)),
        "small_batch_max_abs_diff": float(head_diff.max()),
        "small_batch_mean_abs_diff": float(head_diff.mean()),
        "top1_agreement": same_top1,
    }


def group_argmax(df: pd.DataFrame, column: str) -> np.ndarray:
    """impression별로 가장 점수가 높은 candidate_index를 돌려준다.

    동점 처리를 evaluate/metrics.py의 top1_accuracy(np.argmax)와
    똑같이 맞추기 위해 낮은 인덱스를 고른다.
    """
    values = df[column].to_numpy().reshape(-1, NUM_CANDIDATES)
    return values.argmax(axis=1)


def group_correct(df: pd.DataFrame, column: str) -> np.ndarray:
    """impression별로 Top-1이 정답이었는지 bool 배열."""
    labels = df["label"].to_numpy().reshape(-1, NUM_CANDIDATES)
    top = group_argmax(df, column)
    rows = np.arange(len(top))
    return labels[rows, top] == 1


def evaluate_score_type(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """기존 evaluate_ranking을 그대로 쓴다. 점수 컬럼만 갈아끼운다."""
    work = df[["sample_index", "label", name]].copy()
    work = work.rename(columns={name: "candidate_score"})
    work["label"] = work["label"].astype(int)

    metrics = evaluate_ranking(df=work, group_column="sample_index")

    scores = df[name].to_numpy()
    labels = df["label"].to_numpy().astype(int)

    # tie 분석
    # S1/S2/S3/S12는 항이 적어 exact tie가 자주 난다.
    # evaluate_ranking의 Top-1은 np.argmax라 동점이면 낮은 index를 고른다.
    # 그 값이 실제 변별력인지 동점 처리 결과인지 갈라야 한다.
    grid = scores.reshape(-1, NUM_CANDIDATES)
    label_grid = labels.reshape(-1, NUM_CANDIDATES)

    best_value = grid.max(axis=1, keepdims=True)
    is_best = grid == best_value
    tie_count = is_best.sum(axis=1)

    # 최고점이 k개이고 그 안에 positive가 있으면 1/k만 인정
    positive_is_best = (is_best & (label_grid == 1)).any(axis=1)
    tie_aware_top1 = float(np.mean(np.where(positive_is_best, 1.0 / tie_count, 0.0)))

    tied = tie_count > 1

    positive = scores[labels == 1]
    negative = scores[labels == 0]

    # 후보 5개 안의 상대 확률. scale에 민감하므로 보조 지표로만 본다.
    grouped = scores.reshape(-1, NUM_CANDIDATES)
    shifted = grouped - grouped.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)

    label_grid = labels.reshape(-1, NUM_CANDIDATES)
    positive_index = label_grid.argmax(axis=1)
    rows = np.arange(len(probs))
    positive_prob = probs[rows, positive_index]

    return {
        "score_type": name,
        "top1_accuracy": metrics["top1_accuracy"],
        "naive_argmax_top1": metrics["top1_accuracy"],
        "tie_aware_top1": tie_aware_top1,
        "tie_gap": metrics["top1_accuracy"] - tie_aware_top1,
        "tied_impressions": int(tied.sum()),
        "tied_rate": float(tied.mean()),
        "mean_tie_count": float(tie_count.mean()),
        "all_tied_impressions": int((tie_count == NUM_CANDIDATES).sum()),
        "mrr": metrics["mrr"],
        "ndcg5": metrics["ndcg@5"],
        "auc": metrics["auc"],
        "positive_score_mean": float(positive.mean()),
        "positive_score_std": float(positive.std(ddof=1)),
        "negative_score_mean": float(negative.mean()),
        "negative_score_std": float(negative.std(ddof=1)),
        "score_gap": float(positive.mean() - negative.mean()),
        "positive_prob_mean": float(positive_prob.mean()),
        "preference_loss": float(-np.log(np.clip(positive_prob, 1e-12, None)).mean()),
        "num_impressions": int(metrics["num_impressions"]),
    }


def transition_stats(
    before: np.ndarray,
    after: np.ndarray,
    before_name: str,
    after_name: str,
) -> Dict[str, Any]:
    total = len(before)

    cc = int((before & after).sum())
    cw = int((before & ~after).sum())
    wc = int((~before & after).sum())
    ww = int((~before & ~after).sum())

    return {
        "from": before_name,
        "to": after_name,
        "num_impressions": total,
        "correct_to_correct": cc,
        "correct_to_wrong": cw,
        "wrong_to_correct": wc,
        "wrong_to_wrong": ww,
        "broken": cw,
        "rescued": wc,
        "net_gain": wc - cw,
        "broken_rate": cw / total,
        "rescued_rate": wc / total,
        "rescued_over_broken": (wc / cw) if cw else None,
        f"{before_name}_top1": float(before.mean()),
        f"{after_name}_top1": float(after.mean()),
        "top1_delta": float(after.mean() - before.mean()),
    }


def print_transition(stats: Dict[str, Any]) -> None:
    a, b = stats["from"], stats["to"]
    total = stats["num_impressions"]

    print()
    print(f"  [{a} -> {b}]")
    print(f"    {a} correct -> {b} correct : {stats['correct_to_correct']:>8,}"
          f"  ({stats['correct_to_correct'] / total:>6.2%})")
    print(f"    {a} correct -> {b} wrong   : {stats['correct_to_wrong']:>8,}"
          f"  ({stats['broken_rate']:>6.2%})   <- 망가진 건수")
    print(f"    {a} wrong   -> {b} correct : {stats['wrong_to_correct']:>8,}"
          f"  ({stats['rescued_rate']:>6.2%})   <- 살아난 건수")
    print(f"    {a} wrong   -> {b} wrong   : {stats['wrong_to_wrong']:>8,}"
          f"  ({stats['wrong_to_wrong'] / total:>6.2%})")
    print()
    print(f"    살아난 것 - 망가진 것 = {stats['net_gain']:+,}")

    if stats["rescued_over_broken"] is not None:
        print(f"    살아난 것 / 망가진 것 = {stats['rescued_over_broken']:.3f}")

    print(f"    Top-1 {stats[f'{a}_top1']:.4f} -> {stats[f'{b}_top1']:.4f}"
          f"  ({stats['top1_delta']:+.4f})")


def write_report(
    path: Path,
    rows: List[Dict[str, Any]],
    transitions: List[Dict[str, Any]],
    sanity: Dict[str, Any],
    metadata: Dict[str, Any],
    reference: Dict[str, float],
    reference_source: str,
) -> None:
    by_name = {r["score_type"]: r for r in rows}
    s12, s123 = by_name["S12"], by_name["S123"]
    t_s12_s123 = next(t for t in transitions if t["from"] == "S12")

    delta = s123["top1_accuracy"] - s12["top1_accuracy"]
    broken = t_s12_s123["correct_to_wrong"]
    rescued = t_s12_s123["wrong_to_correct"]

    lines: List[str] = []
    add = lines.append

    add("# V2 Priority 0 — score decomposition (seed 42, validation)")
    add("")
    add("V1 checkpoint를 재학습 없이 추론만 해서, candidate score를")
    add("c1 / c2 / c3 level로 분해한 결과다. Test 데이터는 쓰지 않았다.")
    add("")
    add("## 실행 정보")
    add("")
    add(f"- checkpoint : `{metadata['checkpoint']}`")
    add(f"- dataset    : `{metadata['dataset']}`")
    add(f"- seed       : {metadata['seed']}")
    add(f"- impression : {s123['num_impressions']:,}")
    add(f"- git commit : {metadata.get('git_commit') or '(알 수 없음)'}")
    add("")
    add("## Sanity check")
    add("")
    add("재구성한 S123과 기존 V1 candidate_score 비교")
    add("")
    add(f"- max abs diff  : {sanity['max_abs_diff']:.3e}")
    add(f"- mean abs diff : {sanity['mean_abs_diff']:.3e}")
    add(f"- Top-1 후보 일치율 : {sanity['top1_agreement']:.6%}")
    add("")
    add("추론은 fp32다 (predict_sid.py는 AMP를 쓰지 않는다).")
    add("parquet의 float32 값을 float64로 다시 더하므로 1e-6 수준 차이는 정상이다.")
    add("")
    add(f"학습 당시 결과와의 재현 확인 (기준: `{reference_source}`)")
    add("")
    add("| metric | 학습 당시 | S123 | 차이 |")
    add("|---|---|---|---|")

    for key, label in (
        ("top1_accuracy", "Top-1"), ("mrr", "MRR"),
        ("ndcg5", "nDCG@5"), ("auc", "AUC"),
    ):
        ref_key = "ndcg@5" if key == "ndcg5" else key
        ref = reference[ref_key]
        got = s123[key]
        add(f"| {label} | {ref:.4f} | {got:.4f} | {got - ref:+.4f} |")

    add("")
    add("## score 방식별 성능")
    add("")
    add("| score | Top-1 | MRR | nDCG@5 | AUC | pos mean | neg mean | gap |")
    add("|---|---|---|---|---|---|---|---|")

    for row in rows:
        add(
            f"| {row['score_type']} | {row['top1_accuracy']:.4f} | {row['mrr']:.4f} | "
            f"{row['ndcg5']:.4f} | {row['auc']:.4f} | "
            f"{row['positive_score_mean']:.4f} | {row['negative_score_mean']:.4f} | "
            f"{row['score_gap']:.4f} |"
        )

    add("")
    add("### 동점(tie) 분석")
    add("")
    add("| score | naive argmax Top-1 | tie-aware Top-1 | 차이 | tie impression | 비율 |")
    add("|---|---|---|---|---|---|")

    for row in rows:
        add(
            f"| {row['score_type']} | {row['naive_argmax_top1']:.4f} | "
            f"{row['tie_aware_top1']:.4f} | {row['tie_gap']:+.4f} | "
            f"{row['tied_impressions']:,} | {row['tied_rate']:.2%} |"
        )

    add("")
    add("- **naive argmax**: 현재 `evaluate/metrics.py` 방식. 동점이면 낮은 index")
    add("- **tie-aware**: 최고점이 k개이고 그중 positive가 있으면 1/k credit")
    add("")
    add("차이가 크면 그 score의 Top-1은 실제 변별력이 아니라 동점 처리 결과다.")
    add("")
    add("S1 / S12 / S123은 더하는 항의 개수가 달라 raw score의 scale이 다르다.")
    add("따라서 gap 크기만으로 우열을 판단하지 않는다.")
    add("판단은 Top-1 / MRR / nDCG@5 / AUC로 한다.")
    add("")
    add("무작위 기준: Top-1 0.2000 | MRR 0.4567 | nDCG@5 0.5897 | AUC 0.5000")
    add("")
    add("## transition 분석")
    add("")

    for t in transitions:
        a, b = t["from"], t["to"]
        add(f"### {a} -> {b}")
        add("")
        add(f"- {a} correct -> {b} correct : {t['correct_to_correct']:,}")
        add(f"- {a} correct -> {b} wrong   : **{t['correct_to_wrong']:,}** (망가짐)")
        add(f"- {a} wrong   -> {b} correct : **{t['wrong_to_correct']:,}** (살아남)")
        add(f"- {a} wrong   -> {b} wrong   : {t['wrong_to_wrong']:,}")
        add("")
        add(f"- 순증 = 살아남 - 망가짐 = **{t['net_gain']:+,}**")

        if t["rescued_over_broken"] is not None:
            add(f"- 비율 = 살아남 / 망가짐 = **{t['rescued_over_broken']:.3f}**")

        add(f"- Top-1 {t[f'{a}_top1']:.4f} -> {t[f'{b}_top1']:.4f} ({t['top1_delta']:+.4f})")
        add("")

    add("## 결론")
    add("")

    if delta < 0 and broken > rescued:
        add(f"S123의 Top-1이 S12보다 {abs(delta):.4f} 낮고,")
        add(f"c3를 더해서 망가진 impression({broken:,})이")
        add(f"살아난 impression({rescued:,})보다 {broken - rescued:,}건 많다.")
        add("")
        add("**c3 정보를 버려야 한다는 뜻은 아니다.**")
        add("현재처럼 c1 / c2 / c3를 1:1:1로 동일 가중 합산하는 방식에서")
        add("c3 항이 ranking을 방해하고 있을 가능성이 있다는 뜻이다.")
        add("")
        add("다음 실험 후보:")
        add("")
        add("    S = alpha1 * L1 + alpha2 * L2 + alpha3 * L3")
        add("")
        add("level별 가중치를 두는 방식을 검토한다.")
    elif delta > 0:
        add(f"S123의 Top-1이 S12보다 {delta:.4f} 높다.")
        add(f"c3를 더해서 살아난 impression({rescued:,})이")
        add(f"망가진 impression({broken:,})보다 {rescued - broken:,}건 많다.")
        add("")
        add("**c3가 문제라는 가설은 지지되지 않는다.**")
        add("현재의 1:1:1 합산에서 c3 항은 ranking에 기여하고 있다.")
    else:
        add(f"S123과 S12의 Top-1 차이가 {delta:+.4f}로 거의 없다.")
        add(f"망가진 건수 {broken:,}, 살아난 건수 {rescued:,}.")
        add("")
        add("c3 항이 ranking을 뚜렷하게 돕지도 방해하지도 않는다.")
        add("이 결과만으로는 가중치 실험의 필요성을 단정하기 어렵다.")

    add("")
    add("## 이번 실험에서 하지 않은 것")
    add("")
    add("- alpha 학습 및 weighted score 학습")
    add("- direct scorer")
    add("- c4 추가")
    add("- RQ-VAE 수정")
    add("- Test 데이터 사용")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep-out", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument(
        "--out", default=None,
        help=(
            "결과 폴더. 생략하면 "
            "<sweep-out>/priority0_score_decomposition/seed<N>"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "쓸 checkpoint를 직접 지정한다. "
            "생략하면 selected_final.json에서 해당 seed의 것을 찾는다."
        ),
    )
    parser.add_argument(
        "--reference-summary", default=None,
        help=(
            "재현 확인 기준으로 쓸 run_summary.json. "
            "생략하면 checkpoint 옆의 run_summary.json을 쓴다."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--scores-path",
        default=None,
        help=(
            "이미 만들어 둔 예측 parquet을 재사용한다. "
            "생략하면 새로 추론한다."
        ),
    )
    args = parser.parse_args()

    started_at = time.time()

    base_config = resolve_path(args.config)
    sweep_dir = resolve_path(args.sweep_out)
    validation_path = resolve_path(args.validation_path)

    final_path = sweep_dir / "seed_robustness" / "selected_final.json"

    if not validation_path.exists():
        print(f"validation 파일이 없습니다: {validation_path}")
        return 1

    params: Dict[str, Any] = {}
    binding_strings: Optional[List[str]] = None
    config_source = str(final_path)

    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        params = final.get("config") or {}

        checkpoint = (
            resolve_path(args.checkpoint) if args.checkpoint
            else find_checkpoint(final, args.seed)
        )
    elif args.checkpoint:
        # selected_final.json이 없으면 checkpoint를 만든 학습의
        # gin override를 그대로 쓴다.
        checkpoint = resolve_path(args.checkpoint)
        binding_strings = bindings_from_run_summary(checkpoint)

        if binding_strings is None:
            print(f"{final_path}도 없고 "
                  f"{checkpoint.parent / 'run_summary.json'}도 없습니다.")
            print("--sweep-out을 selected_final.json이 있는 폴더로 지정하세요.")
            return 1

        if not binding_strings:
            print("  (학습 때 gin override가 없었습니다. base config만 씁니다.)")

        config_source = str(checkpoint.parent / "run_summary.json")
    else:
        print(f"{final_path}가 없습니다.")
        print("--checkpoint로 checkpoint를 직접 지정하거나")
        print("--sweep-out을 selected_final.json이 있는 폴더로 지정하세요.")
        return 1

    reference, reference_source = load_reference(checkpoint, args.reference_summary)

    if not checkpoint.exists():
        print(f"checkpoint가 없습니다: {checkpoint}")
        return 1

    output_dir = resolve_path(args.out) if args.out else (
        sweep_dir / "priority0_score_decomposition" / f"seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 74)
    print("V2 Priority 0 — score decomposition")
    print("=" * 74)
    print(f"설정 출처   : {config_source}")
    print(f"checkpoint  : {checkpoint}")
    print(f"validation  : {validation_path}")
    print(f"seed        : {args.seed}")
    print(f"기준값      : {reference_source}")
    print(f"출력        : {output_dir}")
    print("Test 데이터는 사용하지 않습니다. 재학습하지 않습니다.")

    # 1. 예측 (또는 재사용)
    if args.scores_path:
        scores_path = resolve_path(args.scores_path)

        if not scores_path.exists():
            print(f"재사용할 예측 parquet이 없습니다: {scores_path}")
            return 1

        print(f"\n기존 예측을 재사용합니다: {scores_path}")
    else:
        scores_path = output_dir / f"validation_scores_seed_{args.seed}.parquet"
        log_path = output_dir / f"predict_seed_{args.seed}.log"

        print("\n추론 중...")
        code = run_predict(
            base_config=base_config,
            checkpoint=checkpoint,
            data_path=validation_path,
            output_path=scores_path,
            log_path=log_path,
            bindings=params,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            binding_strings=binding_strings,
        )

        if code != 0:
            print(f"추론 실패 (exit {code}). 로그: {log_path}")
            return 1

        print(f"완료: {scores_path}")

    # 2. score 조합
    df = pd.read_parquet(scores_path)
    df = df.sort_values("sample_index", kind="stable").reset_index(drop=True)

    counts = df.groupby("sample_index").size()

    if not (counts == NUM_CANDIDATES).all():
        print(f"impression당 후보가 {NUM_CANDIDATES}개가 아닌 행이 있습니다.")
        return 1

    df["candidate_index"] = df.groupby("sample_index").cumcount()
    df = build_scores(df)

    # 3. sanity check
    sanity = sanity_check(df)

    # 4. score 방식별 평가
    print()
    print("=" * 74)
    print("score 방식별 성능")
    print("=" * 74)
    print(f"  {'score':<7}{'Top-1':>9}{'MRR':>9}{'nDCG@5':>9}{'AUC':>9}"
          f"{'pos mean':>11}{'neg mean':>11}{'gap':>9}")
    print("  " + "-" * 72)

    rows: List[Dict[str, Any]] = []

    for name, _ in SCORE_TYPES:
        row = evaluate_score_type(df, name)
        rows.append(row)
        print(
            f"  {name:<7}{row['top1_accuracy']:>9.4f}{row['mrr']:>9.4f}"
            f"{row['ndcg5']:>9.4f}{row['auc']:>9.4f}"
            f"{row['positive_score_mean']:>11.4f}{row['negative_score_mean']:>11.4f}"
            f"{row['score_gap']:>9.4f}"
        )

    # tie 분석
    print()
    print("=" * 74)
    print("동점(tie) 분석")
    print("=" * 74)
    print(f"  {'score':<7}{'naive argmax':>15}{'tie-aware':>12}{'차이':>10}"
          f"{'tie impression':>17}{'비율':>9}")
    print("  " + "-" * 72)

    for row in rows:
        print(
            f"  {row['score_type']:<7}{row['naive_argmax_top1']:>15.4f}"
            f"{row['tie_aware_top1']:>12.4f}{row['tie_gap']:>+10.4f}"
            f"{row['tied_impressions']:>17,}{row['tied_rate']:>9.2%}"
        )

    print()
    print("  naive argmax : 현재 evaluate/metrics.py 방식. 동점이면 낮은 index")
    print("  tie-aware    : 최고점이 k개이고 그중 positive가 있으면 1/k credit")
    print()
    print("  차이가 크면 그 score의 Top-1은 실제 변별력이 아니라")
    print("  동점 처리 규칙이 만든 값이다.")

    print()
    print("  무작위 기준: Top-1 0.2000 | MRR 0.4567 | nDCG@5 0.5897 | AUC 0.5000")
    print()
    print("  S1/S12/S123은 더하는 항 수가 달라 raw score scale이 다르다.")
    print("  gap 크기로 우열을 판단하지 말 것. 판단은 ranking metric으로 한다.")

    # 재현 확인
    s123 = next(r for r in rows if r["score_type"] == "S123")

    print()
    print("=" * 74)
    print("학습 당시 validation 결과와 재현 확인")
    print("=" * 74)
    print(f"  기준 : {reference_source}")
    print()

    reproduced = True

    for key, label in (
        ("top1_accuracy", "Top-1"), ("mrr", "MRR"),
        ("ndcg5", "nDCG@5"), ("auc", "AUC"),
    ):
        ref_key = "ndcg@5" if key == "ndcg5" else key
        ref = reference[ref_key]
        got = s123[key]
        gap = abs(got - ref)
        ok = gap <= REPRODUCTION_TOLERANCE
        reproduced = reproduced and ok

        print(f"  {label:<8} 학습시 {ref:.4f} | S123 {got:.4f} | 차이 {got - ref:+.4f}"
              f"  {'OK' if ok else '<<< 불일치'}")

    if not reproduced:
        print()
        print("  기존 결과와 차이가 큽니다. 아래를 먼저 확인하세요.")
        print("    - checkpoint가 seed 42의 best가 맞는지")
        print("    - validation parquet이 학습 때와 같은 파일인지")
        print("    - gin binding이 selected_final.json과 같은지")
        print("    - 동점 처리 규칙이 evaluate/metrics.py와 같은지")
        print()
        print("  분석은 계속하지만 결과 해석 전에 위를 확인하세요.")

    # 5. transition 분석
    print()
    print("=" * 74)
    print("transition 분석")
    print("=" * 74)

    correct = {name: group_correct(df, name) for name, _ in SCORE_TYPES}

    transitions = [
        transition_stats(correct["S1"], correct["S12"], "S1", "S12"),
        transition_stats(correct["S12"], correct["S123"], "S12", "S123"),
    ]

    for stats in transitions:
        print_transition(stats)

    # 6. 저장
    summary_path = output_dir / "summary.csv"
    write_csv(summary_path, rows)

    transition_path = output_dir / "transition_summary.json"
    transition_path.write_text(
        json.dumps(
            {"transitions": transitions, "sanity_check": sanity},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    keep = [
        "impression_id", "sample_index", "candidate_index", "label",
        "c1", "c2", "c3", "L1", "L2", "L3",
        "S1", "S2", "S3", "S12", "S123", "candidate_score",
    ]
    keep = [c for c in keep if c in df.columns]

    candidate_path = output_dir / "candidate_scores.parquet"
    df[keep].to_parquet(candidate_path, index=False)

    metadata = {
        "experiment": "v2_priority0_score_decomposition",
        "checkpoint": str(checkpoint),
        "dataset": str(validation_path),
        "scores_parquet": str(scores_path),
        "seed": args.seed,
        "config": params or None,
        "config_source": config_source,
        "gin_bindings": binding_strings,
        "num_impressions": s123["num_impressions"],
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "reused_existing_scores": bool(args.scores_path),
        "reference_comparison": {
            "status": "match" if reproduced else "mismatch",
            "source": reference_source,
            "reference": reference,
        },
        "started_at_utc": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "git_commit": git_commit_hash(),
        "python": sys.version.split()[0],
        "notes": [
            "inference only. 재학습 없음. 모델 코드 수정 없음.",
            "L1/L2/L3는 predict_sid.py가 저장한 c1/c2/c3 log prob 그대로.",
            "metric은 evaluate/metrics.py의 evaluate_ranking을 그대로 사용.",
            "Test 데이터 미사용.",
        ],
    }

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    report_path = output_dir / "report.md"
    write_report(
        report_path, rows, transitions, sanity, metadata,
        reference, reference_source,
    )

    print()
    print("=" * 74)
    print("저장 완료")
    print("=" * 74)
    for path in (summary_path, transition_path, candidate_path, metadata_path, report_path):
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
