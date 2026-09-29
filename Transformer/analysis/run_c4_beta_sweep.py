"""c4 identity residual의 세기 beta4만 비교한다.

hidden weight는 [1.0, 0.5, 0.5]로 고정한다.

    a_semantic = (1.0*h1 + 0.5*h2 + 0.5*h3) / 2.0
    a_final    = a_semantic + beta4 * P(e_c4)

c4는 encoder token으로 들어가지 않는다. self-attention을 거치지 않고
article vector에 residual로만 붙는다.

beta4만 바꾸고 나머지는 전부 고정한다. 네 arm이 같은 구조를 갖도록
beta4=0에서도 c4_proj를 만든다. 그래야 초기값과 parameter 수가 같고
차이가 beta4 하나로만 남는다. beta4=0이면 forward에 0을 곱하므로
gradient도 0이고 c4 쪽은 학습되지 않는다.

    cd Transformer
    python -m analysis.run_c4_beta_sweep \
        --out sweep_out/ebnerd_v2/priority2_direct/c4_beta_sweep
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

import pandas as pd

from sweep.run_stage import write_csv


BASE_DIR = Path(__file__).resolve().parent.parent
RUN_SCRIPT = "analysis.run_priority2_direct"

# hidden weight 고정. hidden weight sweep에서 seed42 기준 가장 높았다.
# 다만 조합 간 차이가 seed 노이즈보다 작았으므로 provisional이다.
HIDDEN_WEIGHTS: Tuple[float, float, float] = (1.0, 0.5, 0.5)

BETAS: List[float] = [0.0, 0.01, 0.05, 0.1]

BASELINE_BETA = 0.0

# beta4=0은 hidden weight sweep의 D와 같은 계산이어야 한다.
# GPU 비결정성이 있으므로 일치하지 않아도 중단하지 않고 알리기만 한다.
REFERENCE_D = 0.266995
REFERENCE_TOLERANCE = 5e-4   # 0.05%p

BASELINES: List[Dict[str, Any]] = [
    {"config": "P2 c1234 mean", "top1_accuracy": 0.26137,
     "note": "history에 c4를 encoder token으로 넣음"},
    {"config": "P2 c123 mean (기존)", "top1_accuracy": 0.26650,
     "note": "mean code path"},
    {"config": "h-sweep A [1,1,1]", "top1_accuracy": 0.266261,
     "note": "weighted code path의 baseline"},
    {"config": "h-sweep D [1,.5,.5]", "top1_accuracy": 0.266995,
     "note": "이번 sweep의 beta4=0과 같은 계산"},
    {"config": "V1", "top1_accuracy": 0.27059,
     "note": "L1+L2+L3 생성확률, seed42"},
    {"config": "Learnable alpha", "top1_accuracy": 0.27166,
     "note": "alpha를 Train에서 학습"},
    {"config": "Fixed weighted V1 reference", "top1_accuracy": 0.29312,
     "note": ("Validation에서 L1/L2/L3 score weight를 탐색해 얻은 "
              "development reference. 동일 조건의 결과가 아니다.")},
]

NORM_KEYS = ["a_semantic", "P(e_c4)", "a_final"]


def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


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


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}초"
    if seconds < 3600:
        return f"{seconds / 60:.1f}분"
    return f"{seconds / 3600:.1f}시간"


def beta_dir(root: Path, beta: float) -> Path:
    return root / f"beta_{beta:.3f}" / "seed42"


def run_one(
    out_dir: Path, beta: float, args: argparse.Namespace, log_path: Path
) -> int:
    command = [
        sys.executable, "-u", "-m", RUN_SCRIPT,
        "--priority0-dir", args.priority0_dir,
        "--out", str(out_dir),
        "--history-levels", "3",
        "--history-mode", "weighted",
        "--level-weights", *(f"{w:g}" for w in HIDDEN_WEIGHTS),
        # beta4=0에서도 residual path를 만들어 네 arm의 구조를 맞춘다.
        "--use-c4-identity",
        "--beta4", f"{beta:g}",
        "--scorer", "bilinear",
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
        "--progress-interval", str(args.progress_interval),
    ]

    if args.config:
        command += ["--config", args.config]

    if args.train_path:
        command += ["--train-path", args.train_path]

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()

        process = subprocess.run(
            command, cwd=str(BASE_DIR),
            stdout=log_file, stderr=subprocess.STDOUT,
        )

    return process.returncode


def pick_norm(norms: List[Dict[str, Any]], needle: str) -> Optional[float]:
    """첫 batch norm 표에서 항 하나를 찾는다."""
    for row in norms:
        term = row["term"]

        if needle == "P(e_c4)":
            # ||P(e_c4)||와 ||beta*P(e_c4)||를 구분한다.
            if term == "mean ||P(e_c4)||":
                return float(row["mean_norm"])
            continue

        if needle in term:
            return float(row["mean_norm"])

    return None


def pick_scaled_norm(norms: List[Dict[str, Any]]) -> Optional[float]:
    for row in norms:
        term = row["term"]

        if "P(e_c4)" in term and term != "mean ||P(e_c4)||":
            return float(row["mean_norm"])

    return None


def collect(
    out_dir: Path, beta: float, elapsed: Optional[float]
) -> Dict[str, Any]:
    best = json.loads(
        (out_dir / "best_metrics.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (out_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    history = pd.read_csv(
        out_dir / "training_history.csv", encoding="utf-8-sig"
    )
    history.columns = [c.strip().lstrip("﻿") for c in history.columns]

    best_epoch = int(best["best_epoch"])
    row = history[history["epoch"] == best_epoch].iloc[0]
    validation = best["validation"]

    norms = metadata.get("first_batch_norms") or []

    semantic = pick_norm(norms, "a_semantic")
    residual = pick_norm(norms, "P(e_c4)")
    scaled = pick_scaled_norm(norms)
    final = pick_norm(norms, "a_final")

    train_top1 = float(row["train_top1"])
    val_top1 = float(validation["top1_accuracy"])

    return {
        "beta4": beta,
        "hidden_weights": (
            f"[{HIDDEN_WEIGHTS[0]:g}, {HIDDEN_WEIGHTS[1]:g}, "
            f"{HIDDEN_WEIGHTS[2]:g}]"
        ),
        "best_epoch": best_epoch,
        "epochs_run": int(len(history)),
        "train_loss": float(row["train_loss"]),
        "train_top1": train_top1,
        "val_loss": float(validation["total_loss"]),
        "val_top1": val_top1,
        "mrr": float(validation["mrr"]),
        "ndcg5": float(validation["ndcg@5"]),
        "auc": float(validation["auc"]),
        "positive_prob": float(validation["positive_prob"]),
        "negative_prob": float(validation["negative_prob"]),
        "score_gap": float(validation["score_gap"]),
        "overfit_gap": train_top1 - val_top1,
        "norm_a_semantic": semantic,
        "norm_residual": residual,
        "norm_scaled_residual": scaled,
        "norm_a_final": final,
        "residual_share": (
            scaled / semantic if semantic and scaled is not None else None
        ),
        "elapsed_seconds": round(elapsed, 1) if elapsed else None,
        "out_dir": str(out_dir),
    }


def print_results(rows: List[Dict[str, Any]]) -> None:
    section("sweep 결과  (Validation Top-1 내림차순)")

    baseline = next(
        (r for r in rows if r["beta4"] == BASELINE_BETA), None
    )

    header = (
        f"  {'beta4':>7} {'best':>5} {'run':>4} {'val Top-1':>10} "
        f"{'MRR':>8} {'nDCG@5':>8} {'AUC':>8} {'val loss':>9} "
        f"{'과적합폭':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) + 8))

    for row in sorted(rows, key=lambda r: -r["val_top1"]):
        if baseline and row["beta4"] != BASELINE_BETA:
            delta = f"  {(row['val_top1'] - baseline['val_top1']) * 100:+.3f}%p"
        else:
            delta = "  기준"

        print(
            f"  {row['beta4']:>7.3f} {row['best_epoch']:>5} "
            f"{row['epochs_run']:>4} {row['val_top1'] * 100:>9.3f}% "
            f"{row['mrr']:>8.4f} {row['ndcg5']:>8.4f} {row['auc']:>8.4f} "
            f"{row['val_loss']:>9.4f} {row['overfit_gap'] * 100:>8.2f}%p"
            f"{delta}"
        )

    print()
    print("  과적합폭 = best epoch의 train Top-1 - val Top-1")


def print_norms(rows: List[Dict[str, Any]]) -> None:
    section("첫 batch 항별 크기")

    header = (
        f"  {'beta4':>7} {'||a_semantic||':>16} {'||P(e_c4)||':>14} "
        f"{'||b*P(e_c4)||':>15} {'||a_final||':>14} {'residual 비중':>14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in sorted(rows, key=lambda r: r["beta4"]):
        share = (
            f"{row['residual_share'] * 100:>13.3f}%"
            if row["residual_share"] is not None else f"{'-':>14}"
        )

        def cell(value: Optional[float], width: int) -> str:
            return f"{value:>{width}.4f}" if value is not None else f"{'-':>{width}}"

        print(
            f"  {row['beta4']:>7.3f} {cell(row['norm_a_semantic'], 16)} "
            f"{cell(row['norm_residual'], 14)} "
            f"{cell(row['norm_scaled_residual'], 15)} "
            f"{cell(row['norm_a_final'], 14)} {share}"
        )

    print()
    print("  residual 비중 = ||beta4*P(e_c4)|| / ||a_semantic||")
    print("  이 값이 작으면 c4가 사실상 관여하지 않는 것이다.")


def check_reference(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("확인 — beta4=0이 hidden weight sweep의 D와 같은가")

    row = next((r for r in rows if r["beta4"] == BASELINE_BETA), None)

    if row is None:
        print("  beta4=0을 돌리지 않아 확인하지 못했습니다.")
        return {"checked": False}

    deviation = abs(row["val_top1"] - REFERENCE_D)
    passed = deviation <= REFERENCE_TOLERANCE

    print(f"  h-sweep D [1,.5,.5] : {REFERENCE_D * 100:.4f}%")
    print(f"  이번 beta4=0        : {row['val_top1'] * 100:.4f}%")
    print(f"  차이                : {deviation * 100:.4f}%p "
          f"(허용 {REFERENCE_TOLERANCE * 100:.2f}%p)")
    print()

    if passed:
        print("  => 같습니다. beta4=0이 c4 없는 계산과 일치합니다.")
    else:
        print("  => 허용치를 벗어납니다. 다만 GPU 비결정성이 있으므로")
        print("     이것만으로 오류라고 단정하지 않습니다. 중단하지 않습니다.")

    print()
    print("  beta4=0에서도 c4_proj를 만들지만 forward에 0을 곱하므로")
    print("  gradient가 0이고 c4 쪽은 학습되지 않습니다.")

    return {
        "checked": True,
        "reference": REFERENCE_D,
        "actual": row["val_top1"],
        "deviation": deviation,
        "tolerance": REFERENCE_TOLERANCE,
        "passed": passed,
    }


def print_comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("비교")

    best = max(rows, key=lambda r: r["val_top1"])

    table = [dict(b) for b in BASELINES]
    table.append({
        "config": f"c4 beta sweep best  beta4={best['beta4']:g}",
        "top1_accuracy": best["val_top1"],
        "note": f"hidden weight {best['hidden_weights']} 고정",
    })

    for row in sorted(table, key=lambda r: r["top1_accuracy"]):
        print(f"  {row['config']:<40} {row['top1_accuracy'] * 100:>8.3f}%")

    print()
    print("  Fixed weighted V1 reference(29.312%)는 Validation에서 L1/L2/L3")
    print("  score weight를 탐색해 얻은 development reference입니다.")
    print("  이번 sweep과 동일 조건의 결과가 아닙니다.")

    spread = (
        max(r["val_top1"] for r in rows) - min(r["val_top1"] for r in rows)
    )

    print()
    print(f"  이번 sweep 전체 폭 : {spread * 100:.3f}%p")

    if spread < 0.005:
        print("  V1의 seed 간 차이(약 0.5%p)보다 작습니다. beta4를 바꿔도")
        print("  성능이 사실상 변하지 않는다고 보는 편이 안전합니다.")

    return best


def write_readme(
    path: Path,
    rows: List[Dict[str, Any]],
    reference: Dict[str, Any],
    best: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    lines: List[str] = []
    add = lines.append

    w = HIDDEN_WEIGHTS

    add("# c4 beta sweep")
    add("")
    add("hidden weight를 고정하고 c4 identity residual의 세기만 비교했다.")
    add("")
    add(f"    a_semantic = ({w[0]:g}*h1 + {w[1]:g}*h2 + {w[2]:g}*h3) "
        f"/ {sum(w):g}")
    add("    a_final    = a_semantic + beta4 * P(e_c4)")
    add("")
    add("c4는 encoder token으로 들어가지 않는다. self-attention을 거치지")
    add("않고 article vector에 residual로만 붙는다.")
    add("")
    add(f"hidden weight `{best['hidden_weights']}`는 hidden weight sweep에서")
    add("seed42 기준 가장 높았던 조합이다. 다만 그 sweep의 조합 간 차이가")
    add("seed 노이즈보다 작았으므로 provisional 값이다.")
    add("")

    add("## 실행 정보")
    add("")
    add(f"- beta4        : {', '.join(f'{b:g}' for b in metadata['betas'])}")
    add(f"- hidden weight: `{best['hidden_weights']}` 고정")
    add(f"- seed         : {metadata['seed']} (from scratch)")
    add(f"- LR           : {metadata['learning_rate']}, "
        f"batch {metadata['batch_size']}")
    add(f"- max epoch    : {metadata['max_epochs']}, "
        f"patience {metadata['patience']}")
    add(f"- best 선택    : Validation Top-1")
    add(f"- git commit   : {metadata.get('git_commit')}")
    add(f"- 총 소요      : {format_duration(metadata['elapsed_seconds'])}")
    add("")
    add("네 arm이 같은 구조를 갖도록 beta4=0에서도 `c4_proj`를 만든다.")
    add("초기값과 parameter 수가 같아 차이가 beta4 하나로만 남는다.")
    add("beta4=0이면 forward에 0을 곱하므로 gradient도 0이고 c4 쪽은")
    add("학습되지 않는다.")
    add("")

    if reference.get("checked"):
        add("## beta4=0 확인")
        add("")
        add("| 항목 | 값 |")
        add("|---|---|")
        add(f"| h-sweep D `[1,.5,.5]` | {reference['reference'] * 100:.4f}% |")
        add(f"| 이번 beta4=0 | {reference['actual'] * 100:.4f}% |")
        add(f"| 차이 | {reference['deviation'] * 100:.4f}%p |")
        add(f"| 판정 | {'OK' if reference['passed'] else '허용치 밖'} |")
        add("")
        add("GPU 비결정성이 있으므로 허용치를 벗어나도 중단하지 않는다.")
        add("")

    add("## 결과")
    add("")
    add("| beta4 | best epoch | epochs | val Top-1 | MRR | nDCG@5 | AUC "
        "| val loss | 과적합폭 | beta4=0 대비 |")
    add("|---|---|---|---|---|---|---|---|---|---|")

    baseline = next((r for r in rows if r["beta4"] == BASELINE_BETA), None)

    for row in sorted(rows, key=lambda r: -r["val_top1"]):
        if baseline and row["beta4"] != BASELINE_BETA:
            delta = f"{(row['val_top1'] - baseline['val_top1']) * 100:+.3f}%p"
        else:
            delta = "기준"

        add(
            f"| {row['beta4']:g} | {row['best_epoch']} | {row['epochs_run']} | "
            f"{row['val_top1'] * 100:.3f}% | {row['mrr']:.4f} | "
            f"{row['ndcg5']:.4f} | {row['auc']:.4f} | {row['val_loss']:.4f} | "
            f"{row['overfit_gap'] * 100:.2f}%p | {delta} |"
        )

    add("")
    add("과적합폭 = best epoch의 train Top-1 − val Top-1")
    add("")

    add("## 첫 batch 항별 크기")
    add("")
    add("| beta4 | `||a_semantic||` | `||P(e_c4)||` | `||beta4*P(e_c4)||` "
        "| `||a_final||` | residual 비중 |")
    add("|---|---|---|---|---|---|")

    for row in sorted(rows, key=lambda r: r["beta4"]):
        def cell(value: Optional[float]) -> str:
            return f"{value:.4f}" if value is not None else "-"

        share = (
            f"{row['residual_share'] * 100:.3f}%"
            if row["residual_share"] is not None else "-"
        )

        add(
            f"| {row['beta4']:g} | {cell(row['norm_a_semantic'])} | "
            f"{cell(row['norm_residual'])} | "
            f"{cell(row['norm_scaled_residual'])} | "
            f"{cell(row['norm_a_final'])} | {share} |"
        )

    add("")
    add("residual 비중 = `||beta4*P(e_c4)|| / ||a_semantic||`. 이 값이")
    add("작으면 c4가 사실상 관여하지 않는 것이다.")
    add("")

    add("## 확률 / score 분리도")
    add("")
    add("| beta4 | positive prob | negative prob | score gap |")
    add("|---|---|---|---|")

    for row in sorted(rows, key=lambda r: r["beta4"]):
        add(
            f"| {row['beta4']:g} | {row['positive_prob']:.4f} | "
            f"{row['negative_prob']:.4f} | {row['score_gap']:.4f} |"
        )

    add("")
    add("무작위면 positive / negative prob이 둘 다 0.2다.")
    add("")

    add("## 비교")
    add("")
    add("| 구분 | Top-1 | 비고 |")
    add("|---|---|---|")

    table = [dict(b) for b in BASELINES]
    table.append({
        "config": f"**c4 beta sweep best** beta4={best['beta4']:g}",
        "top1_accuracy": best["val_top1"],
        "note": f"hidden weight {best['hidden_weights']} 고정",
    })

    for row in sorted(table, key=lambda r: r["top1_accuracy"]):
        add(f"| {row['config']} | {row['top1_accuracy'] * 100:.3f}% | "
            f"{row['note']} |")

    add("")
    spread = (
        max(r["val_top1"] for r in rows) - min(r["val_top1"] for r in rows)
    )
    add(f"이번 sweep 전체 폭은 {spread * 100:.3f}%p다. V1의 seed 간 차이가")
    add("약 0.5%p였으므로, 이보다 작은 차이는 seed 하나로 뒤집힐 수 있다.")
    add("")

    add("## 하지 않은 것")
    add("")
    add("- dropout / weight decay 변경")
    add("- score weighting 변경")
    add("- hidden weight 변경 (고정)")
    add("- c4를 encoder token으로 넣기")
    add("- MLP scorer, hybrid, learnable beta")
    add("- Test 데이터 사용")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="c4 identity residual의 beta4만 비교한다."
    )
    parser.add_argument(
        "--out",
        default="sweep_out/ebnerd_v2/priority2_direct/c4_beta_sweep",
    )
    parser.add_argument(
        "--priority0-dir",
        default="sweep_out/ebnerd_v2/priority0_shuffled/seed42",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument(
        "--only", nargs="+", type=float, default=None, metavar="BETA",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-interval", type=int, default=200)

    args = parser.parse_args()

    started_at = time.time()

    out_root = resolve_path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    betas = list(BETAS)

    if args.only:
        betas = [b for b in BETAS if any(abs(b - o) < 1e-12 for o in args.only)]

        if not betas:
            print(f"해당하는 beta가 없습니다: {args.only}")
            return 1

    w = HIDDEN_WEIGHTS

    section("c4 beta sweep")
    print(f"  출력          : {out_root}")
    print(f"  hidden weight : [{w[0]:g}, {w[1]:g}, {w[2]:g}] 고정")
    print(f"  a_semantic    = ({w[0]:g}*h1 + {w[1]:g}*h2 + {w[2]:g}*h3) "
          f"/ {sum(w):g}")
    print(f"  a_final       = a_semantic + beta4 * P(e_c4)")
    print(f"  beta4         : {', '.join(f'{b:g}' for b in betas)}")
    print(f"  고정          : seed {args.seed}, LR {args.learning_rate}, "
          f"batch {args.batch_size}, max_epochs {args.max_epochs}, "
          f"patience {args.patience}")
    print()
    print("  c4는 encoder token으로 넣지 않습니다. Test를 쓰지 않습니다.")

    rows: List[Dict[str, Any]] = []

    for index, beta in enumerate(betas, start=1):
        out_dir = beta_dir(out_root, beta)
        log_path = out_root / f"beta_{beta:.3f}.log"

        section(f"[{index}/{len(betas)}]  beta4 = {beta:g}")

        if (out_dir / "best_metrics.json").exists():
            print(f"  이미 끝난 조합입니다. 건너뜁니다: {out_dir}")
            rows.append(collect(out_dir, beta, None))
            print(f"  val Top-1 {rows[-1]['val_top1'] * 100:.3f}%")
            continue

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  폴더가 비어 있지 않은데 best_metrics.json이 없습니다.")
            print(f"  이전 실행이 중간에 멈춘 것 같습니다: {out_dir}")
            print(f"  폴더를 지우고 다시 실행하세요.")
            return 1

        print(f"  출력 : {out_dir}")
        print(f"  로그 : {log_path}")
        print(f"  실행 중...", flush=True)

        combo_started = time.time()
        code = run_one(out_dir, beta, args, log_path)
        elapsed = time.time() - combo_started

        if code != 0:
            print(f"  실패 (exit {code}). 로그를 확인하세요: {log_path}")
            return 1

        rows.append(collect(out_dir, beta, elapsed))

        print(f"  완료 ({format_duration(elapsed)})  "
              f"val Top-1 {rows[-1]['val_top1'] * 100:.3f}%  "
              f"best epoch {rows[-1]['best_epoch']}")

    reference = check_reference(rows)
    print_results(rows)
    print_norms(rows)
    best = print_comparison(rows)

    section("저장")

    csv_rows = [
        {k: v for k, v in row.items() if k != "out_dir"}
        for row in sorted(rows, key=lambda r: r["beta4"])
    ]

    write_csv(out_root / "sweep_results.csv", csv_rows)

    metadata = {
        "experiment": "c4_beta_sweep",
        "hidden_weights": list(HIDDEN_WEIGHTS),
        "formula_semantic": (
            f"({HIDDEN_WEIGHTS[0]:g}*h1 + {HIDDEN_WEIGHTS[1]:g}*h2 + "
            f"{HIDDEN_WEIGHTS[2]:g}*h3) / {sum(HIDDEN_WEIGHTS):g}"
        ),
        "formula_final": "a_semantic + beta4 * P(e_c4)",
        "betas": betas,
        "c4_in_encoder": False,
        "beta4_learnable": False,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "selection_criterion": "validation top1_accuracy",
        "reference_check": reference,
        "best": {"beta4": best["beta4"], "val_top1": best["val_top1"]},
        "results": rows,
        "did_not_do": [
            "dropout / weight decay 변경",
            "score weighting 변경",
            "hidden weight 변경",
            "c4를 encoder token으로 넣기",
            "MLP scorer, hybrid, learnable beta",
            "Test 사용",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 1),
        "git_commit": git_commit_hash(),
        "python": sys.version.split()[0],
    }

    (out_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    write_readme(out_root / "README.md", rows, reference, best, metadata)

    for name in sorted(p.name for p in out_root.iterdir() if p.is_file()):
        print(f"  {out_root / name}")

    print()
    print(f"  총 소요 {format_duration(time.time() - started_at)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
