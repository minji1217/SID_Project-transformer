"""dropout x weight decay만 비교한다. 모델 구조는 고정한다.

지금까지 구조를 여섯 번 바꿨는데 전부 26.1 ~ 26.7% 안에 머물렀다.
반면 과적합은 일관되게 심했다. 모든 arm이 epoch 1에서 best를 찍고
train Top-1이 48%까지 오르는 동안 val Top-1은 내려갔다.

그래서 이번에는 구조를 그대로 두고 정규화만 본다.

고정 구조

    a_semantic = (1.0*h1 + 0.5*h2 + 0.5*h3) / 2.0
    a_final    = a_semantic + 0.1 * P(e_c4)
    candidate  = Linear(768, 256) of concat(c1, c2, c3)
    score      = u^T W v

c4는 encoder token으로 넣지 않는다.

dropout은 gin의 NewsEncoderDecoderTransformer.dropout_rate로 들어가
encoder와 attention에 걸린다. dropout은 parameter가 없으므로 25개
arm의 초기값은 전부 같다.

    cd Transformer
    python -m analysis.run_regularization_sweep \
        --out sweep_out/ebnerd_v2/priority2_direct/regularization_sweep
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

# 고정 구조
HIDDEN_WEIGHTS: Tuple[float, float, float] = (1.0, 0.5, 0.5)
BETA4 = 0.1

DROPOUTS: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3]
WEIGHT_DECAYS: List[float] = [0.0, 0.001, 0.01, 0.05, 0.1]

# dropout 0 / weight decay 0이 지금까지의 설정이다.
BASELINE = (0.0, 0.0)

# 그 조합은 c4 beta sweep의 beta4=0.1과 같은 계산이어야 한다.
REFERENCE = 0.26689
REFERENCE_TOLERANCE = 5e-4   # 0.05%p. GPU 비결정성을 감안한 값

BASELINES: List[Dict[str, Any]] = [
    {"config": "P2 c1234 mean", "top1_accuracy": 0.26137,
     "note": "history에 c4를 encoder token으로 넣음"},
    {"config": "P2 c123 mean", "top1_accuracy": 0.26650,
     "note": "mean code path"},
    {"config": "h-sweep D [1,.5,.5]", "top1_accuracy": 0.266995,
     "note": "hidden weight sweep 최고"},
    {"config": "beta sweep beta4=0.1", "top1_accuracy": 0.26689,
     "note": "이번 sweep의 dropout=0, wd=0과 같은 계산"},
    {"config": "V1", "top1_accuracy": 0.27059,
     "note": "L1+L2+L3 생성확률, seed42"},
    {"config": "Learnable alpha", "top1_accuracy": 0.27166,
     "note": "alpha를 Train에서 학습"},
    {"config": "history overlap c1 (규칙)", "top1_accuracy": 0.2594,
     "note": "학습 없는 한 줄 규칙. 실질적 기준선"},
    {"config": "Fixed weighted V1 reference", "top1_accuracy": 0.29312,
     "note": ("Validation에서 L1/L2/L3 score weight를 탐색해 얻은 "
              "development reference. 동일 조건의 결과가 아니다.")},
]


def section(title: str) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


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


def combo_dir(root: Path, dropout: float, weight_decay: float) -> Path:
    return root / f"do{dropout:.2f}_wd{weight_decay:.4f}" / "seed42"


def combos(
    dropouts: List[float], weight_decays: List[float]
) -> List[Tuple[float, float]]:
    """weight decay를 바깥 축으로 돈다.

    wd=0 행이 먼저 끝나므로, 중간에 멈춰도 dropout 단독 효과는 볼 수 있다.
    """
    return [(d, w) for w in weight_decays for d in dropouts]


def run_one(
    out_dir: Path, dropout: float, weight_decay: float,
    args: argparse.Namespace, log_path: Path,
) -> int:
    command = [
        sys.executable, "-u", "-m", RUN_SCRIPT,
        "--priority0-dir", args.priority0_dir,
        "--out", str(out_dir),
        "--history-levels", "3",
        "--history-mode", "weighted",
        "--level-weights", *(f"{w:g}" for w in HIDDEN_WEIGHTS),
        "--use-c4-identity",
        "--beta4", f"{BETA4:g}",
        "--scorer", "bilinear",
        "--dropout", f"{dropout:g}",
        "--weight-decay", f"{weight_decay:g}",
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


def collect(
    out_dir: Path, dropout: float, weight_decay: float,
    elapsed: Optional[float],
) -> Dict[str, Any]:
    best = json.loads(
        (out_dir / "best_metrics.json").read_text(encoding="utf-8")
    )
    history = pd.read_csv(
        out_dir / "training_history.csv", encoding="utf-8-sig"
    )
    history.columns = [c.strip().lstrip("﻿") for c in history.columns]

    best_epoch = int(best["best_epoch"])
    row = history[history["epoch"] == best_epoch].iloc[0]
    validation = best["validation"]

    train_top1 = float(row["train_top1"])
    val_top1 = float(validation["top1_accuracy"])

    return {
        "dropout": dropout,
        "weight_decay": weight_decay,
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
        "elapsed_seconds": round(elapsed, 1) if elapsed else None,
        "out_dir": str(out_dir),
    }


def print_ranked(rows: List[Dict[str, Any]]) -> None:
    section("결과  (Validation Top-1 내림차순)")

    baseline = next(
        (r for r in rows
         if (r["dropout"], r["weight_decay"]) == BASELINE), None
    )

    header = (
        f"  {'dropout':>8} {'wd':>7} {'best':>5} {'run':>4} "
        f"{'val Top-1':>10} {'MRR':>8} {'nDCG@5':>8} {'AUC':>8} "
        f"{'val loss':>9} {'과적합폭':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) + 8))

    for row in sorted(rows, key=lambda r: -r["val_top1"]):
        key = (row["dropout"], row["weight_decay"])

        if baseline and key != BASELINE:
            delta = f"  {(row['val_top1'] - baseline['val_top1']) * 100:+.3f}%p"
        else:
            delta = "  기준"

        print(
            f"  {row['dropout']:>8.2f} {row['weight_decay']:>7.4f} "
            f"{row['best_epoch']:>5} {row['epochs_run']:>4} "
            f"{row['val_top1'] * 100:>9.3f}% {row['mrr']:>8.4f} "
            f"{row['ndcg5']:>8.4f} {row['auc']:>8.4f} "
            f"{row['val_loss']:>9.4f} {row['overfit_gap'] * 100:>8.2f}%p"
            f"{delta}"
        )


def print_grid(rows: List[Dict[str, Any]], key: str, title: str,
               scale: float = 1.0, fmt: str = "7.3f") -> None:
    section(title)

    lookup = {(r["dropout"], r["weight_decay"]): r for r in rows}

    dropouts = sorted({r["dropout"] for r in rows})
    decays = sorted({r["weight_decay"] for r in rows})

    corner = "dropout / wd"
    print(f"  {corner:>14}", end="")
    for wd in decays:
        print(f" {wd:>9.4f}", end="")
    print()
    print("  " + "-" * (14 + 10 * len(decays)))

    for dropout in dropouts:
        print(f"  {dropout:>14.2f}", end="")

        for wd in decays:
            row = lookup.get((dropout, wd))

            if row is None:
                print(f" {'-':>9}", end="")
            else:
                print(f" {row[key] * scale:>{fmt}}", end="")

        print()


def print_overfit_trend(rows: List[Dict[str, Any]]) -> None:
    section("과적합이 완화되었는가")

    baseline = next(
        (r for r in rows
         if (r["dropout"], r["weight_decay"]) == BASELINE), None
    )

    if baseline is None:
        print("  baseline(dropout 0, wd 0)을 돌리지 않아 비교하지 못합니다.")
        return

    print(f"  baseline (dropout 0, wd 0)")
    print(f"    best epoch {baseline['best_epoch']}, "
          f"과적합폭 {baseline['overfit_gap'] * 100:.2f}%p, "
          f"val Top-1 {baseline['val_top1'] * 100:.3f}%")
    print()

    later = [r for r in rows if r["best_epoch"] > baseline["best_epoch"]]
    tighter = [
        r for r in rows if r["overfit_gap"] < baseline["overfit_gap"]
    ]

    print(f"  best epoch가 baseline보다 뒤로 간 조합 : "
          f"{len(later)} / {len(rows)}")
    print(f"  과적합폭이 baseline보다 줄어든 조합     : "
          f"{len(tighter)} / {len(rows)}")

    if later:
        best_epoch_row = max(later, key=lambda r: r["best_epoch"])
        print()
        print(f"  best epoch가 가장 뒤로 간 조합 : "
              f"dropout {best_epoch_row['dropout']:g}, "
              f"wd {best_epoch_row['weight_decay']:g} "
              f"-> epoch {best_epoch_row['best_epoch']}")

    if tighter:
        tightest = min(tighter, key=lambda r: r["overfit_gap"])
        print(f"  과적합폭이 가장 작은 조합      : "
              f"dropout {tightest['dropout']:g}, "
              f"wd {tightest['weight_decay']:g} "
              f"-> {tightest['overfit_gap'] * 100:.2f}%p "
              f"(val Top-1 {tightest['val_top1'] * 100:.3f}%)")

    # 과적합이 줄었는데 성능도 올랐는가
    both = [
        r for r in tighter if r["val_top1"] > baseline["val_top1"]
    ]

    print()

    if both:
        print(f"  과적합폭이 줄면서 val Top-1도 오른 조합 : {len(both)}개")
        best = max(both, key=lambda r: r["val_top1"])
        print(f"    최고: dropout {best['dropout']:g}, "
              f"wd {best['weight_decay']:g} -> "
              f"{best['val_top1'] * 100:.3f}% "
              f"({(best['val_top1'] - baseline['val_top1']) * 100:+.3f}%p), "
              f"과적합폭 {best['overfit_gap'] * 100:.2f}%p")
    else:
        print("  과적합폭이 줄면서 val Top-1도 오른 조합이 없습니다.")
        print("  정규화가 과적합은 줄여도 일반화로 이어지지 않았다는 뜻입니다.")


def check_reference(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("확인 — dropout 0 / wd 0이 beta sweep의 beta4=0.1과 같은가")

    row = next(
        (r for r in rows
         if (r["dropout"], r["weight_decay"]) == BASELINE), None
    )

    if row is None:
        print("  baseline을 돌리지 않아 확인하지 못했습니다.")
        return {"checked": False}

    deviation = abs(row["val_top1"] - REFERENCE)
    passed = deviation <= REFERENCE_TOLERANCE

    print(f"  beta sweep beta4=0.1 : {REFERENCE * 100:.4f}%")
    print(f"  이번 dropout0 / wd0  : {row['val_top1'] * 100:.4f}%")
    print(f"  차이                 : {deviation * 100:.4f}%p "
          f"(허용 {REFERENCE_TOLERANCE * 100:.2f}%p)")
    print()

    if passed:
        print("  => 같습니다. 구조가 그대로이고 정규화만 추가된 것이 맞습니다.")
    else:
        print("  => 허용치를 벗어납니다. GPU 비결정성이 있으므로 중단하지")
        print("     않지만, 폭이 크면 구조가 달라졌는지 확인해야 합니다.")

    return {
        "checked": True, "reference": REFERENCE, "actual": row["val_top1"],
        "deviation": deviation, "tolerance": REFERENCE_TOLERANCE,
        "passed": passed,
    }


def print_comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("비교")

    best = max(rows, key=lambda r: r["val_top1"])

    table = [dict(b) for b in BASELINES]
    table.append({
        "config": (
            f"regularization best  "
            f"dropout {best['dropout']:g} / wd {best['weight_decay']:g}"
        ),
        "top1_accuracy": best["val_top1"],
        "note": "이번 sweep",
    })

    for row in sorted(table, key=lambda r: r["top1_accuracy"]):
        print(f"  {row['config']:<48} {row['top1_accuracy'] * 100:>8.3f}%")

    spread = (
        max(r["val_top1"] for r in rows) - min(r["val_top1"] for r in rows)
    )

    print()
    print(f"  이번 sweep 전체 폭 : {spread * 100:.3f}%p")
    print()
    print("  Fixed weighted V1 reference(29.312%)는 Validation에서 L1/L2/L3")
    print("  score weight를 탐색해 얻은 development reference입니다.")
    print("  이번 sweep과 동일 조건의 결과가 아닙니다.")
    print()
    print("  history overlap c1(25.94%)은 학습 없는 한 줄 규칙입니다.")
    print("  이 값을 확실히 넘는지가 실질적인 기준입니다.")

    return best


def write_readme(
    path: Path, rows: List[Dict[str, Any]], reference: Dict[str, Any],
    best: Dict[str, Any], metadata: Dict[str, Any],
) -> None:
    lines: List[str] = []
    add = lines.append

    w = HIDDEN_WEIGHTS

    add("# dropout x weight decay sweep")
    add("")
    add("모델 구조는 고정하고 정규화만 비교했다.")
    add("")
    add(f"    a_semantic = ({w[0]:g}*h1 + {w[1]:g}*h2 + {w[2]:g}*h3) "
        f"/ {sum(w):g}")
    add(f"    a_final    = a_semantic + {BETA4:g} * P(e_c4)")
    add("    candidate  = Linear(768, 256) of concat(c1, c2, c3)")
    add("    score      = u^T W v")
    add("")
    add("c4는 encoder token으로 넣지 않는다.")
    add("")
    add("dropout은 `NewsEncoderDecoderTransformer.dropout_rate`로 들어가")
    add("encoder와 attention에 걸린다. dropout에는 parameter가 없으므로")
    add(f"{len(rows)}개 arm의 초기값은 전부 같다.")
    add("")

    add("## 실행 정보")
    add("")
    add(f"- dropout      : {', '.join(f'{d:g}' for d in metadata['dropouts'])}")
    add(f"- weight decay : "
        f"{', '.join(f'{w:g}' for w in metadata['weight_decays'])}")
    add(f"- 조합 수      : {len(rows)}")
    add(f"- seed         : {metadata['seed']} (from scratch)")
    add(f"- LR           : {metadata['learning_rate']}, "
        f"batch {metadata['batch_size']}")
    add(f"- max epoch    : {metadata['max_epochs']}, "
        f"patience {metadata['patience']}")
    add(f"- best 선택    : Validation Top-1")
    add(f"- git commit   : {metadata.get('git_commit')}")
    add(f"- 총 소요      : {format_duration(metadata['elapsed_seconds'])}")
    add("")

    if reference.get("checked"):
        add("## baseline 확인")
        add("")
        add("| 항목 | 값 |")
        add("|---|---|")
        add(f"| beta sweep beta4=0.1 | {reference['reference'] * 100:.4f}% |")
        add(f"| 이번 dropout 0 / wd 0 | {reference['actual'] * 100:.4f}% |")
        add(f"| 차이 | {reference['deviation'] * 100:.4f}%p |")
        add(f"| 판정 | {'OK' if reference['passed'] else '허용치 밖'} |")
        add("")

    add("## 결과 (Validation Top-1 내림차순)")
    add("")
    add("| dropout | wd | best epoch | epochs | val Top-1 | MRR | nDCG@5 "
        "| AUC | val loss | 과적합폭 | baseline 대비 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")

    baseline = next(
        (r for r in rows
         if (r["dropout"], r["weight_decay"]) == BASELINE), None
    )

    for row in sorted(rows, key=lambda r: -r["val_top1"]):
        key = (row["dropout"], row["weight_decay"])

        if baseline and key != BASELINE:
            delta = f"{(row['val_top1'] - baseline['val_top1']) * 100:+.3f}%p"
        else:
            delta = "기준"

        add(
            f"| {row['dropout']:g} | {row['weight_decay']:g} | "
            f"{row['best_epoch']} | {row['epochs_run']} | "
            f"{row['val_top1'] * 100:.3f}% | {row['mrr']:.4f} | "
            f"{row['ndcg5']:.4f} | {row['auc']:.4f} | "
            f"{row['val_loss']:.4f} | {row['overfit_gap'] * 100:.2f}%p | "
            f"{delta} |"
        )

    add("")
    add("과적합폭 = best epoch의 train Top-1 − val Top-1")
    add("")

    # 격자 표
    lookup = {(r["dropout"], r["weight_decay"]): r for r in rows}
    dropouts = sorted({r["dropout"] for r in rows})
    decays = sorted({r["weight_decay"] for r in rows})

    for key, title, scale, digits in (
        ("val_top1", "val Top-1 (%)", 100.0, 3),
        ("best_epoch", "best epoch", 1.0, 0),
        ("overfit_gap", "과적합폭 (%p)", 100.0, 2),
    ):
        add(f"## 격자 — {title}")
        add("")
        add("| dropout \\ wd | " + " | ".join(f"{w:g}" for w in decays) + " |")
        add("|---" * (len(decays) + 1) + "|")

        for dropout in dropouts:
            cells = []

            for wd in decays:
                row = lookup.get((dropout, wd))
                cells.append(
                    "-" if row is None
                    else f"{row[key] * scale:.{digits}f}"
                )

            add(f"| **{dropout:g}** | " + " | ".join(cells) + " |")

        add("")

    add("## 비교")
    add("")
    add("| 구분 | Top-1 | 비고 |")
    add("|---|---|---|")

    table = [dict(b) for b in BASELINES]
    table.append({
        "config": (
            f"**regularization best** dropout {best['dropout']:g} / "
            f"wd {best['weight_decay']:g}"
        ),
        "top1_accuracy": best["val_top1"],
        "note": "이번 sweep",
    })

    for row in sorted(table, key=lambda r: r["top1_accuracy"]):
        add(f"| {row['config']} | {row['top1_accuracy'] * 100:.3f}% | "
            f"{row['note']} |")

    add("")
    spread = (
        max(r["val_top1"] for r in rows) - min(r["val_top1"] for r in rows)
    )
    add(f"이번 sweep 전체 폭은 {spread * 100:.3f}%p다. V1의 seed 간 차이가")
    add("약 0.5%p였으므로 그보다 작은 차이는 seed 하나로 뒤집힐 수 있다.")
    add("")

    add("## 하지 않은 것")
    add("")
    add("- LR, hidden weight, beta4 변경")
    add("- score weighting, normalization 변경")
    add("- MLP scorer, hybrid")
    add("- 구조 변경 일체")
    add("- Test 데이터 사용")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="dropout x weight decay sweep. 구조는 고정한다."
    )
    parser.add_argument(
        "--out",
        default="sweep_out/ebnerd_v2/priority2_direct/regularization_sweep",
    )
    parser.add_argument(
        "--priority0-dir",
        default="sweep_out/ebnerd_v2/priority0_shuffled/seed42",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument(
        "--dropouts", nargs="+", type=float, default=None,
        help="기본 0 0.05 0.1 0.2 0.3",
    )
    parser.add_argument(
        "--weight-decays", nargs="+", type=float, default=None,
        help="기본 0 0.001 0.01 0.05 0.1",
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

    dropouts = args.dropouts if args.dropouts else DROPOUTS
    decays = args.weight_decays if args.weight_decays else WEIGHT_DECAYS

    pairs = combos(dropouts, decays)

    w = HIDDEN_WEIGHTS

    section("dropout x weight decay sweep")
    print(f"  출력          : {out_root}")
    print(f"  고정 구조     : a_semantic = ({w[0]:g}*h1 + {w[1]:g}*h2 + "
          f"{w[2]:g}*h3) / {sum(w):g}")
    print(f"                  a_final = a_semantic + {BETA4:g} * P(e_c4)")
    print(f"  dropout       : {', '.join(f'{d:g}' for d in dropouts)}")
    print(f"  weight decay  : {', '.join(f'{d:g}' for d in decays)}")
    print(f"  조합 수       : {len(pairs)}")
    print(f"  고정          : seed {args.seed}, LR {args.learning_rate}, "
          f"batch {args.batch_size}, max_epochs {args.max_epochs}, "
          f"patience {args.patience}")
    print()
    print("  weight decay를 바깥 축으로 돕니다. wd=0 행이 먼저 끝나므로")
    print("  중간에 멈춰도 dropout 단독 효과는 볼 수 있습니다.")
    print()
    print("  정규화가 듣기 시작하면 best epoch가 뒤로 밀려 한 조합당")
    print("  시간이 늘어납니다. 전체 소요가 크게 달라질 수 있습니다.")

    rows: List[Dict[str, Any]] = []

    for index, (dropout, weight_decay) in enumerate(pairs, start=1):
        out_dir = combo_dir(out_root, dropout, weight_decay)
        log_path = out_root / f"do{dropout:.2f}_wd{weight_decay:.4f}.log"

        section(
            f"[{index}/{len(pairs)}]  dropout {dropout:g}  "
            f"weight decay {weight_decay:g}"
        )

        if (out_dir / "best_metrics.json").exists():
            print(f"  이미 끝난 조합입니다. 건너뜁니다.")
            rows.append(collect(out_dir, dropout, weight_decay, None))
            print(f"  val Top-1 {rows[-1]['val_top1'] * 100:.3f}%  "
                  f"best epoch {rows[-1]['best_epoch']}")
            continue

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  폴더가 비어 있지 않은데 best_metrics.json이 없습니다.")
            print(f"  이전 실행이 중간에 멈춘 것 같습니다: {out_dir}")
            print(f"  폴더를 지우고 다시 실행하세요.")
            return 1

        print(f"  로그 : {log_path}")
        print(f"  실행 중...", flush=True)

        combo_started = time.time()
        code = run_one(out_dir, dropout, weight_decay, args, log_path)
        elapsed = time.time() - combo_started

        if code != 0:
            print(f"  실패 (exit {code}). 로그를 확인하세요: {log_path}")
            return 1

        rows.append(collect(out_dir, dropout, weight_decay, elapsed))

        done = len(rows)
        remaining = len(pairs) - index
        average = (time.time() - started_at) / index

        print(f"  완료 ({format_duration(elapsed)})  "
              f"val Top-1 {rows[-1]['val_top1'] * 100:.3f}%  "
              f"best epoch {rows[-1]['best_epoch']}  "
              f"epochs {rows[-1]['epochs_run']}")

        if remaining:
            print(f"  남은 {remaining}개 예상 "
                  f"{format_duration(average * remaining)}")

    reference = check_reference(rows)
    print_ranked(rows)

    print_grid(rows, "val_top1", "격자 — val Top-1 (%)", 100.0, "9.3f")
    print_grid(rows, "best_epoch", "격자 — best epoch", 1.0, "9.0f")
    print_grid(rows, "overfit_gap", "격자 — 과적합폭 (%p)", 100.0, "9.2f")

    print_overfit_trend(rows)
    best = print_comparison(rows)

    section("저장")

    csv_rows = [
        {k: v for k, v in row.items() if k != "out_dir"}
        for row in sorted(rows, key=lambda r: -r["val_top1"])
    ]

    write_csv(out_root / "sweep_results.csv", csv_rows)

    metadata = {
        "experiment": "regularization_sweep",
        "hidden_weights": list(HIDDEN_WEIGHTS),
        "beta4": BETA4,
        "formula_semantic": (
            f"({w[0]:g}*h1 + {w[1]:g}*h2 + {w[2]:g}*h3) / {sum(w):g}"
        ),
        "formula_final": f"a_semantic + {BETA4:g} * P(e_c4)",
        "dropouts": dropouts,
        "weight_decays": decays,
        "num_combos": len(pairs),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "selection_criterion": "validation top1_accuracy",
        "reference_check": reference,
        "best": {
            "dropout": best["dropout"],
            "weight_decay": best["weight_decay"],
            "val_top1": best["val_top1"],
            "best_epoch": best["best_epoch"],
        },
        "results": rows,
        "did_not_do": [
            "LR / hidden weight / beta4 변경",
            "score weighting, normalization 변경",
            "MLP scorer, hybrid",
            "구조 변경 일체",
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
