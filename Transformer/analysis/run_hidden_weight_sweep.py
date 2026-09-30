"""history hidden state aggregation weight만 비교한다.

    a_semantic = (w1*h1 + w2*h2 + w3*h3) / (w1 + w2 + w3)

h1/h2/h3는 history 기사의 c1/c2/c3 위치에서 나온 encoder hidden이다.
이 비율 하나만 바꾸고 나머지는 전부 고정한다.

c4는 완전히 제외한다. encoder token으로도, identity residual로도 쓰지
않는다 (beta4 = 0). c1/c2/c3 weighting 효과만 분리해서 보기 위해서다.

[1, 0.5, 0.1]은 V1의 L1/L2/L3 score weight에서 나온 값이다. hidden
vector에 대한 최적 weight라는 뜻이 아니다. 이번 sweep은 hidden state
aggregation에 맞는 비율을 따로 찾는 것이다.

    cd Transformer
    python -m analysis.run_hidden_weight_sweep \
        --out sweep_out/ebnerd_v2/priority2_direct/hidden_weight_sweep
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

# (라벨, (w1, w2, w3), 메모)
COMBOS: List[Tuple[str, Tuple[float, float, float], str]] = [
    ("A", (1.00, 1.00, 1.00), "기존 mean baseline"),
    ("B", (1.00, 0.75, 0.50), ""),
    ("C", (1.00, 0.75, 0.25), ""),
    ("D", (1.00, 0.50, 0.50), ""),
    ("E", (1.00, 0.50, 0.25), ""),
    ("F", (1.00, 0.50, 0.10), "V1 score weight에서 옮겨온 비율"),
    ("G", (1.00, 0.50, 0.00), "c3 완전 제외"),
]

# A는 기존 c123 mean과 같아야 한다.
#
# weighted [1,1,1]은 mean과 수식이 같고 CPU forward도 비트 단위로
# 같지만, GPU에서는 곱셈/합/나눗셈 순서가 달라 미세한 차이가 나고
# step이 쌓이면서 누적된다. 그래서 A는 아예 mean 경로를 그대로
# 호출한다. 그러면 기존 c123 mean과 완전히 같은 실행이다.
REPRODUCE_LABEL = "A"
REPRODUCE_TARGET = 0.2664974826395972

# train_transformer.set_seed()는 cudnn.deterministic을 켜지 않는다.
# 같은 설정을 두 번 돌려도 GPU에서 결과가 달라질 수 있으므로 허용치를
# 그 폭보다 크게 잡는다. 실제 폭은 analysis/compare_direct_runs.py로 잰다.
REPRODUCE_TOLERANCE = 5e-4   # 0.05%p

# 비교표. sweep과 같은 조건이 아닌 행은 표시해 둔다.
BASELINES: List[Dict[str, Any]] = [
    {"config": "P2 c1234 mean", "top1_accuracy": 0.26137,
     "note": "history에 c4를 encoder token으로 넣음"},
    {"config": "P2 c123 mean [1,1,1]", "top1_accuracy": 0.26650,
     "note": "이번 sweep의 A와 같은 설정"},
    {"config": "P2 weighted [1,.5,.1] + 0.01*c4", "top1_accuracy": 0.26677,
     "note": "참고만. c4 residual이 들어간 설정"},
    {"config": "V1", "top1_accuracy": 0.27059,
     "note": "L1+L2+L3 생성확률, seed42"},
    {"config": "Learnable alpha", "top1_accuracy": 0.27166,
     "note": "alpha를 Train에서 학습"},
    {"config": "Fixed weighted V1 reference", "top1_accuracy": 0.29312,
     "note": ("Validation에서 L1/L2/L3 score weight를 탐색해 얻은 "
              "development reference. 이번 hidden-weight sweep과 "
              "동일 조건의 결과가 아니다.")},
]


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


def combo_dir(root: Path, label: str, weights: Tuple[float, float, float]) -> Path:
    name = f"{label}_{weights[0]:.2f}_{weights[1]:.2f}_{weights[2]:.2f}"
    return root / name / "seed42"


def run_one(
    out_dir: Path,
    weights: Tuple[float, float, float],
    args: argparse.Namespace,
    log_path: Path,
) -> int:
    # 가중치가 전부 같으면 mean 경로를 그대로 쓴다. 수식은 같지만
    # 연산 순서가 달라 GPU에서 미세하게 어긋나기 때문이다. A가 기존
    # c123 mean과 완전히 같은 실행이 되도록 한다.
    uniform = len(set(weights)) == 1

    command = [
        sys.executable, "-u", "-m", RUN_SCRIPT,
        "--priority0-dir", args.priority0_dir,
        "--out", str(out_dir),
        "--history-levels", "3",
        "--history-mode", "mean" if uniform else "weighted",
    ]

    if not uniform:
        command += ["--level-weights", *(f"{w:g}" for w in weights)]

    command += [
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

    # --use-c4-identity를 주지 않으므로 beta4 = 0이다.

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()

        process = subprocess.run(
            command, cwd=str(BASE_DIR),
            stdout=log_file, stderr=subprocess.STDOUT,
        )

    return process.returncode


def collect(
    out_dir: Path,
    label: str,
    weights: Tuple[float, float, float],
    note: str,
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
        "label": label,
        "w1": weights[0], "w2": weights[1], "w3": weights[2],
        "weights": f"[{weights[0]:g}, {weights[1]:g}, {weights[2]:g}]",
        "beta4": 0.0,
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
        "note": note,
        "out_dir": str(out_dir),
    }


def print_results(rows: List[Dict[str, Any]]) -> None:
    section("sweep 결과  (Validation Top-1 내림차순)")

    ordered = sorted(rows, key=lambda r: -r["val_top1"])

    header = (
        f"  {'':<3} {'weights':<18} {'best':>5} {'run':>4} "
        f"{'val Top-1':>10} {'MRR':>8} {'nDCG@5':>8} {'AUC':>8} "
        f"{'val loss':>9} {'과적합폭':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) + 6))

    baseline = next((r for r in rows if r["label"] == REPRODUCE_LABEL), None)

    for row in ordered:
        delta = ""

        if baseline and row["label"] != REPRODUCE_LABEL:
            delta = f"  {(row['val_top1'] - baseline['val_top1']) * 100:+.3f}%p"
        elif row["label"] == REPRODUCE_LABEL:
            delta = "  기준"

        print(
            f"  {row['label']:<3} {row['weights']:<18} "
            f"{row['best_epoch']:>5} {row['epochs_run']:>4} "
            f"{row['val_top1'] * 100:>9.3f}% {row['mrr']:>8.4f} "
            f"{row['ndcg5']:>8.4f} {row['auc']:>8.4f} "
            f"{row['val_loss']:>9.4f} {row['overfit_gap'] * 100:>8.2f}%p"
            f"{delta}"
        )

    print()
    print("  과적합폭 = best epoch의 train Top-1 - val Top-1")


def check_reproduction(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section(f"재현 확인 — {REPRODUCE_LABEL} [1,1,1]이 기존 c123 mean과 같은가")

    row = next((r for r in rows if r["label"] == REPRODUCE_LABEL), None)

    if row is None:
        print("  A를 돌리지 않아 확인하지 못했습니다.")
        return {"checked": False}

    deviation = abs(row["val_top1"] - REPRODUCE_TARGET)
    passed = deviation <= REPRODUCE_TOLERANCE

    print(f"  기존 c123 mean : {REPRODUCE_TARGET * 100:.4f}%")
    print(f"  이번 A         : {row['val_top1'] * 100:.4f}%")
    print(f"  차이           : {deviation * 100:.4f}%p "
          f"(허용 {REPRODUCE_TOLERANCE * 100:.2f}%p)")
    print()

    if passed:
        print("  => 재현됩니다. weighted 경로가 mean과 같게 동작합니다.")
    else:
        print("  => 재현되지 않습니다. 나머지 결과를 해석하기 전에")
        print("     원인을 먼저 확인하세요. weight 외의 무언가가 달라졌습니다.")

    return {
        "checked": True,
        "target": REPRODUCE_TARGET,
        "actual": row["val_top1"],
        "deviation": deviation,
        "tolerance": REPRODUCE_TOLERANCE,
        "passed": passed,
    }


def print_comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("비교")

    best = max(rows, key=lambda r: r["val_top1"])

    table = [dict(b) for b in BASELINES]
    table.append({
        "config": f"h-weight sweep best  {best['weights']}",
        "top1_accuracy": best["val_top1"],
        "note": f"이번 sweep, {best['label']}",
    })

    for row in sorted(table, key=lambda r: r["top1_accuracy"]):
        print(f"  {row['config']:<38} {row['top1_accuracy'] * 100:>8.3f}%")

    print()
    print("  Fixed weighted V1 reference(29.312%)는 Validation에서 L1/L2/L3")
    print("  score weight를 탐색해 얻은 development reference입니다.")
    print("  이번 hidden-weight sweep과 동일 조건의 결과가 아닙니다.")

    return best


def write_readme(
    path: Path,
    rows: List[Dict[str, Any]],
    reproduction: Dict[str, Any],
    best: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    lines: List[str] = []
    add = lines.append

    add("# history hidden weight sweep")
    add("")
    add("history 기사 벡터를 만들 때 c1/c2/c3 hidden state를 어떤 비율로")
    add("합칠지만 비교했다.")
    add("")
    add("    a_semantic = (w1*h1 + w2*h2 + w3*h3) / (w1 + w2 + w3)")
    add("")
    add("**c4는 완전히 제외했다.** encoder token으로도, identity residual로도")
    add("쓰지 않는다 (beta4 = 0). c1/c2/c3 weighting 효과만 분리해서 보기")
    add("위해서다.")
    add("")
    add("`[1, 0.5, 0.1]`은 V1의 L1/L2/L3 score weight에서 나온 값이며 hidden")
    add("vector에 대한 최적 weight라는 뜻이 아니다. 이번 sweep은 hidden state")
    add("aggregation에 맞는 비율을 따로 찾는 것이다.")
    add("")

    add("## 실행 정보")
    add("")
    add(f"- 조합 수      : {len(rows)}")
    add(f"- seed         : {metadata['seed']} (from scratch, warm-start 없음)")
    add(f"- LR           : {metadata['learning_rate']}, batch {metadata['batch_size']}")
    add(f"- max epoch    : {metadata['max_epochs']}, patience {metadata['patience']}")
    add(f"- best 선택    : Validation Top-1")
    add(f"- git commit   : {metadata.get('git_commit')}")
    add(f"- 총 소요      : {format_duration(metadata['elapsed_seconds'])}")
    add("")
    add("weight 외에는 데이터 / config / optimizer / loss / candidate 구조 /")
    add("score 구조가 전부 같다. 각 조합은 같은 seed에서 embedding, encoder,")
    add("pooling, candidate_projection, bilinear가 동일한 초기값으로 시작한다.")
    add("")

    add("## 재현 확인")
    add("")

    if reproduction.get("checked"):
        verdict = "OK" if reproduction["passed"] else "FAIL"
        add(f"| 항목 | 값 |")
        add(f"|---|---|")
        add(f"| 기존 c123 mean | {reproduction['target'] * 100:.4f}% |")
        add(f"| 이번 A `[1,1,1]` | {reproduction['actual'] * 100:.4f}% |")
        add(f"| 차이 | {reproduction['deviation'] * 100:.4f}%p |")
        add(f"| 판정 | {verdict} |")
        add("")

        if not reproduction["passed"]:
            add("**재현되지 않았다.** 아래 결과를 해석하면 안 된다.")
            add("")

    add("## sweep 결과")
    add("")
    add("| | weights | best epoch | epochs | val Top-1 | MRR | nDCG@5 | AUC "
        "| val loss | 과적합폭 | A 대비 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")

    baseline = next((r for r in rows if r["label"] == REPRODUCE_LABEL), None)

    for row in sorted(rows, key=lambda r: -r["val_top1"]):
        if baseline and row["label"] != REPRODUCE_LABEL:
            delta = f"{(row['val_top1'] - baseline['val_top1']) * 100:+.3f}%p"
        else:
            delta = "기준"

        add(
            f"| {row['label']} | `{row['weights']}` | {row['best_epoch']} | "
            f"{row['epochs_run']} | {row['val_top1'] * 100:.3f}% | "
            f"{row['mrr']:.4f} | {row['ndcg5']:.4f} | {row['auc']:.4f} | "
            f"{row['val_loss']:.4f} | {row['overfit_gap'] * 100:.2f}%p | "
            f"{delta} |"
        )

    add("")
    add("과적합폭 = best epoch의 train Top-1 − val Top-1")
    add("")

    add("## 확률 / score 분리도")
    add("")
    add("| | weights | positive prob | negative prob | score gap |")
    add("|---|---|---|---|---|")

    for row in sorted(rows, key=lambda r: -r["val_top1"]):
        add(
            f"| {row['label']} | `{row['weights']}` | "
            f"{row['positive_prob']:.4f} | {row['negative_prob']:.4f} | "
            f"{row['score_gap']:.4f} |"
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
        "config": f"**h-weight sweep best** `{best['weights']}`",
        "top1_accuracy": best["val_top1"],
        "note": f"이번 sweep, {best['label']}",
    })

    for row in sorted(table, key=lambda r: r["top1_accuracy"]):
        add(f"| {row['config']} | {row['top1_accuracy'] * 100:.3f}% | "
            f"{row['note']} |")

    add("")
    add("`Fixed weighted V1 reference`(29.312%)는 Validation에서 L1/L2/L3")
    add("score weight를 탐색해 얻은 development reference다. 이번")
    add("hidden-weight sweep과 동일 조건의 결과가 아니다.")
    add("")

    add("## 다음 단계")
    add("")
    add(f"best weight `{best['weights']}`를 고정하고 c4 beta sweep")
    add("(beta4 = 0, 0.01, 0.05, 0.1)을 한 뒤, representation을 고정하고")
    add("dropout x weight decay sweep으로 간다.")
    add("")

    add("## 하지 않은 것")
    add("")
    add("- c4 (encoder token, identity residual 둘 다)")
    add("- dropout / weight decay / LR / optimizer 변경")
    add("- candidate 구조, score 구조 변경")
    add("- MLP scorer, hybrid, learnable weight")
    add("- Test 데이터 사용")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="history hidden state aggregation weight sweep"
    )
    parser.add_argument(
        "--out",
        default="sweep_out/ebnerd_v2/priority2_direct/hidden_weight_sweep",
    )
    parser.add_argument(
        "--priority0-dir",
        default="sweep_out/ebnerd_v2/priority0_shuffled/seed42",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="LABEL",
        help="일부 조합만 돌린다. 예: --only A F",
    )
    parser.add_argument("--config", default=None,
                        help="생략하면 run_priority2_direct의 기본 gin config")
    parser.add_argument("--train-path", default=None)
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

    combos = COMBOS

    if args.only:
        wanted = {label.upper() for label in args.only}
        combos = [c for c in COMBOS if c[0] in wanted]

        if not combos:
            print(f"해당하는 조합이 없습니다: {args.only}")
            return 1

    section("history hidden weight sweep")
    print(f"  출력   : {out_root}")
    print(f"  조합   : {len(combos)}개")
    print(f"  c4     : 완전히 제외 (encoder token X, residual X, beta4 = 0)")
    print(f"  고정   : seed {args.seed}, LR {args.learning_rate}, "
          f"batch {args.batch_size}, "
          f"max_epochs {args.max_epochs}, patience {args.patience}")
    print()
    print(f"  {'':<3} {'weights':<18} 메모")
    print("  " + "-" * 60)

    for label, weights, note in combos:
        text = f"[{weights[0]:g}, {weights[1]:g}, {weights[2]:g}]"
        print(f"  {label:<3} {text:<18} {note}")

    rows: List[Dict[str, Any]] = []

    for index, (label, weights, note) in enumerate(combos, start=1):
        out_dir = combo_dir(out_root, label, weights)
        log_path = out_root / f"{label}.log"

        section(
            f"[{index}/{len(combos)}]  {label}  "
            f"[{weights[0]:g}, {weights[1]:g}, {weights[2]:g}]"
        )

        done = (out_dir / "best_metrics.json").exists()

        if done:
            print(f"  이미 끝난 조합입니다. 건너뜁니다: {out_dir}")
            rows.append(collect(out_dir, label, weights, note, None))
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
        code = run_one(out_dir, weights, args, log_path)
        elapsed = time.time() - combo_started

        if code != 0:
            print(f"  실패 (exit {code}). 로그를 확인하세요: {log_path}")
            return 1

        rows.append(collect(out_dir, label, weights, note, elapsed))

        print(f"  완료 ({format_duration(elapsed)})  "
              f"val Top-1 {rows[-1]['val_top1'] * 100:.3f}%  "
              f"best epoch {rows[-1]['best_epoch']}")

    reproduction = check_reproduction(rows)
    print_results(rows)
    best = print_comparison(rows)

    # 저장
    section("저장")

    csv_rows = [
        {k: v for k, v in row.items() if k != "out_dir"}
        for row in sorted(rows, key=lambda r: -r["val_top1"])
    ]

    write_csv(out_root / "sweep_results.csv", csv_rows)

    metadata = {
        "experiment": "history_hidden_weight_sweep",
        "formula": "a_semantic = (w1*h1 + w2*h2 + w3*h3) / (w1+w2+w3)",
        "beta4": 0.0,
        "c4_used": False,
        "combos": [
            {"label": l, "weights": list(w), "note": n} for l, w, n in combos
        ],
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "selection_criterion": "validation top1_accuracy",
        "reproduction_check": reproduction,
        "best": {
            "label": best["label"],
            "weights": [best["w1"], best["w2"], best["w3"]],
            "val_top1": best["val_top1"],
        },
        "results": rows,
        "did_not_do": [
            "c4 (encoder token / identity residual 둘 다)",
            "dropout / weight decay / LR / optimizer 변경",
            "candidate 구조 변경",
            "score 구조 변경",
            "MLP scorer, hybrid, learnable weight",
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

    write_readme(
        out_root / "README.md", rows, reproduction, best, metadata
    )

    for name in sorted(p.name for p in out_root.iterdir() if p.is_file()):
        print(f"  {out_root / name}")

    print()
    print(f"  총 소요 {format_duration(time.time() - started_at)}")

    if reproduction.get("checked") and not reproduction["passed"]:
        print()
        print("  재현 확인이 실패했습니다. 결과를 해석하기 전에 원인을 보세요.")
        return 1

    return 0


from analysis.live_output import enable_line_buffering

enable_line_buffering()

if __name__ == "__main__":
    raise SystemExit(main())
