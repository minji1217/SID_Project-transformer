"""candidate-order shuffle만으로 V1 최종 성능이 얼마나 변하는지 확인한다.

V1에서 확정된 최종 config를 그대로 쓴다. grid search를 다시 돌리지 않는다.
shuffle된 데이터로 처음부터 재학습하고, 기존 seed별 결과와 표로 비교한다.

    max_history_length=50, use_sep=False
    d_model=256, num_heads=8, num_layers=2, d_ff=1024
    learning_rate=5e-05, batch_size=128, dropout=0.0, weight_decay=0.0
    30 epoch / patience 5

기본은 seed 42 하나만 돌린다.
차이가 작으면 --seeds 42 123 2026으로 나머지를 추가한다.
이미 끝난 seed는 건너뛰므로 같은 명령을 다시 써도 된다.

    cd Transformer
    python -m analysis.rerun_final_config \
        --config configs/transformer_ebnerd.gin \
        --out sweep_out/ebnerd_v2/shuffle_check \
        --seeds 42
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

from sweep.run_stage import format_binding, write_csv


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = BASE_DIR / "train_transformer.py"

# V1에서 확정된 최종 config. 바꾸지 않는다.
FINAL_CONFIG: Dict[str, Any] = {
    "NewsSequenceDataset.max_history_length": 50,
    "NewsEncoderDecoderTransformer.use_sep": False,
    "NewsEncoderDecoderTransformer.d_model": 256,
    "NewsEncoderDecoderTransformer.num_heads": 8,
    "NewsEncoderDecoderTransformer.num_layers": 2,
    "NewsEncoderDecoderTransformer.d_ff": 1024,
    "NewsEncoderDecoderTransformer.dropout_rate": 0.0,
    "train.learning_rate": 5e-05,
    "train.batch_size": 128,
    "train.weight_decay": 0.0,
}

# seed robustness와 같은 학습 예산
NUM_EPOCHS = 30
PATIENCE = 5

# V1 seed robustness 결과 (per_seed_results.csv).
# 같은 config, 같은 예산, shuffle 전 데이터.
V1_REFERENCE: Dict[int, Dict[str, float]] = {
    42: {
        "val_top1_accuracy": 0.2701858032297285,
        "val_mrr": 0.5205259494090299,
        "val_ndcg@5": 0.6389069164316497,
        "val_auc": 0.5857514137202262,
        "val_preference_loss": 1.7730745677192212,
    },
    123: {
        "val_top1_accuracy": 0.2618462818954051,
        "val_mrr": 0.5136571327790134,
        "val_ndcg@5": 0.6336487872950608,
        "val_auc": 0.5775271524043444,
        "val_preference_loss": 1.7683666293472757,
    },
    2026: {
        "val_top1_accuracy": 0.2655346024855364,
        "val_mrr": 0.5159567571542164,
        "val_ndcg@5": 0.6353626330752576,
        "val_auc": 0.5789643326342933,
        "val_preference_loss": 1.7835721502199098,
    },
}

V1_MEAN = {
    "val_top1_accuracy": 0.26585556253688997,
    "val_mrr": 0.5167132797807532,
    "val_ndcg@5": 0.6359727789339894,
    "val_auc": 0.5807476329196213,
    "val_preference_loss": 1.775004449095469,
}

V1_STD_TOP1 = 0.004179014900348233

METRICS = [
    ("val_top1_accuracy", "Top-1", "max"),
    ("val_mrr", "MRR", "max"),
    ("val_ndcg@5", "nDCG@5", "max"),
    ("val_auc", "AUC", "max"),
    ("val_preference_loss", "Pref Loss", "min"),
]

# Top-1이 이 이상 바뀌면 재탐색을 검토한다
TOP1_THRESHOLD = 0.005


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
    print("=" * 78)
    print(title)
    print("=" * 78)


def train_one(
    base_config: Path,
    run_dir: Path,
    seed: int,
    num_workers: int,
    progress_interval: int,
) -> int:
    bindings = dict(FINAL_CONFIG)
    bindings.update({
        "train.num_epochs": NUM_EPOCHS,
        "train.early_stopping_patience": PATIENCE,
        "train.seed": seed,
        "train.save_every_epoch": False,
        "train.save_optimizer_state": False,
        "train.save_dir": str(run_dir),
        "train.num_workers": num_workers,
        "train.progress_interval": progress_interval,
    })

    command = [sys.executable, "-u", str(TRAIN_SCRIPT),
               "--config", str(base_config)]

    for key in sorted(bindings):
        command += ["--gin-binding", format_binding(key, bindings[key])]

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.txt"

    print(f"  로그: {log_path}")
    print()

    with log_path.open("w", encoding="utf-8") as log_file:
        popen = subprocess.Popen(
            command, cwd=str(BASE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )

        assert popen.stdout is not None

        with popen.stdout as stream:
            for line in stream:
                log_file.write(line)
                log_file.flush()
                sys.stdout.write("    " + line)
                sys.stdout.flush()

        return popen.wait()


def read_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = run_dir / "run_summary.json"

    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best_metrics") or {}

    row: Dict[str, Any] = {
        "best_epoch": summary.get("best_epoch"),
        "num_epochs_run": summary.get("num_epochs_run"),
        "stopped_early": summary.get("stopped_early"),
        "mean_epoch_seconds": summary.get("mean_epoch_seconds"),
        "total_parameters": summary.get("total_parameters"),
    }

    for key, _, _ in METRICS:
        row[key] = best.get(key)

    row["val_score_gap"] = best.get("val_score_gap")
    row["train_top1_accuracy"] = best.get("train_top1_accuracy")

    return row


def print_comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("shuffle 전/후 비교  (같은 config, 30 epoch / patience 5)")

    verdicts = []

    for row in rows:
        seed = row["seed"]
        reference = V1_REFERENCE.get(seed)

        print()
        print(f"  seed {seed}")
        print(f"  {'metric':<12}{'V1 (shuffle 전)':>18}{'V2 (shuffle 후)':>18}"
              f"{'차이':>12}{'판정':>10}")
        print("  " + "-" * 70)

        for key, label, direction in METRICS:
            after = row.get(key)

            if reference is None or after is None:
                print(f"  {label:<12}{'-':>18}{'-':>18}{'-':>12}{'-':>10}")
                continue

            before = reference[key]
            delta = after - before

            if key == "val_top1_accuracy":
                mark = "OK" if abs(delta) < TOP1_THRESHOLD else "주의"
                verdicts.append(abs(delta) < TOP1_THRESHOLD)
                print(f"  {label:<12}{before:>18.4f}{after:>18.4f}"
                      f"{delta:>+12.4f}{mark:>10}")
            else:
                print(f"  {label:<12}{before:>18.4f}{after:>18.4f}"
                      f"{delta:>+12.4f}{'':>10}")

        print()
        print(f"    best_epoch {row.get('best_epoch')} / "
              f"{row.get('num_epochs_run')} epoch 실행, "
              f"train Top-1 {row.get('train_top1_accuracy'):.4f}"
              if row.get("train_top1_accuracy") is not None else "")

    # 3 seed가 다 있으면 평균도 비교
    summary: Dict[str, Any] = {"per_seed_within_threshold": verdicts}

    complete = [r for r in rows if r.get("val_top1_accuracy") is not None]

    if len(complete) >= 2:
        section("평균 비교")

        print(f"  {'metric':<12}{'V1 3-seed 평균':>18}"
              f"{'V2 평균':>18}{'차이':>12}")
        print("  " + "-" * 60)

        for key, label, _ in METRICS:
            values = [r[key] for r in complete if r.get(key) is not None]

            if not values:
                continue

            after = sum(values) / len(values)
            before = V1_MEAN[key]

            print(f"  {label:<12}{before:>18.4f}{after:>18.4f}"
                  f"{after - before:>+12.4f}")
            summary[f"v2_mean_{key}"] = after

        print()
        print(f"  참고: V1의 seed 간 Top-1 표준편차는 {V1_STD_TOP1:.4f}였다.")
        print("        이보다 작은 차이는 seed 노이즈와 구분되지 않는다.")

    section("판정")

    if not verdicts:
        print("  비교할 결과가 없습니다.")
        return summary

    if all(verdicts):
        print(f"  Top-1 변화가 모두 {TOP1_THRESHOLD:.3f}(0.5%p) 미만입니다.")
        print()
        print("  >>> candidate-order shuffle이 최종 성능을 바꾸지 않았습니다.")
        print("      기존 hyperparameter를 유지하고 V2로 진행할 수 있습니다.")
        print("      seed 42만 돌렸다면 123, 2026을 추가해 확인하세요.")
        summary["verdict"] = "within_threshold"
    else:
        print(f"  Top-1 변화가 {TOP1_THRESHOLD:.3f}(0.5%p) 이상인 seed가 있습니다.")
        print()
        print("  >>> 다른 ranking metric도 같은 방향으로 움직였는지 확인하세요.")
        print("      같은 방향이면 제한적 재탐색 또는 전체 sweep 재실행을 검토합니다.")
        print("      한 seed만으로는 단정할 수 없으니 나머지 seed를 먼저 돌리세요.")
        summary["verdict"] = "exceeds_threshold"

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--report-only", action="store_true",
        help="학습하지 않고 기존 결과로 비교표만 만든다.",
    )
    args = parser.parse_args()

    started_at = time.time()

    base_config = resolve_path(args.config)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    section("V1 최종 config 재학습  (shuffle된 데이터)")
    print("  grid search를 다시 돌리지 않습니다. config는 V1 확정값 그대로입니다.")
    print()

    for key in sorted(FINAL_CONFIG):
        print(f"    {key:<48} {FINAL_CONFIG[key]}")

    print()
    print(f"    학습 예산 : {NUM_EPOCHS} epoch / patience {PATIENCE}")
    print(f"    seed      : {', '.join(str(s) for s in args.seeds)}")
    print(f"    출력      : {out_dir}")

    rows: List[Dict[str, Any]] = []

    for index, seed in enumerate(args.seeds, 1):
        run_dir = out_dir / f"seed{seed}"

        section(f"[{index}/{len(args.seeds)}] seed {seed}")

        if (run_dir / "run_summary.json").exists():
            print("  이미 끝난 seed입니다. 재학습하지 않고 결과를 읽습니다.")
        elif args.report_only:
            print("  결과가 없습니다. --report-only라 건너뜁니다.")
            continue
        else:
            code = train_one(
                base_config=base_config,
                run_dir=run_dir,
                seed=seed,
                num_workers=args.num_workers,
                progress_interval=args.progress_interval,
            )

            if code != 0:
                print(f"  학습 실패 (exit {code}). 로그: {run_dir / 'train_log.txt'}")
                continue

        metrics = read_metrics(run_dir)

        if metrics is None:
            print("  run_summary.json이 없습니다.")
            continue

        metrics["seed"] = seed
        metrics["run_dir"] = str(run_dir)
        rows.append(metrics)

    if not rows:
        print()
        print("결과가 없습니다.")
        return 1

    summary = print_comparison(rows)

    # 저장
    csv_rows = []

    for row in rows:
        seed = row["seed"]
        reference = V1_REFERENCE.get(seed, {})

        entry: Dict[str, Any] = {"seed": seed}

        for key, label, _ in METRICS:
            entry[f"v1_{key}"] = reference.get(key)
            entry[f"v2_{key}"] = row.get(key)

            if reference.get(key) is not None and row.get(key) is not None:
                entry[f"delta_{key}"] = row[key] - reference[key]

        entry["best_epoch"] = row.get("best_epoch")
        entry["num_epochs_run"] = row.get("num_epochs_run")
        entry["train_top1_accuracy"] = row.get("train_top1_accuracy")
        entry["run_dir"] = row.get("run_dir")

        csv_rows.append(entry)

    write_csv(out_dir / "shuffle_comparison.csv", csv_rows)

    (out_dir / "shuffle_comparison.json").write_text(
        json.dumps(
            {
                "experiment": "candidate_order_shuffle_rerun",
                "note": (
                    "V1 최종 config를 그대로 쓰고 shuffle된 데이터로 재학습했다. "
                    "grid search를 다시 돌리지 않았다."
                ),
                "final_config": FINAL_CONFIG,
                "num_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "top1_threshold": TOP1_THRESHOLD,
                "v1_reference": V1_REFERENCE,
                "v1_mean": V1_MEAN,
                "v1_std_top1": V1_STD_TOP1,
                "results": rows,
                "summary": summary,
                "git_commit": git_commit_hash(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - started_at, 2),
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(f"  CSV  : {out_dir / 'shuffle_comparison.csv'}")
    print(f"  JSON : {out_dir / 'shuffle_comparison.json'}")

    return 0


from analysis.live_output import enable_line_buffering

enable_line_buffering()

if __name__ == "__main__":
    raise SystemExit(main())
