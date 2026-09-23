"""
Final Top-3 설정을 여러 seed로 전체 학습해 안정성을 비교한다.

탐색 단계는 12 epoch / patience 3의 짧은 예산으로 돌기 때문에
그 결과를 최종 성능으로 쓸 수 없다.
Final Top-3 x seed 3개를 모두 30 epoch / patience 5로 새로 학습한다.
seed 42도 예외 없이 다시 학습한다.

    cd Transformer
    python -m sweep.run_seed_robustness \\
      --config configs/transformer_ebnerd.gin \\
      --sweep-out sweep_out/ebnerd \\
      --seeds 42 123 2026 \\
      --num-epochs 30 --patience 5

최종 설정은 Validation seed 평균으로 고른다. Test는 쓰지 않는다.
seed는 파라미터가 아니므로 성능이 좋은 seed를 고르지 않는다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sweep.hashing import (
    compute_config_hash,
    file_sha256,
    load_base_bindings,
    source_fingerprint,
)
from sweep.run_stage import (
    COMPLETE_MARKER,
    FAILED_MARKER,
    backup_run_dir,
    describe,
    run_training,
    short_name,
    write_csv,
)
from sweep.selection import rank_rows
from sweep.stages import METRIC_PRIORITY, STAGES, MetricRule, stage_dir_name


BASE_DIR = Path(__file__).resolve().parent.parent

# 평균을 낼 지표와 CSV 컬럼 이름
AGGREGATED_METRICS = [
    ("val_top1_accuracy", "val_top1"),
    ("val_mrr", "val_mrr"),
    ("val_ndcg@5", "val_ndcg5"),
    ("val_auc", "val_auc"),
    ("val_preference_loss", "val_preference_loss"),
    ("val_positive_prob", "val_positive_prob"),
    ("val_score_gap", "val_score_gap"),
]

# 최종 설정 선택 규칙.
# 지표 평균을 METRIC_PRIORITY 순서로 보고,
# 마지막까지 동률이면 Top-1 표준편차가 작은 설정을 고른다.
FINAL_RULES: List[MetricRule] = list(METRIC_PRIORITY) + [
    MetricRule("std_val_top1", "min", 0.0, "Top-1 표준편차")
]


def load_final_configs(sweep_dir: Path) -> List[Dict[str, Any]]:
    final_stage_dir = sweep_dir / stage_dir_name(len(STAGES))
    selected_path = final_stage_dir / "selected.json"

    if not selected_path.exists():
        raise FileNotFoundError(
            f"마지막 단계의 확정 파일이 없습니다: {selected_path}\n"
            f"{len(STAGES)}단계를 --accept-auto로 실행하거나 "
            "selected.json을 직접 작성하세요."
        )

    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    configs = payload.get("configs")

    if not configs:
        raise ValueError(f"{selected_path}에 configs가 비어 있습니다.")

    return configs


def mean_and_std(values: List[float]) -> Dict[str, Optional[float]]:
    usable = [value for value in values if value is not None and value == value]

    if not usable:
        return {"mean": None, "std": None}

    if len(usable) == 1:
        return {"mean": usable[0], "std": 0.0}

    # 표본 표준편차(n-1). 논문에 평균±표준편차로 보고할 때 쓰는 값이다.
    return {
        "mean": statistics.fmean(usable),
        "std": statistics.stdev(usable),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Top-3 설정의 seed 안정성 검증",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sweep-out", type=str, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2026],
    )
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--keep-optimizer-state",
        action="store_true",
    )
    parser.add_argument("--accept-auto", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sweep_dir = Path(args.sweep_out)
    if not sweep_dir.is_absolute():
        sweep_dir = BASE_DIR / sweep_dir

    base_config = Path(args.config)
    if not base_config.is_absolute():
        base_config = BASE_DIR / base_config

    if not base_config.exists():
        print(f"gin config를 찾을 수 없습니다: {base_config}")
        return 1

    final_configs = load_final_configs(sweep_dir)

    output_dir = sweep_dir / "seed_robustness"
    runs_dir = output_dir / "runs"

    base_bindings = load_base_bindings(base_config)
    base_sha256 = file_sha256(base_config)
    fingerprint = source_fingerprint(BASE_DIR)

    print("=" * 78)
    print("Seed Robustness")
    print("=" * 78)
    print(f"Base config : {base_config}")
    print(f"설정        : {len(final_configs)}개 (Final Top-{len(final_configs)})")
    print(f"Seed        : {args.seeds}")
    print(f"학습 예산   : {args.num_epochs} epoch, patience {args.patience}")
    print(f"총 run      : {len(final_configs) * len(args.seeds)}개")
    print(f"출력        : {output_dir}")
    print("=" * 78)
    print(
        "탐색 단계의 짧은 예산 결과는 재사용하지 않는다. "
        "seed 42도 전체 예산으로 다시 학습한다."
    )

    planned: List[Dict[str, Any]] = []

    for config_index, params in enumerate(final_configs, 1):
        for seed in args.seeds:
            full_bindings = dict(params)
            full_bindings.update(
                {
                    "train.num_epochs": args.num_epochs,
                    "train.early_stopping_patience": args.patience,
                    "train.seed": seed,
                    "train.save_every_epoch": False,
                    "train.save_optimizer_state": bool(
                        args.keep_optimizer_state
                    ),
                }
            )

            planned.append(
                {
                    "config_index": config_index,
                    "params": params,
                    "seed": seed,
                    "full": full_bindings,
                    "hash": compute_config_hash(
                        base_config_path=base_config,
                        bindings=full_bindings,
                        base_bindings=base_bindings,
                    ),
                }
            )

    if args.dry_run:
        print()
        for index, item in enumerate(planned, 1):
            run_dir = runs_dir / item["hash"]

            if (run_dir / COMPLETE_MARKER).exists():
                status = "재사용"
            elif (run_dir / FAILED_MARKER).exists():
                status = "재시도" if args.retry_failed else "실패기록"
            else:
                status = "실행"

            print(
                f"{index:>3}. [{status}] config#{item['config_index']} "
                f"seed={item['seed']} {describe(item['params'])}"
            )

        print()
        print(f"총 {len(planned)}개")
        return 0

    per_seed_rows: List[Dict[str, Any]] = []

    for index, item in enumerate(planned, 1):
        run_dir = runs_dir / item["hash"]

        complete_marker = run_dir / COMPLETE_MARKER
        failed_marker = run_dir / FAILED_MARKER

        print()
        print("-" * 78)
        print(
            f"[{index}/{len(planned)}] config#{item['config_index']} "
            f"seed={item['seed']} {describe(item['params'])}"
        )

        if args.retry_failed and run_dir.exists() and not complete_marker.exists():
            backup_path = backup_run_dir(run_dir)
            print(f"이전 실패 결과를 보관했습니다: {backup_path.name}")

        row: Dict[str, Any] = {
            "config_index": item["config_index"],
            "config": describe(item["params"]) or "(base config)",
            "seed": item["seed"],
            "config_hash": item["hash"],
        }

        for key, value in item["params"].items():
            row[short_name(key)] = value

        if complete_marker.exists():
            print("이미 완료된 run이라 재사용합니다.")

        elif failed_marker.exists():
            record = json.loads(failed_marker.read_text(encoding="utf-8"))
            print(f"이전에 실패한 run입니다 ({record.get('error_type')}).")
            row.update(
                {
                    "status": "FAILED",
                    "error_type": record.get("error_type"),
                    "message": record.get("message"),
                }
            )
            per_seed_rows.append(row)
            continue

        elif run_dir.exists():
            record = {
                "status": "FAILED",
                "error_type": "UNKNOWN",
                "return_code": None,
                "message": "완료 표시가 없는 폴더가 이미 존재합니다.",
                "log_path": str(run_dir / "train_log.txt"),
            }
            failed_marker.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(record["message"])

            if args.fail_fast:
                return 1

            row.update(
                {
                    "status": "FAILED",
                    "error_type": "UNKNOWN",
                    "message": record["message"],
                }
            )
            per_seed_rows.append(row)
            continue

        else:
            record = run_training(
                run_dir=run_dir,
                base_config=base_config,
                bindings=item["full"],
                summary_extra={
                    "config_hash": item["hash"],
                    "stage": "seed_robustness",
                    "config_index": item["config_index"],
                    "source_fingerprint": fingerprint,
                },
            )

            if record is not None:
                failed_marker.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"실패 [{record['error_type']}]: {record['message']}")

                if args.fail_fast:
                    return 1

                row.update(
                    {
                        "status": "FAILED",
                        "error_type": record["error_type"],
                        "message": record["message"],
                    }
                )
                per_seed_rows.append(row)
                continue

            complete_marker.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "stage": "seed_robustness",
                        "config_hash": item["hash"],
                        "base_config_sha256": base_sha256,
                        "source_fingerprint": fingerprint,
                        "bindings": item["full"],
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            print("완료")

        summary = json.loads(
            (run_dir / "run_summary.json").read_text(encoding="utf-8")
        )

        row["status"] = "SUCCESS"
        row["best_epoch"] = summary.get("best_epoch")
        row["num_epochs_run"] = summary.get("num_epochs_run")
        row["stopped_early"] = summary.get("stopped_early")
        row["total_seconds"] = summary.get("total_seconds")
        row["checkpoint_best"] = str(run_dir / "checkpoint_best.pt")

        for key, value in (summary.get("best_metrics") or {}).items():
            if key in ("epoch", "epoch_seconds"):
                continue
            row[key] = value

        per_seed_rows.append(row)

    per_seed_path = output_dir / "per_seed_results.csv"
    write_csv(per_seed_path, per_seed_rows)

    # 설정별 집계
    aggregate_rows: List[Dict[str, Any]] = []

    for config_index, params in enumerate(final_configs, 1):
        config_rows = [
            row
            for row in per_seed_rows
            if row["config_index"] == config_index
            and row.get("status") == "SUCCESS"
        ]

        aggregate: Dict[str, Any] = {
            "config_index": config_index,
            "config": describe(params) or "(base config)",
            "successful_seed_count": len(config_rows),
        }

        for key, value in params.items():
            aggregate[short_name(key)] = value

        for metric_key, column in AGGREGATED_METRICS:
            stats = mean_and_std(
                [row.get(metric_key) for row in config_rows]
            )

            aggregate[f"mean_{column}"] = stats["mean"]
            aggregate[f"std_{column}"] = stats["std"]

            # 선택 규칙이 METRIC_PRIORITY의 키 이름을 쓰므로
            # 평균값을 같은 이름으로도 넣어 둔다.
            aggregate[metric_key] = stats["mean"]

        aggregate["_params"] = params
        aggregate_rows.append(aggregate)

    aggregate_path = output_dir / "config_aggregate.csv"

    usable = [
        row for row in aggregate_rows if row["successful_seed_count"] > 0
    ]

    if not usable:
        write_csv(aggregate_path, aggregate_rows)
        print()
        print("성공한 run이 없어 최종 설정을 고를 수 없습니다.")
        print(f"per-seed CSV : {per_seed_path}")
        print(f"집계 CSV     : {aggregate_path}")
        return 1

    ordered = rank_rows(usable, rules=FINAL_RULES)

    for position, (row, trace) in enumerate(ordered, 1):
        row["rank"] = position
        row["decision"] = " | ".join(trace)

    write_csv(aggregate_path, [row for row, _ in ordered])

    winner = ordered[0][0]

    final_payload = {
        "seeds": args.seeds,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "selection": (
            "Validation seed 평균을 METRIC_PRIORITY 순서로 비교하고, "
            "마지막 동률이면 Top-1 표준편차가 작은 설정을 고른다."
        ),
        "config": winner["_params"],
        "config_description": winner["config"],
        "successful_seed_count": winner["successful_seed_count"],
        "mean_metrics": {
            f"mean_{column}": winner.get(f"mean_{column}")
            for _, column in AGGREGATED_METRICS
        },
        "std_metrics": {
            f"std_{column}": winner.get(f"std_{column}")
            for _, column in AGGREGATED_METRICS
        },
        "checkpoints": [
            row["checkpoint_best"]
            for row in per_seed_rows
            if row["config_index"] == winner["config_index"]
            and row.get("status") == "SUCCESS"
        ],
        "note": (
            "이 파일은 자동 선택 결과입니다. "
            "확정하려면 selected_final.json으로 복사하세요. "
            "Test 평가는 확정 이후에만 수행합니다."
        ),
    }

    auto_path = output_dir / "selected_final_auto.json"
    auto_path.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("Seed Robustness 완료")
    print("=" * 78)

    for position, (row, _) in enumerate(ordered, 1):
        marker = "->" if position == 1 else "  "
        print(f"{marker} {position}. {row['config']}")
        print(
            f"      seed {row['successful_seed_count']}개 | "
            f"Top-1 {row.get('mean_val_top1')} ± {row.get('std_val_top1')}"
        )
        print(
            f"      MRR {row.get('mean_val_mrr')} ± {row.get('std_val_mrr')} | "
            f"nDCG@5 {row.get('mean_val_ndcg5')} ± {row.get('std_val_ndcg5')} | "
            f"AUC {row.get('mean_val_auc')} ± {row.get('std_val_auc')}"
        )

    print()
    print(f"per-seed CSV : {per_seed_path}")
    print(f"집계 CSV     : {aggregate_path}")
    print(f"자동 선택     : {auto_path}")

    final_path = output_dir / "selected_final.json"

    if args.accept_auto:
        final_path.write_text(
            json.dumps(final_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"확정 완료     : {final_path}")
        print()
        print("이제 Test를 평가할 수 있습니다:")
        print(
            f"  python -m sweep.run_final_test "
            f"--config {args.config} --sweep-out {args.sweep_out}"
        )
    else:
        print()
        print(
            "config_aggregate.csv를 확인한 뒤 --accept-auto로 확정하거나 "
            "selected_final.json을 직접 작성하세요."
        )
        print("Test 평가는 최종 설정을 확정한 뒤에만 수행합니다.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
