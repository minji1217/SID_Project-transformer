"""
확정된 최종 설정에 대해서만 Test를 평가한다.

Test는 파라미터 선택에 쓰지 않는다.
selected_final.json이 있어야만 실행되며, 그 파일은
Validation seed 평균으로 최종 설정을 확정했다는 표시다.

    cd Transformer
    python -m sweep.run_final_test \\
      --config configs/transformer_ebnerd.gin \\
      --sweep-out sweep_out/ebnerd \\
      --test-path datasets/ebnerd/test_sequences_1pos4neg.parquet

seed별 결과와 평균±표준편차를 모두 저장한다.
이 결과를 보고 파라미터를 바꾸면 안 된다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluate.metrics import evaluate as evaluate_predictions
from sweep.run_stage import describe, format_binding, write_csv


BASE_DIR = Path(__file__).resolve().parent.parent
PREDICT_SCRIPT = BASE_DIR / "predict_sid.py"

TEST_METRICS = ("top1_accuracy", "auc", "mrr", "ndcg@5", "ndcg@10")


def resolve_path(value: str) -> Path:
    # "~/shared/datasets/..." 표기를 지원한다
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def mean_and_std(values: List[float]) -> Dict[str, Optional[float]]:
    usable = [value for value in values if value is not None and value == value]

    if not usable:
        return {"mean": None, "std": None}

    if len(usable) == 1:
        return {"mean": usable[0], "std": 0.0}

    return {
        "mean": statistics.fmean(usable),
        "std": statistics.stdev(usable),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="확정된 최종 설정의 Test 평가",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sweep-out", type=str, required=True)
    parser.add_argument(
        "--test-path",
        type=str,
        default="datasets/ebnerd/test_sequences_1pos4neg.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sweep_dir = Path(args.sweep_out)
    if not sweep_dir.is_absolute():
        sweep_dir = BASE_DIR / sweep_dir

    final_path = sweep_dir / "seed_robustness" / "selected_final.json"

    if not final_path.exists():
        print(
            "최종 설정이 확정되지 않았습니다.\n"
            f"필요한 파일: {final_path}\n\n"
            "Test는 Validation으로 최종 설정을 확정한 뒤에만 평가합니다.\n"
            "먼저 sweep.run_seed_robustness를 --accept-auto로 실행하거나 "
            "selected_final.json을 직접 작성하세요."
        )
        return 1

    final = json.loads(final_path.read_text(encoding="utf-8"))
    params: Dict[str, Any] = final.get("config") or {}
    checkpoints: List[str] = final.get("checkpoints") or []

    if not checkpoints:
        print(f"{final_path}에 checkpoints가 없습니다.")
        return 1

    base_config = resolve_path(args.config)
    test_path = resolve_path(args.test_path)

    if not base_config.exists():
        print(f"gin config를 찾을 수 없습니다: {base_config}")
        return 1

    if not test_path.exists():
        print(f"Test 파일을 찾을 수 없습니다: {test_path}")
        return 1

    output_dir = sweep_dir / "final_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Final Test Evaluation")
    print("=" * 78)
    print(f"최종 설정 : {describe(params) or '(base config)'}")
    print(f"Test      : {test_path}")
    print(f"Checkpoint: {len(checkpoints)}개")
    print("=" * 78)
    print("이 결과를 보고 파라미터를 바꾸지 않는다.")

    rows: List[Dict[str, Any]] = []

    for index, checkpoint in enumerate(checkpoints, 1):
        checkpoint_path = Path(checkpoint)

        print()
        print("-" * 78)
        print(f"[{index}/{len(checkpoints)}] {checkpoint_path}")

        if not checkpoint_path.exists():
            print("checkpoint를 찾을 수 없습니다. 건너뜁니다.")
            rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "status": "MISSING",
                }
            )
            continue

        # seed는 run 폴더의 run_summary.json에서 읽는다
        seed: Optional[int] = None
        summary_path = checkpoint_path.parent / "run_summary.json"

        if summary_path.exists():
            seed = json.loads(
                summary_path.read_text(encoding="utf-8")
            ).get("seed")

        prediction_path = output_dir / f"test_scores_seed_{seed}.parquet"
        log_path = output_dir / f"predict_seed_{seed}.log"

        command = [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--config",
            str(base_config),
            "--checkpoint",
            str(checkpoint_path),
            "--test_path",
            str(test_path),
            "--output_path",
            str(prediction_path),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ]

        # 학습 때와 같은 모델 구조로 만들어야 checkpoint를 불러올 수 있다
        for key in sorted(params):
            command += ["--gin-binding", format_binding(key, params[key])]

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        if process.returncode != 0:
            print(f"예측 실패 (exit {process.returncode}). 로그: {log_path}")
            rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "seed": seed,
                    "status": "FAILED",
                    "log_path": str(log_path),
                }
            )
            continue

        metrics = evaluate_predictions(prediction_path)

        row: Dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "seed": seed,
            "status": "SUCCESS",
            "num_impressions": metrics.get("num_impressions"),
        }

        for key in TEST_METRICS:
            row[f"test_{key}"] = metrics.get(key)

        rows.append(row)

        print(
            "Top-1={top1_accuracy:.6f} AUC={auc:.6f} MRR={mrr:.6f} "
            "nDCG@5={ndcg5:.6f} nDCG@10={ndcg10:.6f}".format(
                top1_accuracy=metrics["top1_accuracy"],
                auc=metrics["auc"],
                mrr=metrics["mrr"],
                ndcg5=metrics["ndcg@5"],
                ndcg10=metrics["ndcg@10"],
            )
        )

    per_seed_path = output_dir / "per_seed_test_results.csv"
    write_csv(per_seed_path, rows)

    successful = [row for row in rows if row.get("status") == "SUCCESS"]

    aggregate: Dict[str, Any] = {
        "config": params,
        "config_description": describe(params) or "(base config)",
        "successful_seed_count": len(successful),
        "seeds": [row.get("seed") for row in successful],
    }

    for key in TEST_METRICS:
        stats = mean_and_std([row.get(f"test_{key}") for row in successful])
        aggregate[f"mean_test_{key}"] = stats["mean"]
        aggregate[f"std_test_{key}"] = stats["std"]

    summary_path = output_dir / "test_summary.json"
    summary_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("Test 결과 (평균 ± 표준편차)")
    print("=" * 78)

    for key in TEST_METRICS:
        mean = aggregate[f"mean_test_{key}"]
        std = aggregate[f"std_test_{key}"]

        if mean is None:
            print(f"  {key:<14} 측정 실패")
            continue

        print(f"  {key:<14} {mean:.6f} ± {std:.6f}")

    print()
    print(f"seed별 CSV : {per_seed_path}")
    print(f"요약 JSON  : {summary_path}")

    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
