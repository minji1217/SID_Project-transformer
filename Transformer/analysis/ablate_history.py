"""history encoder가 실제로 기여하는지 확인하는 ablation.

확정된 V1 checkpoint를 그대로 쓰고, 재학습하지 않는다.
모델 코드도 건드리지 않는다. 바꾸는 것은 입력 데이터뿐이다.
Test는 쓰지 않고 validation만 쓴다.

세 가지 조건을 같은 checkpoint로 평가한다.

  normal    원본 validation
  shuffle   history를 impression 사이에서 섞는다.
            각 impression이 다른 사용자의 history를 받는다.
            history 분포는 그대로고 짝만 깨진다.
  constant  모든 history를 길이 1의 PAD 기사로 바꾼다.
            사용자 정보가 전혀 없는 상태.

읽는 법
  shuffle에서 성능이 normal과 비슷하면
    → history encoder가 사실상 일을 하지 않는다.
      모델은 "어떤 SID가 정답이 되기 쉬운가"라는 사전분포만 배운 것이다.
  shuffle에서 constant 수준으로 떨어지면
    → history가 실제로 기여하고 있다.

    cd Transformer
    python -m analysis.ablate_history \
        --config configs/transformer_ebnerd.gin \
        --sweep-out sweep_out/ebnerd \
        --validation-path datasets/ebnerd/validation_sequences_1pos4neg_half.parquet
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from evaluate.metrics import evaluate as evaluate_predictions
from sweep.run_stage import format_binding, write_csv


BASE_DIR = Path(__file__).resolve().parent.parent
PREDICT_SCRIPT = BASE_DIR / "predict_sid.py"

HISTORY_COLUMNS = ("history_c1", "history_c2", "history_c3", "history_c4")
OPTIONAL_HISTORY_COLUMNS = ("history_article_ids",)

# data/sequence.py의 PAD_SID_VALUE와 같아야 한다
PAD_SID_VALUE = 0

CONDITIONS = ("normal", "shuffle", "constant")

METRIC_KEYS = ("top1_accuracy", "auc", "mrr", "ndcg@5")


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        return candidate

    return (BASE_DIR / candidate).resolve()


def make_variant(
    source: Path,
    condition: str,
    out_path: Path,
    seed: int = 12345,
) -> int:
    """조건에 맞게 history만 바꾼 parquet을 만든다. 행 수를 돌려준다."""
    df = pd.read_parquet(source)

    if condition == "normal":
        shutil.copyfile(source, out_path)
        return len(df)

    if condition == "shuffle":
        # history 컬럼만 통째로 다른 행의 것으로 바꾼다.
        # 고정점(자기 자신을 받는 경우)을 줄이기 위해 한 칸 회전 후 섞는다.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))

        # 자기 자신과 짝지어진 위치를 한 칸씩 밀어 없앤다
        self_paired = np.flatnonzero(order == np.arange(len(df)))
        if len(self_paired):
            order[self_paired] = order[(self_paired + 1) % len(df)]

        columns = [c for c in HISTORY_COLUMNS + OPTIONAL_HISTORY_COLUMNS
                   if c in df.columns]

        for col in columns:
            df[col] = df[col].to_numpy()[order]

        df.to_parquet(out_path)
        return len(df)

    if condition == "constant":
        # 길이 1, 값은 전부 PAD. 모든 행이 동일하므로 정보가 0이다.
        constant = np.array([PAD_SID_VALUE], dtype=np.int64)

        for col in HISTORY_COLUMNS:
            df[col] = [constant.copy() for _ in range(len(df))]

        for col in OPTIONAL_HISTORY_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

        df.to_parquet(out_path)
        return len(df)

    raise ValueError(f"알 수 없는 조건: {condition}")


def run_predict(
    base_config: Path,
    checkpoint: Path,
    data_path: Path,
    output_path: Path,
    log_path: Path,
    bindings: Dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> int:
    command = [
        sys.executable,
        "-u",
        str(PREDICT_SCRIPT),
        "--config", str(base_config),
        "--checkpoint", str(checkpoint),
        "--test_path", str(data_path),
        "--output_path", str(output_path),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
    ]

    for key in sorted(bindings):
        command += ["--gin-binding", format_binding(key, bindings[key])]

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    return process.returncode


def mean_and_std(values: List[float]) -> Dict[str, Any]:
    usable = [v for v in values if v is not None]

    if not usable:
        return {"mean": None, "std": None}

    if len(usable) == 1:
        return {"mean": usable[0], "std": 0.0}

    return {
        "mean": statistics.fmean(usable),
        "std": statistics.stdev(usable),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep-out", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="평가할 seed. 생략하면 selected_final.json의 checkpoint 전부",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="생성한 변형 parquet을 지우지 않는다",
    )
    args = parser.parse_args()

    base_config = resolve_path(args.config)
    sweep_dir = resolve_path(args.sweep_out)
    validation_path = resolve_path(args.validation_path)

    final_path = sweep_dir / "seed_robustness" / "selected_final.json"

    if not final_path.exists():
        print(f"{final_path}가 없습니다.")
        print("seed 검증을 --accept-auto로 확정한 뒤에 실행하세요.")
        return 1

    if not validation_path.exists():
        print(f"validation 파일을 찾을 수 없습니다: {validation_path}")
        return 1

    final = json.loads(final_path.read_text(encoding="utf-8"))
    params: Dict[str, Any] = final.get("config") or {}
    checkpoints: List[str] = final.get("checkpoints") or []
    seeds: List[int] = final.get("seeds") or []

    if len(checkpoints) != len(seeds):
        print("checkpoints와 seeds 개수가 다릅니다.")
        return 1

    pairs = list(zip(seeds, checkpoints))

    if args.seeds:
        pairs = [(s, c) for s, c in pairs if s in args.seeds]

    if not pairs:
        print("평가할 checkpoint가 없습니다.")
        return 1

    output_dir = sweep_dir / "history_ablation"
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 74)
    print("history ablation")
    print("=" * 74)
    print(f"설정       : {final.get('config_description')}")
    print(f"validation : {validation_path}")
    print(f"checkpoint : {len(pairs)}개  (seed {', '.join(str(s) for s, _ in pairs)})")
    print("Test 데이터는 사용하지 않습니다.")
    print()

    # 1. 변형 데이터 생성
    variant_paths: Dict[str, Path] = {}

    for condition in CONDITIONS:
        path = data_dir / f"validation_{condition}.parquet"
        rows = make_variant(validation_path, condition, path)
        variant_paths[condition] = path
        print(f"  [{condition:<8}] {rows:,} rows  ->  {path.name}")

    print()

    # 2. 조건 x checkpoint 평가
    result_rows: List[Dict[str, Any]] = []
    total = len(CONDITIONS) * len(pairs)
    done = 0

    for condition in CONDITIONS:
        for seed, checkpoint in pairs:
            done += 1
            checkpoint_path = Path(checkpoint)

            print(f"[{done}/{total}] condition={condition}  seed={seed}")

            if not checkpoint_path.exists():
                print(f"  checkpoint 없음: {checkpoint_path}")
                result_rows.append({
                    "condition": condition,
                    "seed": seed,
                    "status": "MISSING_CHECKPOINT",
                    "checkpoint": str(checkpoint_path),
                })
                continue

            prediction_path = output_dir / f"scores_{condition}_seed_{seed}.parquet"
            log_path = output_dir / f"predict_{condition}_seed_{seed}.log"

            code = run_predict(
                base_config=base_config,
                checkpoint=checkpoint_path,
                data_path=variant_paths[condition],
                output_path=prediction_path,
                log_path=log_path,
                bindings=params,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            if code != 0:
                print(f"  실패 (exit {code}). 로그: {log_path}")
                result_rows.append({
                    "condition": condition,
                    "seed": seed,
                    "status": "FAILED",
                    "return_code": code,
                    "log_path": str(log_path),
                })
                continue

            metrics = evaluate_predictions(prediction_path)

            row: Dict[str, Any] = {
                "condition": condition,
                "seed": seed,
                "status": "SUCCESS",
                "checkpoint": str(checkpoint_path),
            }

            for key in METRIC_KEYS:
                row[key] = metrics.get(key)

            result_rows.append(row)

            print(
                f"  Top-1 {row['top1_accuracy']:.4f} | "
                f"AUC {row['auc']:.4f} | "
                f"MRR {row['mrr']:.4f} | "
                f"nDCG@5 {row['ndcg@5']:.4f}"
            )

    # 3. 저장
    per_run_path = output_dir / "ablation_per_run.csv"
    write_csv(per_run_path, result_rows)

    summary: Dict[str, Any] = {
        "config": params,
        "config_description": final.get("config_description"),
        "validation_path": str(validation_path),
        "seeds": [s for s, _ in pairs],
        "conditions": {},
    }

    print()
    print("=" * 74)
    print("요약  (validation, seed 평균 +- 표준편차)")
    print("=" * 74)
    print(f"  {'condition':<12}" + "".join(f"{k:>16}" for k in METRIC_KEYS))
    print("  " + "-" * 70)

    for condition in CONDITIONS:
        rows = [
            r for r in result_rows
            if r["condition"] == condition and r.get("status") == "SUCCESS"
        ]

        entry: Dict[str, Any] = {"successful_seed_count": len(rows)}
        cells = []

        for key in METRIC_KEYS:
            stats = mean_and_std([r.get(key) for r in rows])
            entry[f"mean_{key}"] = stats["mean"]
            entry[f"std_{key}"] = stats["std"]

            if stats["mean"] is None:
                cells.append(f"{'-':>16}")
            else:
                cells.append(f"{stats['mean']:>9.4f}+-{stats['std']:.4f}")

        summary["conditions"][condition] = entry
        print(f"  {condition:<12}" + "".join(cells))

    print()
    print("  무작위 기준 : Top-1 0.2000 | AUC 0.5000 | MRR 0.4567 | nDCG@5 0.5897")
    print()

    normal = summary["conditions"]["normal"].get("mean_top1_accuracy")
    shuffled = summary["conditions"]["shuffle"].get("mean_top1_accuracy")
    constant = summary["conditions"]["constant"].get("mean_top1_accuracy")

    if normal is not None and shuffled is not None and constant is not None:
        drop = normal - shuffled
        span = normal - constant

        print(f"  normal - shuffle  = {drop:+.4f}   <- history 짝을 깬 손실")
        print(f"  normal - constant = {span:+.4f}   <- history 전체의 기여")
        print()

        if abs(drop) < 0.005:
            print("  >>> history를 섞어도 성능이 거의 그대로입니다.")
            print("      history encoder가 사실상 기여하지 않고 있습니다.")
            print("      모델은 어떤 SID가 정답이 되기 쉬운지에 대한 사전분포만")
            print("      배운 것으로 보입니다.")
        elif span > 0 and drop / span > 0.7:
            print("  >>> history가 실제로 기여하고 있습니다.")
            print("      짝을 깨면 성능 대부분이 사라집니다.")
        else:
            print("  >>> history가 부분적으로 기여하고 있습니다.")

    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.keep_data:
        shutil.rmtree(data_dir, ignore_errors=True)

    print()
    print(f"run별 CSV : {per_run_path}")
    print(f"요약 JSON : {summary_path}")
    return 0


from analysis.live_output import enable_line_buffering

enable_line_buffering()

if __name__ == "__main__":
    raise SystemExit(main())
