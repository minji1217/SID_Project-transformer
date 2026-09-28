"""학습에 실제로 쓰인 parquet이 shuffled 버전인지 확인한다.

왜 필요한가
    train_transformer.py의 resolve_path()는 .resolve()를 부르지 않아
    심볼릭 링크를 따라가지 않는다. 그래서 run_summary.json에 남는 경로는

        .../Transformer/datasets/ebnerd/train_sequences_1pos4neg.parquet

    로, 그 자리의 파일이 V1이든 shuffled든 문자열이 같다.
    경로만으로는 어느 데이터로 돌았는지 알 수 없다.

    파일 내용으로 확인해야 한다. candidate_labels에서 positive가
    몇 번째에 있는지 세면 바로 갈린다.
        V1       : index 0이 100%
        shuffled : 각 index가 20% 근처

같이 확인하는 것
    심볼릭 링크를 푼 실제 경로, 수정 시각, 파일 크기, 행 수.
    ~/shared/datasets/ebnerd_v1_backup 이 있으면 대조군으로 함께 찍는다.

    cd Transformer
    python -m analysis.verify_shuffled_data \
        --run-summary sweep_out/ebnerd_v2/shuffle_check/seed42/run_summary.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl


BASE_DIR = Path(__file__).resolve().parent.parent

NUM_CANDIDATES = 5
UNIFORM = 1.0 / NUM_CANDIDATES

# 각 index가 20%에서 이만큼 안이면 shuffled로 본다
SHUFFLE_TOLERANCE = 0.02

DEFAULT_BACKUP_DIR = "~/shared/datasets/ebnerd_v1_backup"

BACKUP_FILES = [
    ("V1 backup train", "train_sequences_1pos4neg.parquet"),
    ("V1 backup validation half", "validation_sequences_1pos4neg_half.parquet"),
    ("V1 backup test", "test_sequences_1pos4neg.parquet"),
]


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def inspect(label: str, path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"  [{label}] 파일이 없습니다: {path}")
        return None

    real = os.path.realpath(path)
    stat = path.stat()

    df = pl.read_parquet(path)

    if "candidate_labels" not in df.columns:
        print(f"  [{label}] candidate_labels 컬럼이 없습니다.")
        return None

    label_lists = df["candidate_labels"].to_list()
    counter: collections.Counter = collections.Counter()

    for labels in label_lists:
        positions = [i for i, v in enumerate(labels) if v == 1]

        if len(positions) != 1:
            raise ValueError(
                f"{path}: positive 수가 1이 아닌 row가 있습니다 ({len(positions)}개)"
            )

        counter[positions[0]] += 1

    total = df.height
    ratios = [counter[i] / total for i in range(NUM_CANDIDATES)]
    deviation = max(abs(r - UNIFORM) for r in ratios)

    print()
    print(f"  {label}")
    print(f"    지정 경로 : {path}")

    if str(path.resolve()) != real or str(path) != real:
        print(f"    실제 경로 : {real}")

    print(f"    수정 시각 : "
          f"{datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M:%S}"
          f"   크기 {stat.st_size / 1024 ** 2:.1f} MB")
    print(f"    행 수     : {total:,}")
    print(f"    positive index 분포")

    for i in range(NUM_CANDIDATES):
        bar = "#" * int(round(ratios[i] * 50))
        print(f"      index {i}  {counter[i]:>9,}  {ratios[i]:>7.2%}  {bar}")

    print(f"    균등(20.00%) 대비 최대 편차 : {deviation:.2%}")

    is_shuffled = deviation <= SHUFFLE_TOLERANCE
    is_v1 = ratios[0] > 0.99

    if is_shuffled:
        print(f"    => shuffled 버전입니다.")
    elif is_v1:
        print(f"    => V1 버전입니다 (positive가 index 0에 고정).")
    else:
        print(f"    => 판단 불가. 분포가 균등하지도 0번 고정도 아닙니다.")

    return {
        "label": label,
        "path": str(path),
        "real_path": real,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "size_mb": round(stat.st_size / 1024 ** 2, 2),
        "num_rows": total,
        "counts": [counter[i] for i in range(NUM_CANDIDATES)],
        "ratios": ratios,
        "max_deviation_from_uniform": deviation,
        "is_shuffled": is_shuffled,
        "is_v1": is_v1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-summary",
        default="sweep_out/ebnerd_v2/shuffle_check/seed42/run_summary.json",
    )
    parser.add_argument("--backup-dir", default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    summary_path = Path(args.run_summary).expanduser()

    if not summary_path.is_absolute():
        summary_path = BASE_DIR / summary_path

    if not summary_path.exists():
        print(f"run_summary.json이 없습니다: {summary_path}")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    section("run_summary.json에 기록된 경로")
    print(f"  파일 : {summary_path}")
    print()

    recorded: List[Tuple[str, Path]] = []

    for key, label in (("train_path", "train"), ("validation_path", "validation")):
        value = summary.get(key)
        print(f"  {key:<18} {value}")

        if value:
            recorded.append((label, Path(value)))

    print()
    print("  주의: train_transformer.py의 resolve_path()는 .resolve()를 부르지")
    print("        않으므로 이 경로는 심볼릭 링크를 푼 값이 아니다.")
    print("        경로가 같아도 그 자리의 파일이 바뀌었을 수 있다.")
    print("        아래에서 파일 내용으로 확인한다.")

    section("학습에 실제로 쓰인 파일")

    results: List[Dict[str, Any]] = []

    for label, path in recorded:
        info = inspect(label, path)

        if info:
            results.append(info)

    backup_dir = Path(args.backup_dir).expanduser()

    if backup_dir.exists():
        section("대조군 — V1 백업")

        for label, filename in BACKUP_FILES:
            info = inspect(label, backup_dir / filename)

            if info:
                results.append(info)
    else:
        print()
        print(f"백업 폴더가 없어 대조군은 생략합니다: {backup_dir}")

    section("결론")

    used = [r for r in results if not r["label"].startswith("V1 backup")]

    if not used:
        print("  확인된 파일이 없습니다.")
        return 1

    all_shuffled = all(r["is_shuffled"] for r in used)
    any_v1 = any(r["is_v1"] for r in used)

    print()
    for r in used:
        verdict = (
            "shuffled" if r["is_shuffled"]
            else ("V1" if r["is_v1"] else "판단 불가")
        )
        print(f"  {r['label']:<12} {r['num_rows']:>9,} rows   "
              f"편차 {r['max_deviation_from_uniform']:>6.2%}   {verdict}")

    print()

    if all_shuffled:
        print("  >>> train과 validation 모두 shuffled 버전입니다.")
        print("      이번 seed42 결과를 shuffle 검증 결과로 확정할 수 있습니다.")
        code = 0
    elif any_v1:
        print("  >>> V1 버전으로 학습됐습니다.")
        print("      파일을 교체한 뒤 다시 학습해야 합니다.")
        code = 1
    else:
        print("  >>> 판단할 수 없습니다. 위 분포를 확인하세요.")
        code = 1

    if args.out:
        out_path = Path(args.out).expanduser()

        if not out_path.is_absolute():
            out_path = BASE_DIR / out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "run_summary": str(summary_path),
                    "recorded_paths": {
                        "train_path": summary.get("train_path"),
                        "validation_path": summary.get("validation_path"),
                    },
                    "note": (
                        "resolve_path()가 심볼릭 링크를 풀지 않아 "
                        "기록된 경로는 데이터 버전을 구분하지 못한다. "
                        "파일 내용으로 확인했다."
                    ),
                    "shuffle_tolerance": SHUFFLE_TOLERANCE,
                    "files": results,
                    "all_shuffled": all_shuffled,
                },
                ensure_ascii=False, indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print()
        print(f"  저장 : {out_path}")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
