"""1pos4neg parquet의 candidate 순서만 deterministic하게 섞는다.

왜 builder가 아니라 후처리인가
    1pos4neg를 만든 builder 코드가 이 repo에 없다.
    그리고 요구사항이 "candidate 재샘플링이 아니라 order permutation만"이므로,
    이미 만들어진 파일의 순서만 바꾸는 쪽이 제약을 구조적으로 보장한다.

    positive 선택, negative 4개 샘플링, negative set, history, SID 값,
    split, impression 수는 이 스크립트가 건드릴 수 없다.
    행을 추가/삭제/이동하지 않고, candidate 관련 컬럼의 원소 순서만 바꾼다.

row 단위 deterministic seed
    global_seed | split_name | impression_id | positive_article_id | row_uid
    를 이어 붙여 SHA-256으로 접는다.

    같은 impression에서 positive가 갈라져 여러 row가 될 수 있으므로
    impression_id만 쓰지 않는다. 그래도 (impression_id, positive)가
    겹치는 row가 있을 수 있어 그 그룹 안의 순번(row_uid)까지 넣는다.

    전역 RNG 순서에 의존하지 않으므로 행 순서를 바꾸거나 일부만 다시 만들어도
    같은 permutation이 나온다.

    cd Transformer
    # 먼저 10행만 미리보기
    python -m tools.shuffle_candidate_order \
        --input /home/ubuntu/shared/datasets/ebnerd/train_sequences_1pos4neg.parquet \
        --split train --preview 10

    # 확인 후 실제 생성
    python -m tools.shuffle_candidate_order \
        --input /home/ubuntu/shared/datasets/ebnerd/train_sequences_1pos4neg.parquet \
        --split train --out-dir ~/shared/datasets/ebnerd_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NUM_CANDIDATES = 5

# 순서를 함께 바꿔야 하는 컬럼. 실제 존재 여부는 파일에서 확인한다.
CANDIDATE_COLUMN_CANDIDATES = (
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_c4",
    "candidate_labels",
)

# 절대 건드리지 않는 컬럼. candidate 순서와 무관하다.
NEVER_PERMUTE_PREFIX = ("history_", "target_")

DEFAULT_SEED = 42


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def detect_candidate_columns(df: pd.DataFrame) -> List[str]:
    """길이 5의 candidate 컬럼만 고른다."""
    found: List[str] = []

    for column in CANDIDATE_COLUMN_CANDIDATES:
        if column not in df.columns:
            continue

        lengths = df[column].head(1000).map(lambda v: len(np.asarray(v)))

        if (lengths == NUM_CANDIDATES).all():
            found.append(column)
        else:
            raise ValueError(
                f"{column}의 길이가 {NUM_CANDIDATES}가 아닌 행이 있습니다. "
                f"발견된 길이: {sorted(set(lengths))}"
            )

    if "candidate_labels" not in found:
        raise ValueError("candidate_labels 컬럼이 없습니다.")

    return found


def row_permutation(
    global_seed: int,
    split_name: str,
    impression_id: Any,
    positive_article_id: Any,
    row_uid: int,
) -> np.ndarray:
    """row 단위 deterministic permutation.

    전역 RNG 상태에 의존하지 않는다.
    같은 입력이면 실행 순서와 무관하게 같은 permutation이 나온다.
    """
    key = "|".join(
        [
            str(global_seed),
            str(split_name),
            str(impression_id),
            str(positive_article_id),
            str(row_uid),
        ]
    )

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)

    return np.random.default_rng(seed).permutation(NUM_CANDIDATES)


def positive_index_of(labels: np.ndarray) -> int:
    positives = np.flatnonzero(np.asarray(labels) == 1)

    if len(positives) != 1:
        raise ValueError(
            f"row의 positive 수가 1이 아닙니다: {len(positives)}개"
        )

    return int(positives[0])


def build_row_uids(df: pd.DataFrame, positive_article: pd.Series) -> np.ndarray:
    """(impression_id, positive_article) 그룹 안의 순번.

    같은 impression의 같은 positive가 여러 row에 나와도
    서로 다른 permutation을 받도록 한다.
    """
    key = pd.DataFrame(
        {
            "impression_id": df.get("impression_id", pd.Series(range(len(df)))).astype(str),
            "positive": positive_article.astype(str),
        }
    )

    return key.groupby(["impression_id", "positive"]).cumcount().to_numpy()


def position_distribution(labels_series: pd.Series) -> Dict[str, Any]:
    positions = labels_series.map(lambda v: positive_index_of(v)).to_numpy()
    counts = np.bincount(positions, minlength=NUM_CANDIDATES)
    total = len(positions)

    return {
        "counts": counts.tolist(),
        "ratios": (counts / total).tolist(),
        "total": int(total),
        "max_deviation_from_uniform": float(
            np.abs(counts / total - 1.0 / NUM_CANDIDATES).max()
        ),
    }


def print_distribution(title: str, dist: Dict[str, Any]) -> None:
    print()
    print(f"  {title}")
    print(f"  {'index':<8}{'count':>12}{'%':>10}")
    print("  " + "-" * 30)

    for i, (c, r) in enumerate(zip(dist["counts"], dist["ratios"])):
        print(f"  {i:<8}{c:>12,}{r:>10.2%}")

    print(f"  균등(20.00%) 대비 최대 편차 : {dist['max_deviation_from_uniform']:.2%}")


def shuffle_dataframe(
    df: pd.DataFrame,
    columns: List[str],
    split_name: str,
    global_seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """candidate 컬럼에 같은 permutation을 적용한 새 DataFrame을 만든다."""
    labels = df["candidate_labels"]
    positive_positions = labels.map(positive_index_of).to_numpy()

    if "candidate_article_ids" in df.columns:
        positive_article = pd.Series(
            [
                np.asarray(v)[p]
                for v, p in zip(df["candidate_article_ids"], positive_positions)
            ],
            index=df.index,
        )
    else:
        # 기사 ID가 없으면 SID 삼중항으로 대체한다
        positive_article = pd.Series(
            [
                tuple(
                    int(np.asarray(df[c].iloc[i])[positive_positions[i]])
                    for c in ("candidate_c1", "candidate_c2", "candidate_c3")
                )
                for i in range(len(df))
            ],
            index=df.index,
        )

    row_uids = build_row_uids(df, positive_article)
    impressions = df.get("impression_id", pd.Series(range(len(df)))).to_numpy()

    permutations = np.stack(
        [
            row_permutation(
                global_seed=global_seed,
                split_name=split_name,
                impression_id=impressions[i],
                positive_article_id=positive_article.iloc[i],
                row_uid=int(row_uids[i]),
            )
            for i in range(len(df))
        ]
    )

    out = df.copy()

    for column in columns:
        out[column] = [
            np.asarray(v)[permutations[i]]
            for i, v in enumerate(df[column])
        ]

    return out, permutations


def verify(
    original: pd.DataFrame,
    shuffled: pd.DataFrame,
    columns: List[str],
) -> Dict[str, Any]:
    """섞기 전후가 같은 내용인지 검증한다."""
    section("검증")

    checks: List[Tuple[str, bool, str]] = []

    # 행 수
    same_rows = len(original) == len(shuffled)
    checks.append(("total row count 동일", same_rows,
                   f"{len(original):,} -> {len(shuffled):,}"))

    # 다른 컬럼은 그대로인지
    untouched = [c for c in original.columns if c not in columns]
    untouched_ok = True

    for column in untouched:
        a, b = original[column], shuffled[column]

        if a.dtype == object:
            equal = all(
                np.array_equal(np.asarray(x), np.asarray(y))
                for x, y in zip(a, b)
            )
        else:
            equal = a.equals(b)

        if not equal:
            untouched_ok = False
            checks.append((f"{column} 변경 없음", False, "바뀜"))

    checks.append((
        f"candidate 외 {len(untouched)}개 컬럼 변경 없음",
        untouched_ok,
        "history_*, target_* 포함",
    ))

    # row당 candidate 수 5
    counts_ok = all(
        len(np.asarray(v)) == NUM_CANDIDATES for v in shuffled["candidate_labels"]
    )
    checks.append(("row당 candidate 수 = 5", counts_ok, ""))

    # row당 positive 수 1
    positive_counts = shuffled["candidate_labels"].map(
        lambda v: int((np.asarray(v) == 1).sum())
    )
    positives_ok = bool((positive_counts == 1).all())
    checks.append(("row당 positive 수 = 1", positives_ok,
                   f"분포 {dict(Counter(positive_counts.tolist()))}"))

    # candidate set 동일 (순서 무시)
    set_ok = True
    positive_ok = True
    negative_ok = True

    id_column = (
        "candidate_article_ids"
        if "candidate_article_ids" in original.columns
        else None
    )

    for i in range(len(original)):
        a_labels = np.asarray(original["candidate_labels"].iloc[i])
        b_labels = np.asarray(shuffled["candidate_labels"].iloc[i])

        if id_column:
            a = np.asarray(original[id_column].iloc[i])
            b = np.asarray(shuffled[id_column].iloc[i])
        else:
            a = np.stack([
                np.asarray(original[c].iloc[i])
                for c in ("candidate_c1", "candidate_c2", "candidate_c3")
            ], axis=1)
            b = np.stack([
                np.asarray(shuffled[c].iloc[i])
                for c in ("candidate_c1", "candidate_c2", "candidate_c3")
            ], axis=1)

        if sorted(map(str, a.tolist())) != sorted(map(str, b.tolist())):
            set_ok = False
            break

        if str(a[a_labels == 1].tolist()) != str(b[b_labels == 1].tolist()):
            positive_ok = False
            break

        if sorted(map(str, a[a_labels == 0].tolist())) != sorted(
            map(str, b[b_labels == 0].tolist())
        ):
            negative_ok = False
            break

    checks.append(("shuffle 전/후 candidate set 동일", set_ok, "순서만 다름"))
    checks.append(("shuffle 전/후 positive article 동일", positive_ok, ""))
    checks.append(("shuffle 전/후 negative 4개 set 동일", negative_ok, ""))

    # 중복/누락
    dup_ok = True
    for i in range(min(len(shuffled), 20000)):
        if id_column:
            v = np.asarray(shuffled[id_column].iloc[i])
            if len(set(map(str, v.tolist()))) != NUM_CANDIDATES:
                dup_ok = False
                break

    checks.append(("row 안 candidate 중복 없음", dup_ok,
                   "앞 20,000행 확인" if id_column else "기사 ID 없어 생략"))

    print()
    for name, ok, note in checks:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"   ({note})" if note else ""))

    all_ok = all(ok for _, ok, _ in checks)

    print()
    print("  => " + ("모든 검증 통과" if all_ok else "검증 실패. 파일을 쓰지 않습니다."))

    return {
        "all_passed": all_ok,
        "checks": [
            {"name": n, "passed": bool(o), "note": note} for n, o, note in checks
        ],
    }


def print_preview(
    original: pd.DataFrame,
    shuffled: pd.DataFrame,
    permutations: np.ndarray,
    n: int,
) -> None:
    section(f"미리보기 {n}행")

    id_column = (
        "candidate_article_ids"
        if "candidate_article_ids" in original.columns
        else None
    )

    for i in range(min(n, len(original))):
        a_labels = np.asarray(original["candidate_labels"].iloc[i])
        b_labels = np.asarray(shuffled["candidate_labels"].iloc[i])

        print()
        print(f"  --- row {i}"
              + (f"  impression_id={original['impression_id'].iloc[i]}"
                 if "impression_id" in original.columns else "")
              + f"  permutation={permutations[i].tolist()}")

        if id_column:
            a_ids = np.asarray(original[id_column].iloc[i]).tolist()
            b_ids = np.asarray(shuffled[id_column].iloc[i]).tolist()
            print(f"    before ids    : {a_ids}")
            print(f"    after  ids    : {b_ids}")

        a_sid = [
            tuple(int(np.asarray(original[c].iloc[i])[k]) for c in
                  ("candidate_c1", "candidate_c2", "candidate_c3"))
            for k in range(NUM_CANDIDATES)
        ]
        b_sid = [
            tuple(int(np.asarray(shuffled[c].iloc[i])[k]) for c in
                  ("candidate_c1", "candidate_c2", "candidate_c3"))
            for k in range(NUM_CANDIDATES)
        ]

        print(f"    before sids   : {a_sid}")
        print(f"    after  sids   : {b_sid}")
        print(f"    before labels : {a_labels.tolist()}   positive index = {positive_index_of(a_labels)}")
        print(f"    after  labels : {b_labels.tolist()}   positive index = {positive_index_of(b_labels)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--split", required=True,
        help="train / validation / test. row seed에 들어간다.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--suffix", default="_shuffled_v2")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--preview", type=int, default=0,
        help="N행만 미리 보고 파일을 쓰지 않는다.",
    )
    parser.add_argument(
        "--verify-rows", type=int, default=0,
        help="검증에 쓸 행 수. 0이면 전체 (기본 0)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()

    if not input_path.exists():
        print(f"입력 파일이 없습니다: {input_path}")
        return 1

    section("candidate order shuffle")
    print(f"  입력   : {input_path}")
    print(f"  split  : {args.split}")
    print(f"  seed   : {args.seed}")
    print()
    print("  바꾸는 것   : candidate 관련 컬럼의 원소 순서")
    print("  안 바꾸는 것: positive 선택, negative 샘플링, candidate set,")
    print("                history, SID 값, split, impression 수, 행 순서")

    df = pd.read_parquet(input_path)
    print()
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    columns = detect_candidate_columns(df)
    untouched = [c for c in df.columns if c not in columns]

    print()
    print(f"  permutation 적용 ({len(columns)}개):")
    for c in columns:
        print(f"    {c}")
    print()
    print(f"  그대로 두는 컬럼 ({len(untouched)}개):")
    for c in untouched:
        print(f"    {c}")

    for c in untouched:
        if c.startswith("candidate_"):
            print()
            print(f"  경고: {c}는 candidate_로 시작하는데 길이 5가 아니라 제외됐습니다.")

    before = position_distribution(df["candidate_labels"])

    section("shuffle 전 positive_position_distribution")
    print_distribution(f"{args.split} (before)", before)

    if before["max_deviation_from_uniform"] > 0.05:
        print()
        print("  >>> position bias가 확인됩니다.")

    work = df.head(args.preview) if args.preview else df

    shuffled, permutations = shuffle_dataframe(
        df=work,
        columns=columns,
        split_name=args.split,
        global_seed=args.seed,
    )

    if args.preview:
        print_preview(work, shuffled, permutations, args.preview)
        print()
        print("=" * 74)
        print("미리보기만 했습니다. 파일을 쓰지 않았습니다.")
        print("확인 후 --preview 없이 다시 실행하세요.")
        print("=" * 74)
        return 0

    after = position_distribution(shuffled["candidate_labels"])

    section("shuffle 후 positive_position_distribution")
    print_distribution(f"{args.split} (after)", after)

    verify_target = (
        (df.head(args.verify_rows), shuffled.head(args.verify_rows))
        if args.verify_rows else (df, shuffled)
    )
    verification = verify(verify_target[0], verify_target[1], columns)

    if not verification["all_passed"]:
        return 1

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{input_path.stem}{args.suffix}.parquet"

    if out_path.exists():
        print()
        print(f"이미 존재합니다. 덮어쓰지 않습니다: {out_path}")
        return 1

    shuffled.to_parquet(out_path, index=False)

    report = {
        "tool": "shuffle_candidate_order",
        "description": (
            "1pos4neg parquet의 candidate 순서만 deterministic하게 섞는다. "
            "재샘플링이 아니라 order permutation만 수행한다."
        ),
        "candidate_order_shuffled": True,
        "input_path": str(input_path),
        "output_path": str(out_path),
        "split": args.split,
        "global_seed": args.seed,
        "row_seed_formula": (
            "sha256(global_seed | split_name | impression_id | "
            "positive_article_id | row_uid)"
        ),
        "row_uid_definition": (
            "(impression_id, positive_article_id) 그룹 안의 0부터 시작하는 순번"
        ),
        "num_rows": int(len(df)),
        "permuted_columns": columns,
        "untouched_columns": untouched,
        "positive_position_distribution_before": before,
        "positive_position_distribution_after": after,
        "verification": verification,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "unchanged_by_construction": [
            "positive 기사 선택 방식",
            "negative 4개 샘플링 방식",
            "negative candidate set",
            "history",
            "SID 값",
            "Train/Validation/Test split",
            "impression 수",
            "행 순서",
        ],
    }

    report_path = out_dir / f"{input_path.stem}{args.suffix}_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    section("완료")
    print(f"  parquet : {out_path}")
    print(f"  report  : {report_path}")
    print()
    print("  기존 파일은 그대로입니다.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
