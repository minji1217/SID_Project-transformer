"""P2 direct score와 V1 weighted generation score가 서로 다른 정보를 갖는가.

재학습하지 않는다. 추론하지 않는다. 이미 저장된 두 score를 읽어
합쳐 보기만 한다.

    D = P2 direct bilinear score       u^T W v
    G = V1 weighted generation score   L1 + 0.5*L2 + 0.1*L3

impression 안의 후보 5개 각각에 대해 z-normalize한 뒤

    H(lambda) = lambda * zD + (1 - lambda) * zG

lambda를 0에서 1까지 훑는다.

    lambda = 0  ->  G만
    lambda = 1  ->  D만

z-normalize는 impression 안에서의 affine 변환이므로 순위를 바꾸지
않는다. 따라서 lambda=0은 기존 G 결과를, lambda=1은 기존 D 결과를
그대로 재현해야 한다. 두 끝점이 곧 재현 검증이다.

이 sweep은 Validation에서 고른다. G 자체도 Validation에서 42개 weight
조합을 훑어 얻은 값이다. 따라서 결과는 최종 성능 추정치가 아니라
"direct score가 보완적인 정보를 갖는가"를 보는 진단이다.

    cd Transformer
    python -m analysis.run_score_fusion_diagnostic \
        --direct sweep_out/ebnerd_v2/priority2_direct/regularization_sweep/do0.00_wd0.0000/seed42/validation_predictions.parquet \
        --priority0-dir sweep_out/ebnerd_v2/priority0_shuffled/seed42 \
        --out sweep_out/ebnerd_v2/priority2_direct/score_fusion
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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from evaluate.metrics import evaluate_ranking
from sweep.run_stage import write_csv


BASE_DIR = Path(__file__).resolve().parent.parent

NUM_CANDIDATES = 5
GROUP_COLUMN = "sample_index"

# G = L1 + 0.5*L2 + 0.1*L3
G_WEIGHTS = (1.0, 0.5, 0.1)

LAMBDAS = [round(0.1 * i, 1) for i in range(11)]

EPS = 1e-8

# 끝점 검증은 입력 파일에서 직접 다시 계산한 raw metric으로 한다.
# 아래 값들은 표에 같이 보여주는 참고값일 뿐, 합격 기준이 아니다.
REPRODUCE_TOLERANCE = 1e-9

RANDOM_TOP1 = 0.20
V1_EQUAL_REFERENCE = 0.27059
G_PAST_REFERENCE = 0.29312
D_PAST_REFERENCE = 0.26689

CAVEAT_EN = (
    "This hybrid sweep is Validation-tuned and is used only to diagnose\n"
    "whether the direct score contains complementary ranking information.\n"
    "It is not a final unbiased performance estimate."
)


def spearman(a: pd.Series, b: pd.Series) -> float:
    """rank를 매긴 뒤 Pearson을 낸다.

    pandas의 method="spearman"은 scipy를 부르는데 이 환경에는 없다.
    동점은 평균 순위로 처리한다 (pandas rank 기본값). 정의상 같다.
    """
    return float(a.rank().corr(b.rank(), method="pearson"))


_STARTED_AT = time.time()


def section(title: str) -> None:
    print()
    print("=" * 96)
    print(f"{title}   [+{time.time() - _STARTED_AT:.0f}s]")
    print("=" * 96)


def step(message: str) -> None:
    """오래 걸리는 단계 앞에 찍는다. 멈춘 게 아니라는 표시."""
    print(f"  ... {message}  [+{time.time() - _STARTED_AT:.0f}s]")


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


HISTORY_FILTER_COLUMN = "history_c1"

SOURCE_COLUMNS = [
    "impression_id",
    "candidate_article_ids",
    "candidate_c1", "candidate_c2", "candidate_c3",
    "candidate_labels",
]


def load_candidate_source(validation_path: Path) -> pd.DataFrame:
    """원본 validation parquet을 후보 단위로 편다.

    impression_id는 원본에서 유일하지 않으므로 키로 쓰지 않는다.
    대신 예측이 들고 있는 `sample_index`를 쓴다. 이 값은 validation
    loader를 shuffle 없이 돌 때의 일련번호이고, Dataset은 원본을 읽은
    순서 그대로 두되 history가 빈 행만 건너뛴다. 그래서 같은 규칙으로
    거르면 sample_index -> 원본 row를 그대로 되살릴 수 있다.

    되살린 대응은 믿고 쓰는 게 아니라, 붙인 뒤 impression_id / label /
    (c1,c2,c3) / (있으면) 기사 ID가 원본과 같은지로 확인한다.
    """
    available = set(
        pq.ParquetFile(validation_path).schema_arrow.names
    )
    needed = SOURCE_COLUMNS + [HISTORY_FILTER_COLUMN]
    missing = [c for c in needed if c not in available]

    if missing:
        raise ValueError(
            f"원본 validation parquet에 필요한 컬럼이 없습니다: "
            f"{', '.join(missing)}\n"
            f"  파일 : {validation_path}\n"
            "  이 진단은 기사 ID를 원본에서만 가져옵니다. "
            "(c1,c2,c3)를 대체키로 쓰지 않습니다."
        )

    step(f"원본 읽는 중: {validation_path.name}")
    frame = pd.read_parquet(validation_path, columns=needed)

    total_rows = len(frame)

    step(f"history가 빈 행을 거르는 중 ({total_rows:,} row)")

    # NewsSequenceDataset._build_sample_index와 같은 규칙.
    # drop_empty_history는 gin에서 항상 True다.
    keep = np.array([
        row_index
        for row_index, value in enumerate(frame[HISTORY_FILTER_COLUMN])
        if len(np.asarray(value)) > 0
    ], dtype=np.int64)

    dropped = total_rows - len(keep)

    print(f"  원본 row {total_rows:,}개")
    print(f"  history가 빈 행 제외 {dropped:,}개")
    print(f"  Dataset이 쓰는 sample {len(keep):,}개")

    duplicated = int(frame["impression_id"].duplicated().sum())

    if duplicated:
        print(f"  impression_id 중복 {duplicated:,}개 "
              "-> 키로 쓰지 않고 확인용으로만 씁니다")

    frame = frame.iloc[keep].reset_index(drop=True)

    def stack(column: str) -> np.ndarray:
        values = [np.asarray(v) for v in frame[column]]
        lengths = {len(v) for v in values}

        if lengths != {NUM_CANDIDATES}:
            raise ValueError(
                f"{column}의 길이가 {NUM_CANDIDATES}가 아닌 행이 있습니다: "
                f"{sorted(lengths)}"
            )

        return np.stack(values)

    step(f"후보 단위로 펴는 중 ({len(frame):,} sample)")

    article_ids = stack("candidate_article_ids")
    c1 = stack("candidate_c1")
    c2 = stack("candidate_c2")
    c3 = stack("candidate_c3")
    labels = stack("candidate_labels")

    num_rows = len(frame)

    return pd.DataFrame({
        GROUP_COLUMN: np.repeat(np.arange(num_rows), NUM_CANDIDATES),
        "candidate_index": np.tile(
            np.arange(NUM_CANDIDATES), num_rows
        ),
        "source_row": np.repeat(keep, NUM_CANDIDATES),
        "source_impression_id": np.repeat(
            frame["impression_id"].to_numpy(), NUM_CANDIDATES
        ),
        "source_article_id": article_ids.reshape(-1),
        "source_c1": c1.reshape(-1).astype(np.int64),
        "source_c2": c2.reshape(-1).astype(np.int64),
        "source_c3": c3.reshape(-1).astype(np.int64),
        "source_label": labels.reshape(-1).astype(np.int64),
    })


RAW_PREDICTION_GLOB = "validation_scores_*.parquet"


def attach_own_article_id(
    frame: pd.DataFrame, priority0_dir: Path
) -> pd.DataFrame:
    """generation 예측에 그 쪽 dataloader가 준 기사 ID를 붙인다.

    candidate_scores.parquet은 기사 ID를 버리지만, 같은 폴더의 raw
    예측(predict_sid.py 출력)에는 남아 있다. 이것은 원본에서 붙이는
    값과 독립적인 증거이므로, 있으면 붙여서 확인 항목을 하나 늘린다.
    원본 대조를 대신하지는 않는다. 없으면 그냥 건너뛴다.
    """
    if "candidate_article_id" in frame.columns:
        return frame

    candidates = sorted(priority0_dir.glob(RAW_PREDICTION_GLOB))

    for path in candidates:
        step(f"generation 기사 ID를 찾는 중: {path.name}")

        columns = set(pq.ParquetFile(path).schema_arrow.names)

        if not {GROUP_COLUMN, "article_id"} <= columns:
            continue

        raw = pd.read_parquet(path, columns=[GROUP_COLUMN, "article_id"])
        raw = raw.sort_values(GROUP_COLUMN, kind="stable").reset_index(drop=True)
        raw["candidate_index"] = raw.groupby(GROUP_COLUMN).cumcount()

        merged = frame.merge(
            raw.rename(columns={"article_id": "candidate_article_id"}),
            on=[GROUP_COLUMN, "candidate_index"],
            how="left", validate="1:1",
        )

        if int(merged["candidate_article_id"].isna().sum()):
            continue

        print()
        print(f"  generation 기사 ID를 raw 예측에서 읽었습니다: {path.name}")
        print("  (원본 대조를 대신하지 않고, 확인 항목만 하나 늘립니다.)")

        return merged

    return frame


def attach_source(
    frame: pd.DataFrame, source: pd.DataFrame, name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """예측에 원본 기사 ID를 붙이고, 원본과 맞는지 확인한다.

    키는 `(sample_index, candidate_index)`다. impression_id는 원본에서
    유일하지 않으므로 (positive가 여럿인 impression을 positive마다
    따로 편 행들이 같은 id를 유지한다) 키로 쓰지 않고, 붙인 뒤
    맞는지 확인하는 항목으로만 쓴다. 중복 impression_id는 버리지
    않는다.

    붙인 뒤 예측이 들고 있던 impression_id / label / c1/c2/c3 /
    (있으면) 기사 ID가 원본과 같은지 본다. 여기가 맞으면 그 예측이
    원본의 어느 후보를 매겼는지 확정된다.
    """
    section(f"원본 대조 — {name}")

    before = len(frame)

    for column in (GROUP_COLUMN, "candidate_index"):
        if column not in frame.columns:
            raise ValueError(
                f"{name} 예측에 {column}이 없습니다. "
                f"({GROUP_COLUMN}, candidate_index)로 붙일 수 없습니다."
            )

    duplicated_key = int(
        frame.duplicated([GROUP_COLUMN, "candidate_index"]).sum()
    )

    if duplicated_key:
        raise ValueError(
            f"{name}에 ({GROUP_COLUMN}, candidate_index)가 중복됩니다 "
            f"({duplicated_key:,}개). 후보를 유일하게 잡을 수 없습니다."
        )

    # validate="1:1"이 양쪽 키 중복을 다시 한 번 막는다.
    merged = frame.merge(
        source, on=[GROUP_COLUMN, "candidate_index"],
        how="left", validate="1:1",
    )

    if len(merged) != before:
        raise ValueError(
            f"{name}에 원본을 붙이면서 row 수가 바뀌었습니다: "
            f"{before:,} -> {len(merged):,}"
        )

    unmatched = int(merged["source_article_id"].isna().sum())

    print(f"  row {before:,}개")
    print(f"  원본에서 찾지 못한 후보: {unmatched:,}")

    if unmatched:
        raise ValueError(
            f"{name}의 후보 {unmatched:,}개가 원본에 없습니다. "
            "impression_id 또는 후보 위치가 맞지 않습니다."
        )

    checks: List[Dict[str, Any]] = []

    def check(label: str, left: pd.Series, right: pd.Series) -> None:
        a = left.astype(np.int64).to_numpy()
        b = right.astype(np.int64).to_numpy()
        mismatched = int((a != b).sum())

        checks.append({
            "field": label, "mismatched": mismatched,
            "passed": mismatched == 0,
        })

        print(f"  {'OK  ' if mismatched == 0 else 'FAIL'}  "
              f"{label:<28} 불일치 {mismatched:,} / {len(a):,}")

        if mismatched:
            raise ValueError(
                f"{name}의 {label}이 원본과 다릅니다 ({mismatched:,}개). "
                "예측이 원본과 다른 후보를 가리킵니다."
            )

    check("impression_id", merged["impression_id"],
          merged["source_impression_id"])
    check("label", merged["label"], merged["source_label"])

    for level in ("c1", "c2", "c3"):
        if level in merged.columns:
            check(f"candidate {level}", merged[level],
                  merged[f"source_{level}"])

    source_article_id = merged["source_article_id"].astype(str)

    # 예측이 스스로 들고 온 기사 ID가 있으면 원본과 대조한다. 이 값은
    # dataloader가 모델에 실제로 넣어 준 것이라, 원본에서 붙인 값과
    # 독립적인 증거다. 없으면 (article_id 저장 전에 만든 예측이면)
    # 건너뛴다. 어느 쪽이든 최종 기사 ID는 원본 값을 쓴다.
    own = merged.get("candidate_article_id")
    own_present = own is not None and int(own.notna().sum()) > 0

    if own_present:
        a = own.astype(str).to_numpy()
        b = source_article_id.to_numpy()
        mismatched = int((a != b).sum())

        checks.append({
            "field": "candidate_article_id (예측 자체)",
            "mismatched": mismatched, "passed": mismatched == 0,
        })

        print(f"  {'OK  ' if mismatched == 0 else 'FAIL'}  "
              f"{'candidate_article_id (예측 자체)':<28} "
              f"불일치 {mismatched:,} / {len(a):,}")

        if mismatched:
            raise ValueError(
                f"{name}이 저장한 기사 ID가 원본과 다릅니다 "
                f"({mismatched:,}개). 예측이 원본과 다른 후보를 "
                "가리킵니다."
            )
    else:
        checks.append({
            "field": "candidate_article_id (예측 자체)",
            "mismatched": None, "passed": None,
        })
        print(f"  --    {'candidate_article_id (예측 자체)':<28} "
              f"예측에 없어 건너뜀")

    merged["candidate_article_id"] = source_article_id

    print()
    print(f"  => {name}의 후보가 원본과 일치합니다. 기사 ID를 붙였습니다.")

    if not own_present:
        print(f"     ({name} 예측에는 기사 ID가 없어 원본 값만 씁니다. "
              "그 예측을 기사 ID까지 저장하도록 다시 만들면 "
              "교차 확인됩니다.)")

    return merged.drop(columns=[
        "source_row", "source_impression_id", "source_article_id",
        "source_c1", "source_c2", "source_c3", "source_label",
    ]), {"checks": checks, "unmatched": unmatched}


# ---------------------------------------------------------------- 정렬 검증


def align(
    direct: pd.DataFrame, generation: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """두 score가 같은 후보를 가리키는지 확인한 뒤 합친다."""
    section("정렬 검증")

    for name, df in (("direct", direct), ("generation", generation)):
        counts = df.groupby(GROUP_COLUMN).size()

        if not (counts == NUM_CANDIDATES).all():
            raise ValueError(
                f"{name}에 후보가 {NUM_CANDIDATES}개가 아닌 impression이 있습니다."
            )

    if len(direct) != len(generation):
        raise ValueError(
            f"row 수가 다릅니다: direct {len(direct):,} vs "
            f"generation {len(generation):,}"
        )

    direct = direct.sort_values(
        [GROUP_COLUMN, "candidate_index"], kind="stable"
    ).reset_index(drop=True)
    generation = generation.sort_values(
        [GROUP_COLUMN, "candidate_index"], kind="stable"
    ).reset_index(drop=True)

    print(f"  키는 ({GROUP_COLUMN}, candidate_index)입니다.")
    print(f"  row {len(direct):,}개, sample "
          f"{len(direct) // NUM_CANDIDATES:,}개")
    print()

    checks: List[Dict[str, Any]] = []
    all_ok = True

    def check(label: str, a: pd.Series, b: pd.Series) -> None:
        nonlocal all_ok

        left = a.astype(str).to_numpy()
        right = b.astype(str).to_numpy()
        mismatched = int((left != right).sum())
        ok = mismatched == 0

        all_ok = all_ok and ok
        checks.append({
            "field": label, "mismatched": mismatched, "passed": ok,
        })

        print(f"  {'OK  ' if ok else 'FAIL'}  {label:<28} "
              f"불일치 {mismatched:,} / {len(left):,}")

    check("sample_index", direct[GROUP_COLUMN], generation[GROUP_COLUMN])
    check("candidate position", direct["candidate_index"],
          generation["candidate_index"])
    # impression_id는 원본에서 유일하지 않다. 키가 아니라 확인 항목이다.
    check("impression_id", direct["impression_id"], generation["impression_id"])
    check("label", direct["label"].astype(int),
          generation["label"].astype(int))

    for level in ("c1", "c2", "c3"):
        if level in direct.columns and level in generation.columns:
            check(f"candidate {level}", direct[level], generation[level])

    article_column = "candidate_article_id"

    for name, frame in (("direct", direct), ("generation", generation)):
        if article_column not in frame.columns:
            raise ValueError(
                f"{name}에 {article_column}이 없습니다. 원본 붙이기가 "
                "먼저 끝났는지 확인하세요."
            )

    check("candidate_article_id", direct[article_column],
          generation[article_column])

    print()

    if not all_ok:
        raise ValueError(
            "두 score의 후보 정렬이 어긋납니다. 위 불일치를 먼저 확인하세요."
        )

    print("  => 같은 후보를 가리킵니다.")

    merged = pd.DataFrame({
        "impression_id": direct["impression_id"],
        GROUP_COLUMN: direct[GROUP_COLUMN],
        "candidate_index": direct["candidate_index"],
        "label": direct["label"].astype(int),
        "c1": direct["c1"], "c2": direct["c2"], "c3": direct["c3"],
        "D": direct["direct_score"].astype(np.float64),
        "L1": generation["L1"].astype(np.float64),
        "L2": generation["L2"].astype(np.float64),
        "L3": generation["L3"].astype(np.float64),
    })

    merged[article_column] = direct[article_column]

    merged["G"] = (
        G_WEIGHTS[0] * merged["L1"]
        + G_WEIGHTS[1] * merged["L2"]
        + G_WEIGHTS[2] * merged["L3"]
    )

    return merged, {"checks": checks, "all_passed": all_ok}


# ---------------------------------------------------------------- z-normalize


def z_normalize(
    values: np.ndarray, num_impressions: int
) -> Tuple[np.ndarray, int]:
    """impression 안의 후보 5개끼리 z-normalize한다.

    std가 0에 가까우면 후보 5개 점수가 사실상 같다는 뜻이다. eps를
    더해 0으로 나누는 것을 막고, 그런 impression 수를 따로 센다.
    """
    grid = values.reshape(num_impressions, NUM_CANDIDATES)

    mean = grid.mean(axis=1, keepdims=True)
    std = grid.std(axis=1, ddof=0, keepdims=True)

    degenerate = int((std.reshape(-1) <= EPS).sum())

    return ((grid - mean) / (std + EPS)).reshape(-1), degenerate


# ---------------------------------------------------------------- 평가


def evaluate_scores(
    frame: pd.DataFrame, scores: np.ndarray
) -> Dict[str, float]:
    work = frame[[GROUP_COLUMN, "label"]].copy()
    work["candidate_score"] = scores
    metrics = evaluate_ranking(df=work, group_column=GROUP_COLUMN)

    return {
        "top1_accuracy": metrics["top1_accuracy"],
        "mrr": metrics["mrr"],
        "ndcg5": metrics["ndcg@5"],
        "auc": metrics["auc"],
        "num_impressions": metrics["num_impressions"],
    }


def impression_ranks(scores: np.ndarray, num_impressions: int) -> np.ndarray:
    """impression 안에서 점수 내림차순 순위. 동점은 안정 정렬로 가른다."""
    grid = scores.reshape(num_impressions, NUM_CANDIDATES)
    order = np.argsort(-grid, axis=1, kind="stable")

    ranks = np.empty_like(order)
    positions = np.arange(NUM_CANDIDATES)

    for i in range(num_impressions):
        ranks[i, order[i]] = positions

    return ranks


def check_endpoint(
    name: str, raw: np.ndarray, z: np.ndarray,
    ordered: pd.DataFrame, num_impressions: int,
    past_reference: float,
) -> Dict[str, Any]:
    """z 적용 전후가 같은 순위를 내는지, metric이 같은지 본다.

    기준은 입력 파일에서 직접 다시 계산한 raw metric이다. 예전 실행의
    수치는 참고로 같이 보여주기만 한다.
    """
    raw_metrics = evaluate_scores(ordered, raw)
    z_metrics = evaluate_scores(ordered, z)

    raw_ranks = impression_ranks(raw, num_impressions)
    z_ranks = impression_ranks(z, num_impressions)

    changed = int((raw_ranks != z_ranks).any(axis=1).sum())

    deltas = {
        key: abs(z_metrics[key] - raw_metrics[key])
        for key in ("top1_accuracy", "mrr", "ndcg5", "auc")
    }

    worst = max(deltas.values())
    passed = changed == 0 and worst <= REPRODUCE_TOLERANCE

    print(f"  {name}")
    print(f"    {'':<14} {'raw':>14} {'z 적용 후':>14} {'차이':>12}")
    print(f"    " + "-" * 56)

    for label, key in (
        ("Top-1", "top1_accuracy"), ("MRR", "mrr"),
        ("nDCG@5", "ndcg5"), ("AUC", "auc"),
    ):
        print(f"    {label:<14} {raw_metrics[key]:>14.10f} "
              f"{z_metrics[key]:>14.10f} {deltas[key]:>12.3e}")

    print()
    print(f"    후보 순위가 바뀐 impression : {changed:,} / "
          f"{num_impressions:,}")
    print(f"    판정 : {'OK' if passed else 'FAIL'}")
    print(f"    (참고) 예전 실행 값 : {past_reference * 100:.3f}%  "
          f"— 합격 기준이 아닙니다")
    print()

    return {
        "name": name,
        "raw": raw_metrics,
        "z": z_metrics,
        "deltas": deltas,
        "ranks_changed": changed,
        "passed": passed,
        "past_reference": past_reference,
    }


def correct_mask(
    scores: np.ndarray, labels: np.ndarray, num_impressions: int
) -> np.ndarray:
    """impression별로 1등이 정답인지."""
    grid = scores.reshape(num_impressions, NUM_CANDIDATES)
    label_grid = labels.reshape(num_impressions, NUM_CANDIDATES)

    return grid.argmax(axis=1) == label_grid.argmax(axis=1)


# ---------------------------------------------------------------- 보고


def print_lambda_table(rows: List[Dict[str, Any]]) -> None:
    section("lambda sweep")

    header = (
        f"  {'lambda':>7} {'Top-1':>10} {'MRR':>9} {'nDCG@5':>9} "
        f"{'AUC':>9}   설명"
    )
    print(header)
    print("  " + "-" * (len(header) + 6))

    best = max(rows, key=lambda r: r["top1_accuracy"])

    for row in rows:
        note = ""

        if row["lambda"] == 0.0:
            note = "G만 (generation)"
        elif row["lambda"] == 1.0:
            note = "D만 (direct)"

        if row is best:
            note = (note + "  <- 최고").strip()

        print(
            f"  {row['lambda']:>7.1f} {row['top1_accuracy'] * 100:>9.3f}% "
            f"{row['mrr']:>9.4f} {row['ndcg5']:>9.4f} {row['auc']:>9.4f}   "
            f"{note}"
        )


def print_agreement(agreement: Dict[str, Any]) -> None:
    section("G와 D의 정답 분포")

    total = agreement["total"]

    print(f"  {'':<26} {'impression':>12} {'비율':>9}")
    print("  " + "-" * 50)

    for label, key in (
        ("G correct / D correct", "both_correct"),
        ("G correct / D wrong", "g_only"),
        ("G wrong   / D correct", "d_only"),
        ("G wrong   / D wrong", "both_wrong"),
    ):
        count = agreement[key]
        print(f"  {label:<26} {count:>12,} {100 * count / total:>8.2f}%")

    print()
    print(f"  {'D-only correct / 전체 Validation':<44} "
          f"{agreement['d_only']:>8,} / {total:,}  "
          f"({agreement['d_only_of_total'] * 100:.2f}%)")

    g_wrong = agreement["g_only"] + agreement["both_wrong"]

    print(f"  {'D-only correct / G가 틀린 subset':<44} "
          f"{agreement['d_only']:>8,} / {g_wrong:,}  "
          f"({agreement['d_only_of_g_wrong'] * 100:.2f}%)")

    print()
    print(f"  {'oracle union Top-1':<32} "
          f"{agreement['union'] * 100:>9.3f}%")
    print(f"  {'G Top-1':<32} {agreement['g_top1'] * 100:>9.3f}%")
    print(f"  {'D Top-1':<32} {agreement['d_top1'] * 100:>9.3f}%")
    print(f"  {'best Hybrid Top-1':<32} "
          f"{agreement['best_hybrid_top1'] * 100:>9.3f}%  "
          f"(lambda {agreement['best_lambda']:g})")
    print(f"  {'best Hybrid - G':<32} "
          f"{agreement['best_hybrid_minus_g'] * 100:>+9.3f}%p")

    print()
    print("  oracle union은 두 score 중 맞힌 쪽을 매번 골랐을 때의 Top-1로,")
    print("  어떤 fusion도 넘을 수 없는 상한입니다.")


def print_correlation(correlation: Dict[str, float]) -> None:
    section("D와 G의 상관")

    print(f"  {'기준':<34} {'Pearson':>10} {'Spearman':>10}")
    print("  " + "-" * 56)

    for label, pearson_key, spearman_key in (
        ("raw", "pearson_raw", "spearman_raw"),
        ("z-score", "pearson_z", "spearman_z"),
        ("impression-level", "pearson_per_impression",
         "spearman_per_impression"),
    ):
        print(f"  {label:<34} {correlation[pearson_key]:>10.4f} "
              f"{correlation[spearman_key]:>10.4f}")

    print()
    print("  raw와 z-score는 후보 전체를 한 번에 본 값이고,")
    print("  impression-level은 impression마다 계산해 평균한 값입니다.")
    print("  ranking에 직접 관련되는 것은 impression-level입니다.")


def print_comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("비교")

    best = max(rows, key=lambda r: r["top1_accuracy"])

    g_row = next(r for r in rows if r["lambda"] == 0.0)
    d_row = next(r for r in rows if r["lambda"] == 1.0)

    table = [
        ("Random", RANDOM_TOP1, "이론값"),
        ("P2 Direct  D (이번 입력)", d_row["top1_accuracy"],
         "lambda=1"),
        ("V1 weighted G (이번 입력)", g_row["top1_accuracy"],
         "lambda=0"),
        (f"Hybrid best  lambda={best['lambda']:g}",
         best["top1_accuracy"], "이번 진단"),
    ]

    for name, value, note in table:
        print(f"  {name:<36} {value * 100:>8.3f}%   {note}")

    print()
    print("  참고값 (예전 실행. 이번 검증 기준이 아닙니다)")
    print(f"    {'V1 equal  L1+L2+L3':<34} "
          f"{V1_EQUAL_REFERENCE * 100:>8.3f}%")
    print(f"    {'V1 weighted G':<34} {G_PAST_REFERENCE * 100:>8.3f}%")
    print(f"    {'P2 Direct':<34} {D_PAST_REFERENCE * 100:>8.3f}%")

    print()
    print(f"  best lambda : {best['lambda']:g}")
    print(f"  G 대비      : "
          f"{(best['top1_accuracy'] - g_row['top1_accuracy']) * 100:+.3f}%p")
    print(f"  D 대비      : "
          f"{(best['top1_accuracy'] - d_row['top1_accuracy']) * 100:+.3f}%p")

    print()
    print("  " + CAVEAT_EN.replace("\n", "\n  "))

    return best


def write_readme(
    path: Path, rows: List[Dict[str, Any]], agreement: Dict[str, Any],
    correlation: Dict[str, float], alignment: Dict[str, Any],
    endpoint_checks: List[Dict[str, Any]], best: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    lines: List[str] = []
    add = lines.append

    add("# score fusion diagnostic")
    add("")
    add("재학습하지 않았다. 추론하지 않았다. 이미 저장된 두 score를 읽어")
    add("합쳐 보기만 했다.")
    add("")
    add("    D = P2 direct bilinear score       u^T W v")
    add(f"    G = V1 weighted generation score   "
        f"L1 + {G_WEIGHTS[1]:g}*L2 + {G_WEIGHTS[2]:g}*L3")
    add("")
    add("G는 generation 파일의 `L1`/`L2`/`L3`에서 다시 계산한다. 그 파일의")
    add("`candidate_score`와 `S123`은 `L1 + L2 + L3`이라 이번 G가 아니므로")
    add("읽지 않는다.")
    add("")
    add("impression 안의 후보 5개끼리 z-normalize한 뒤")
    add("")
    add("    H(lambda) = lambda * zD + (1 - lambda) * zG")
    add("")
    add("## 주의")
    add("")
    add("```")
    add(CAVEAT_EN)
    add("```")
    add("")
    add(f"`G = L1 + {G_WEIGHTS[1]:g}*L2 + {G_WEIGHTS[2]:g}*L3` 자체가 이미")
    add("Validation에서 42개 weight 조합을 탐색해 얻은 값이다. 이번 lambda")
    add("sweep도 Validation에서 하므로, hybrid 최고 성능은 최종 성능")
    add("추정치가 아니라 development diagnostic이다.")
    add("")

    add("## 핵심 결과")
    add("")
    total = agreement["total"]
    add(f"1. **G wrong / D correct** : "
        f"**{agreement['d_only']:,}** impression "
        f"(전체의 {agreement['d_only_of_total'] * 100:.2f}%, "
        f"G가 틀린 것 중 {agreement['d_only_of_g_wrong'] * 100:.2f}%)")
    add(f"2. **oracle union Top-1** : "
        f"**{agreement['union'] * 100:.3f}%** "
        f"(G {agreement['g_top1'] * 100:.3f}%, "
        f"D {agreement['d_top1'] * 100:.3f}%)")
    add(f"3. **best Hybrid Top-1** : "
        f"**{agreement['best_hybrid_top1'] * 100:.3f}%** "
        f"(lambda {agreement['best_lambda']:g}), "
        f"G 대비 **{agreement['best_hybrid_minus_g'] * 100:+.3f}%p**")
    add("")
    add("oracle union은 두 score 중 맞힌 쪽을 매번 골랐을 때의 Top-1로,")
    add("어떤 fusion도 넘을 수 없는 상한이다.")
    add("")

    add("## 실행 정보")
    add("")
    add(f"- direct score  : `{metadata['direct_path']}`")
    add(f"- generation    : `{metadata['generation_path']}`")
    add(f"- impression    : {metadata['num_impressions']:,}")
    add(f"- lambda        : {', '.join(f'{v:g}' for v in metadata['lambdas'])}")
    add(f"- git commit    : {metadata.get('git_commit')}")
    add("")

    add("## 원본 대조")
    add("")
    add("기사 ID는 예측 파일이 아니라 모델에 실제로 들어간 원본")
    add(f"`{alignment['source_path']}`에서 가져왔다.")
    add("키는 `(sample_index, candidate_index)`다. `impression_id`는 원본에서")
    add("유일하지 않다 — positive가 여럿인 impression을 positive마다 따로 편")
    add("행들이 같은 id를 유지하기 때문이다. 그래서 키로 쓰지 않고 붙인 뒤")
    add("확인하는 항목으로만 쓴다. 중복된 `impression_id`는 하나도 버리지")
    add("않았다.")
    add("")
    add("`sample_index`는 shuffle 없이 도는 validation loader의 일련번호이고,")
    add("Dataset은 원본을 읽은 순서를 지키되 history가 빈 행만 건너뛴다")
    add("(`NewsSequenceDataset._build_sample_index`, `drop_empty_history=True`).")
    add("같은 규칙으로 걸러 `sample_index -> 원본 row`를 되살린 뒤,")
    add("`impression_id` / `label` / `(c1,c2,c3)` / 기사 ID로 확인했다.")
    add("")
    add("예측이 스스로 저장한 `candidate_article_id`는 dataloader가 모델에")
    add("넣어 준 값이라, 원본에서 붙인 값과 독립적인 증거다. 예측에 그")
    add("컬럼이 있으면 같이 대조하고, 없으면 건너뛴다.")
    add("")

    for name, key in (("direct", "direct_vs_source"),
                      ("generation", "generation_vs_source")):
        add(f"**{name}**")
        add("")
        add("| 항목 | 불일치 | 판정 |")
        add("|---|---|---|")

        for check in alignment[key]["checks"]:
            if check["mismatched"] is None:
                add(f"| {check['field']} | - | 예측에 없어 건너뜀 |")
            else:
                add(f"| {check['field']} | {check['mismatched']:,} | "
                    f"{'OK' if check['passed'] else 'FAIL'} |")

        add("")

    add("## 정렬 검증  (direct vs generation)")
    add("")
    add("| 항목 | 불일치 | 판정 |")
    add("|---|---|---|")

    for check in alignment["checks"]:
        add(f"| {check['field']} | {check['mismatched']:,} | "
            f"{'OK' if check['passed'] else 'FAIL'} |")

    add("")

    add("## 끝점 검증")
    add("")
    add("z-normalize는 impression 안의 affine 변환이라 순위를 바꾸지 않아야")
    add("한다. 기준은 **이번에 읽은 입력 파일에서 직접 다시 계산한 raw**")
    add("metric이다. 예전 실행 수치는 참고로만 적는다.")
    add("")

    for item in endpoint_checks:
        add(f"**{item['name']}**")
        add("")
        add("| 지표 | raw | z 적용 후 | 차이 |")
        add("|---|---|---|---|")

        for label, key in (
            ("Top-1", "top1_accuracy"), ("MRR", "mrr"),
            ("nDCG@5", "ndcg5"), ("AUC", "auc"),
        ):
            add(f"| {label} | {item['raw'][key]:.10f} | "
                f"{item['z'][key]:.10f} | {item['deltas'][key]:.3e} |")

        add("")
        add(f"- 후보 순위가 바뀐 impression : {item['ranks_changed']:,}")
        add(f"- 판정 : {'OK' if item['passed'] else 'FAIL'}")
        add(f"- (참고) 예전 실행 값 : "
            f"{item['past_reference'] * 100:.3f}% — 합격 기준이 아니다")
        add("")

    add("## lambda sweep")
    add("")
    add("| lambda | Top-1 | MRR | nDCG@5 | AUC | |")
    add("|---|---|---|---|---|---|")

    for row in rows:
        note = ""

        if row["lambda"] == 0.0:
            note = "G만"
        elif row["lambda"] == 1.0:
            note = "D만"

        if row["lambda"] == best["lambda"]:
            note = (note + " **최고**").strip()

        add(
            f"| {row['lambda']:g} | {row['top1_accuracy'] * 100:.3f}% | "
            f"{row['mrr']:.4f} | {row['ndcg5']:.4f} | {row['auc']:.4f} | "
            f"{note} |"
        )

    add("")

    add("## G와 D의 정답 분포")
    add("")
    add("| 경우 | impression | 비율 |")
    add("|---|---|---|")

    total = agreement["total"]

    for label, key in (
        ("G 정답 / D 정답", "both_correct"),
        ("G 정답 / D 오답", "g_only"),
        ("**G 오답 / D 정답**", "d_only"),
        ("G 오답 / D 오답", "both_wrong"),
    ):
        add(f"| {label} | {agreement[key]:,} | "
            f"{100 * agreement[key] / total:.2f}% |")

    add("")
    add("| 항목 | 값 |")
    add("|---|---|")
    add(f"| D-only correct / 전체 Validation | "
        f"{agreement['d_only']:,} / {agreement['total']:,} "
        f"({agreement['d_only_of_total'] * 100:.2f}%) |")
    add(f"| D-only correct / G가 틀린 subset | "
        f"{agreement['d_only']:,} / "
        f"{agreement['g_only'] + agreement['both_wrong']:,} "
        f"({agreement['d_only_of_g_wrong'] * 100:.2f}%) |")
    add(f"| oracle union Top-1 | {agreement['union'] * 100:.3f}% |")
    add(f"| G Top-1 | {agreement['g_top1'] * 100:.3f}% |")
    add(f"| D Top-1 | {agreement['d_top1'] * 100:.3f}% |")
    add(f"| best Hybrid Top-1 | "
        f"{agreement['best_hybrid_top1'] * 100:.3f}% "
        f"(lambda {agreement['best_lambda']:g}) |")
    add(f"| best Hybrid - G | "
        f"{agreement['best_hybrid_minus_g'] * 100:+.3f}%p |")
    add("")

    add("## D와 G의 상관")
    add("")
    add("| 기준 | Pearson | Spearman |")
    add("|---|---|---|")
    add(f"| raw | {correlation['pearson_raw']:.4f} | "
        f"{correlation['spearman_raw']:.4f} |")
    add(f"| z-score | {correlation['pearson_z']:.4f} | "
        f"{correlation['spearman_z']:.4f} |")
    add(f"| impression-level | "
        f"{correlation['pearson_per_impression']:.4f} | "
        f"{correlation['spearman_per_impression']:.4f} |")
    add("")
    add("raw와 z-score는 후보 전체를 한 번에 본 값이고, impression-level은")
    add("impression마다 계산해 평균한 값이다. ranking에 직접 관련되는 것은")
    add("impression-level이다.")
    add("")

    add("## 비교")
    add("")
    g_row = next(r for r in rows if r["lambda"] == 0.0)
    d_row = next(r for r in rows if r["lambda"] == 1.0)

    add("| 구분 | Top-1 | 비고 |")
    add("|---|---|---|")
    add(f"| Random | {RANDOM_TOP1 * 100:.2f}% | 이론값 |")
    add(f"| P2 Direct D | {d_row['top1_accuracy'] * 100:.3f}% | "
        f"이번 입력, lambda=1 |")
    add(f"| V1 weighted G | {g_row['top1_accuracy'] * 100:.3f}% | "
        f"이번 입력, lambda=0 |")
    add(f"| **Hybrid best** (lambda={best['lambda']:g}) | "
        f"{best['top1_accuracy'] * 100:.3f}% | 이번 진단 |")
    add("")
    add("참고값 (예전 실행. 이번 검증 기준이 아니다)")
    add("")
    add("| 구분 | Top-1 |")
    add("|---|---|")
    add(f"| V1 equal L1+L2+L3 | {V1_EQUAL_REFERENCE * 100:.3f}% |")
    add(f"| V1 weighted G | {G_PAST_REFERENCE * 100:.3f}% |")
    add(f"| P2 Direct | {D_PAST_REFERENCE * 100:.3f}% |")
    add("")

    add(f"- best lambda : {best['lambda']:g}")
    add(f"- G 대비 : "
        f"{(best['top1_accuracy'] - g_row['top1_accuracy']) * 100:+.3f}%p")
    add(f"- D 대비 : "
        f"{(best['top1_accuracy'] - d_row['top1_accuracy']) * 100:+.3f}%p")
    add("")

    add("## 하지 않은 것")
    add("")
    add("- P2 재학습, 추론")
    add("- dropout / weight decay 재탐색, LR 변경")
    add("- hidden weight, beta4 변경")
    add("- normalization layer 추가, MLP, candidate c4")
    add("- level-wise direct scorer (D1/D2/D3)")
    add("- Test 평가")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P2 direct score와 V1 weighted generation score의 보완성 진단"
    )
    parser.add_argument(
        "--direct", required=True,
        help="P2 validation_predictions.parquet",
    )
    parser.add_argument(
        "--priority0-dir",
        default="sweep_out/ebnerd_v2/priority0_shuffled/seed42",
    )
    parser.add_argument(
        "--generation", default=None,
        help="생략하면 priority0-dir의 candidate_scores.parquet",
    )
    parser.add_argument(
        "--validation-path", default=None,
        help=(
            "원본 validation parquet. 생략하면 Priority 0 metadata의 "
            "dataset을 쓴다. 여기서 candidate_article_ids를 가져온다."
        ),
    )
    parser.add_argument(
        "--out",
        default="sweep_out/ebnerd_v2/priority2_direct/score_fusion",
    )

    args = parser.parse_args()

    started_at = time.time()

    direct_path = resolve_path(args.direct)
    out_dir = resolve_path(args.out)

    priority0_dir = resolve_path(args.priority0_dir)

    generation_path = (
        resolve_path(args.generation) if args.generation
        else priority0_dir / "candidate_scores.parquet"
    )

    if args.validation_path:
        validation_path = resolve_path(args.validation_path)
    else:
        meta_path = priority0_dir / "run_metadata.json"

        if not meta_path.exists():
            print(f"Priority 0 run_metadata.json이 없습니다: {meta_path}")
            print("--validation-path로 원본 parquet을 지정하세요.")
            return 1

        validation_path = resolve_path(
            json.loads(meta_path.read_text(encoding="utf-8"))["dataset"]
        )

    section("score fusion diagnostic")

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"출력 폴더가 비어 있지 않습니다. 덮어쓰지 않습니다: {out_dir}")
        return 1

    for path, label in (
        (direct_path, "direct score parquet"),
        (generation_path, "generation score parquet"),
        (validation_path, "원본 validation parquet"),
    ):
        if not path.exists():
            print(f"{label}가 없습니다: {path}")
            return 1

    print(f"  direct     : {direct_path}")
    print(f"  generation : {generation_path}")
    print(f"  원본       : {validation_path}")
    print(f"  출력       : {out_dir}")
    print()
    print("  재학습하지 않습니다. 추론하지 않습니다. Test를 쓰지 않습니다.")

    step(f"direct 읽는 중: {direct_path.name}")
    direct = pd.read_parquet(direct_path)
    print(f"      row {len(direct):,}개")

    step(f"generation 읽는 중: {generation_path.name}")
    generation = pd.read_parquet(generation_path)
    print(f"      row {len(generation):,}개")

    for column, frame, name in (
        ("direct_score", direct, "direct"),
        ("L1", generation, "generation"),
    ):
        if column not in frame.columns:
            print(f"{name} parquet에 {column}이 없습니다.")
            return 1

    # 원본 validation parquet에서 기사 ID를 가져온다.
    # 예측 파일이 아니라 모델에 실제로 들어간 데이터다.
    section("원본 읽기")

    source = load_candidate_source(validation_path)

    print()
    print(f"  후보 {len(source):,}개 "
          f"(sample {len(source) // NUM_CANDIDATES:,})")

    # generation은 candidate_scores.parquet에 기사 ID를 남기지 않지만,
    # 같은 폴더의 raw 예측에는 dataloader가 준 값이 그대로 있다.
    # 있으면 붙여서 generation도 direct와 같은 교차 확인을 받게 한다.
    generation = attach_own_article_id(generation, priority0_dir)

    direct, direct_source_check = attach_source(direct, source, "direct")
    generation, generation_source_check = attach_source(
        generation, source, "generation"
    )

    merged, alignment = align(direct, generation)

    alignment["source_path"] = str(validation_path)
    alignment["direct_vs_source"] = direct_source_check
    alignment["generation_vs_source"] = generation_source_check

    num_impressions = len(merged) // NUM_CANDIDATES

    # ---- z-normalize
    section("z-normalize")

    d_values = merged["D"].to_numpy()
    g_values = merged["G"].to_numpy()

    zd, degenerate_d = z_normalize(d_values, num_impressions)
    zg, degenerate_g = z_normalize(g_values, num_impressions)

    merged["zD"] = zd
    merged["zG"] = zg

    print(f"  impression 안의 후보 5개끼리 정규화했습니다. eps = {EPS:g}")
    print()
    print(f"  {'':<22} {'평균':>10} {'표준편차':>10} {'최소':>10} {'최대':>10}")
    print("  " + "-" * 66)

    for label, values in (
        ("D (raw)", d_values), ("G (raw)", g_values),
        ("zD", zd), ("zG", zg),
    ):
        print(f"  {label:<22} {values.mean():>10.4f} {values.std():>10.4f} "
              f"{values.min():>10.4f} {values.max():>10.4f}")

    print()
    print(f"  후보 5개 점수가 사실상 같은 impression")
    print(f"    D : {degenerate_d:,} / {num_impressions:,}")
    print(f"    G : {degenerate_g:,} / {num_impressions:,}")

    if degenerate_d or degenerate_g:
        print("    이런 impression은 z가 전부 0에 가까워 무작위와 같습니다.")

    # ---- lambda sweep
    section(f"{len(LAMBDAS)}개 lambda 평가")

    ordered = merged[[GROUP_COLUMN, "label"]]
    labels = merged["label"].to_numpy()

    rows: List[Dict[str, Any]] = []

    for value in LAMBDAS:
        scores = value * zd + (1.0 - value) * zg
        metrics = evaluate_scores(ordered, scores)

        rows.append({"lambda": value, **metrics})

        print(f"  lambda {value:>4.1f}  Top-1 {metrics['top1_accuracy'] * 100:6.3f}%  "
              f"MRR {metrics['mrr']:.4f}  AUC {metrics['auc']:.4f}")

    # ---- 끝점 검증
    section("끝점 검증  (입력 파일에서 직접 다시 계산한 raw 기준)")

    print("  z-normalize는 impression 안의 affine 변환이라 순위를 바꾸지")
    print("  않아야 합니다. 같은 입력 파일에서 raw로 계산한 값과 z를 거친")
    print("  값을 대조합니다. 예전 실행 수치는 참고로만 보여줍니다.")
    print()

    endpoint_checks = [
        check_endpoint(
            "lambda=0  (G, generation)", g_values, zg,
            ordered, num_impressions, G_PAST_REFERENCE,
        ),
        check_endpoint(
            "lambda=1  (D, direct)", d_values, zd,
            ordered, num_impressions, D_PAST_REFERENCE,
        ),
    ]

    for item, value in zip(endpoint_checks, (0.0, 1.0)):
        row = next(r for r in rows if r["lambda"] == value)
        delta = abs(row["top1_accuracy"] - item["z"]["top1_accuracy"])
        item["sweep_delta"] = delta

        if delta > REPRODUCE_TOLERANCE:
            print(f"  !! sweep의 lambda={value:g}이 끝점 계산과 다릅니다 "
                  f"({delta:.3e})")

    endpoints_ok = all(item["passed"] for item in endpoint_checks)

    if endpoints_ok:
        print("  => 두 끝점 모두 raw와 순위, metric이 같습니다.")
    else:
        print("  => 끝점이 raw와 다릅니다. 아래 결과를 해석하기 전에")
        print("     원인을 확인하세요.")

    # ---- 정답 분포
    d_correct = correct_mask(d_values, labels, num_impressions)
    g_correct = correct_mask(g_values, labels, num_impressions)

    best = max(rows, key=lambda r: r["top1_accuracy"])
    g_row = next(r for r in rows if r["lambda"] == 0.0)

    g_wrong_count = int((~g_correct).sum())

    agreement = {
        "total": num_impressions,
        "both_correct": int((g_correct & d_correct).sum()),
        "g_only": int((g_correct & ~d_correct).sum()),
        "d_only": int((~g_correct & d_correct).sum()),
        "both_wrong": int((~g_correct & ~d_correct).sum()),
        "g_top1": float(g_correct.mean()),
        "d_top1": float(d_correct.mean()),
        "union": float((g_correct | d_correct).mean()),
        "best_lambda": best["lambda"],
        "best_hybrid_top1": best["top1_accuracy"],
        "best_hybrid_minus_g": (
            best["top1_accuracy"] - g_row["top1_accuracy"]
        ),
    }

    agreement["d_only_of_total"] = agreement["d_only"] / num_impressions
    agreement["d_only_of_g_wrong"] = (
        agreement["d_only"] / g_wrong_count if g_wrong_count else float("nan")
    )

    print_agreement(agreement)

    # ---- 상관
    d_grid = d_values.reshape(num_impressions, NUM_CANDIDATES)
    g_grid = g_values.reshape(num_impressions, NUM_CANDIDATES)

    def per_impression(method: str) -> float:
        values = []

        for i in range(num_impressions):
            a = pd.Series(d_grid[i])
            b = pd.Series(g_grid[i])

            if a.std(ddof=0) <= EPS or b.std(ddof=0) <= EPS:
                continue

            values.append(
                spearman(a, b) if method == "spearman"
                else a.corr(b, method="pearson")
            )

        return float(np.nanmean(values)) if values else float("nan")

    d_series = pd.Series(d_values)
    g_series = pd.Series(g_values)
    zd_series = pd.Series(zd)
    zg_series = pd.Series(zg)

    correlation = {
        "pearson_raw": float(d_series.corr(g_series, method="pearson")),
        "spearman_raw": spearman(d_series, g_series),
        "pearson_z": float(zd_series.corr(zg_series, method="pearson")),
        "spearman_z": spearman(zd_series, zg_series),
        "pearson_per_impression": per_impression("pearson"),
        "spearman_per_impression": per_impression("spearman"),
    }

    print_correlation(correlation)

    print_lambda_table(rows)
    print_comparison(rows)

    # ---- 저장
    section("저장")

    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "lambda_sweep.csv", rows)

    keep = [
        "impression_id", GROUP_COLUMN, "candidate_index", "label",
        "c1", "c2", "c3", "D", "G", "zD", "zG", "L1", "L2", "L3",
    ]

    if "candidate_article_id" in merged.columns:
        keep.insert(3, "candidate_article_id")

    merged[keep].to_parquet(
        out_dir / "fused_scores.parquet", index=False
    )

    metadata = {
        "experiment": "score_fusion_diagnostic",
        "direct_path": str(direct_path),
        "generation_path": str(generation_path),
        "validation_path": str(validation_path),
        "g_weights": list(G_WEIGHTS),
        "g_formula": (
            f"L1 + {G_WEIGHTS[1]:g}*L2 + {G_WEIGHTS[2]:g}*L3"
        ),
        "hybrid_formula": "lambda * zD + (1 - lambda) * zG",
        "normalization": "per-impression z over 5 candidates",
        "eps": EPS,
        "lambdas": LAMBDAS,
        "num_impressions": num_impressions,
        "num_rows": int(len(merged)),
        "degenerate_impressions": {"D": degenerate_d, "G": degenerate_g},
        "alignment": alignment,
        "endpoint_checks": endpoint_checks,
        "endpoints_passed": endpoints_ok,
        "lambda_results": rows,
        "agreement": agreement,
        "correlation": correlation,
        "best": best,
        "caveat": CAVEAT_EN,
        "did_not_do": [
            "P2 재학습, 추론",
            "dropout / weight decay 재탐색, LR 변경",
            "hidden weight, beta4 변경",
            "normalization layer, MLP, candidate c4",
            "level-wise direct scorer (D1/D2/D3)",
            "Test 평가",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 1),
        "git_commit": git_commit_hash(),
        "python": sys.version.split()[0],
    }

    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    write_readme(
        out_dir / "README.md", rows, agreement, correlation,
        alignment, endpoint_checks, best, metadata,
    )

    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {out_dir / name}")

    print()
    print(f"  총 소요 {time.time() - started_at:.0f}초")

    return 0


from analysis.live_output import enable_line_buffering

enable_line_buffering()

if __name__ == "__main__":
    raise SystemExit(main())
