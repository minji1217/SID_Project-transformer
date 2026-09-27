"""Semantic ID 자체가 추천에 쓸 만한 정보를 담고 있는지 진단한다.

Transformer의 성능이 낮을 때, 원인이 모델에 있는지 입력(SID)에 있는지
먼저 갈라내기 위한 스크립트다. 모델을 전혀 쓰지 않고 데이터만 본다.

    cd Transformer
    python -m analysis.diagnose_sid \
        --train datasets/ebnerd/train_sequences_1pos4neg.parquet \
        --validation datasets/ebnerd/validation_sequences_1pos4neg_half.parquet

확인하는 것
  1. codebook 사용률
     c1/c2/c3/c4가 각각 몇 개의 코드를 실제로 쓰는지.
     선언된 vocab보다 훨씬 적게 쓰면 RQ-VAE codebook collapse다.

  2. candidate 식별 가능성
     candidate는 (c1,c2,c3)만 쓴다. c4는 history에만 있다.
     서로 다른 기사가 같은 (c1,c2,c3)를 가지면 모델이 원리적으로 구분할 수 없다.

  3. impression 내부 충돌률
     한 impression의 positive와 negative가 같은 (c1,c2,c3)면
     어떤 모델도 그 impression을 확실히 맞힐 수 없다.

  4. 이론적 상한 (oracle)
     3번의 충돌을 감안했을 때 Top-1 정확도의 최대값.

  5. popularity baseline
     history를 전혀 보지 않고 "train에서 자주 클릭된 SID"만으로 점수를 매긴다.
     학습한 모델이 이 값을 못 넘으면 history encoder가 일을 안 하고 있는 것이다.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


CAND_LEVELS = ("candidate_c1", "candidate_c2", "candidate_c3")
HIST_LEVELS = ("history_c1", "history_c2", "history_c3", "history_c4")

# 후보 5개 중 정답 1개일 때의 무작위 기준값
RANDOM_TOP1 = 0.2
RANDOM_MRR = float(np.mean([1.0 / r for r in range(1, 6)]))
RANDOM_NDCG5 = float(np.mean([1.0 / np.log2(r + 1) for r in range(1, 6)]))
RANDOM_AUC = 0.5


def stack_candidates(df: pd.DataFrame) -> np.ndarray:
    # [N, 5, 3] 정수 배열로 만든다.
    levels = [np.stack(df[col].to_numpy()) for col in CAND_LEVELS]
    return np.stack(levels, axis=-1).astype(np.int64)


def labels_array(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["candidate_labels"].to_numpy()).astype(np.int64)


def triple_keys(candidates: np.ndarray) -> np.ndarray:
    # (c1,c2,c3)를 하나의 정수 key로 접는다.
    c1, c2, c3 = candidates[..., 0], candidates[..., 1], candidates[..., 2]
    return (c1.astype(np.int64) * 1_000_000) + (c2.astype(np.int64) * 1_000) + c3


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def report_codebook_usage(df: pd.DataFrame, name: str) -> None:
    section(f"1. codebook 사용률  [{name}]")

    for col in CAND_LEVELS:
        values = np.concatenate(df[col].to_numpy())
        used = np.unique(values)
        print(
            f"  {col:<14} 사용 코드 {len(used):>5}개 | "
            f"최소 {used.min():>4} 최대 {used.max():>4}"
        )

    for col in HIST_LEVELS:
        if col not in df.columns:
            continue
        values = np.concatenate([np.asarray(v) for v in df[col].to_numpy() if len(v)])
        used = np.unique(values)
        print(
            f"  {col:<14} 사용 코드 {len(used):>5}개 | "
            f"최소 {used.min():>4} 최대 {used.max():>4}"
        )


def report_identifiability(df: pd.DataFrame, name: str) -> None:
    section(f"2. candidate 식별 가능성  [{name}]")

    candidates = stack_candidates(df)
    keys = triple_keys(candidates).reshape(-1)
    unique_triples = len(np.unique(keys))

    print(f"  등장한 서로 다른 (c1,c2,c3)  : {unique_triples:,}개")
    print(f"  전체 candidate 슬롯          : {len(keys):,}개")
    print()
    print("  (c1,c2,c3)가 같으면 모델의 점수가 수학적으로 같아진다.")
    print("  서로 다른 기사 수보다 이 값이 훨씬 작으면 그만큼 구분이 불가능하다.")


def report_collisions(df: pd.DataFrame, name: str) -> None:
    section(f"3. impression 내부 충돌 + 이론적 상한  [{name}]")

    candidates = stack_candidates(df)
    labels = labels_array(df)
    keys = triple_keys(candidates)

    positive_index = labels.argmax(axis=1)
    rows = np.arange(len(keys))
    positive_key = keys[rows, positive_index]

    # positive와 같은 key를 가진 후보 수 (자기 자신 포함)
    tie_counts = (keys == positive_key[:, None]).sum(axis=1)

    collided = int((tie_counts > 1).sum())
    total = len(keys)

    print(f"  positive와 같은 (c1,c2,c3)를 가진 negative가 있는 impression")
    print(f"    {collided:,} / {total:,}  ({collided / total:.2%})")
    print()

    for k in range(2, 6):
        n = int((tie_counts == k).sum())
        if n:
            print(f"    동점 후보 {k}개: {n:,} ({n / total:.2%})")

    # 동점은 무작위로 갈린다고 보면 기대 정확도는 1/tie_counts
    oracle_top1 = float(np.mean(1.0 / tie_counts))

    print()
    print(f"  이론적 Top-1 상한 (완벽한 모델 + 동점은 동전던지기)")
    print(f"    {oracle_top1:.4f}  ({oracle_top1:.2%})")


def report_popularity_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:
    section("4. popularity baseline (history를 전혀 안 보는 모델)")

    train_candidates = stack_candidates(train_df)
    train_labels = labels_array(train_df)
    train_keys = triple_keys(train_candidates)

    positive_index = train_labels.argmax(axis=1)
    rows = np.arange(len(train_keys))

    # train에서 각 SID가 positive로 등장한 횟수
    counter = Counter(train_keys[rows, positive_index].tolist())

    val_candidates = stack_candidates(val_df)
    val_labels = labels_array(val_df)
    val_keys = triple_keys(val_candidates)

    scores = np.vectorize(lambda k: counter.get(int(k), 0))(val_keys).astype(np.float64)

    val_positive_index = val_labels.argmax(axis=1)
    val_rows = np.arange(len(val_keys))
    positive_score = scores[val_rows, val_positive_index]

    greater = (scores > positive_score[:, None]).sum(axis=1)
    equal = (scores == positive_score[:, None]).sum(axis=1)  # 자기 자신 포함

    # 동점은 평균 순위로 처리
    rank = greater + (equal + 1) / 2.0

    top1 = float(np.mean(1.0 / equal * (greater == 0)))
    mrr = float(np.mean(1.0 / rank))
    ndcg = float(np.mean(1.0 / np.log2(rank + 1)))
    # 4개 negative 중 positive보다 낮은 것의 비율. 동점은 0.5.
    auc = float(np.mean((4 - greater - (equal - 1) * 0.5) / 4.0))

    print(f"  Top-1 Accuracy : {top1:.4f}  ({top1:.2%})")
    print(f"  MRR            : {mrr:.4f}")
    print(f"  nDCG@5         : {ndcg:.4f}")
    print(f"  AUC            : {auc:.4f}")
    print()
    print("  학습한 Transformer가 이 값을 크게 못 넘으면")
    print("  history encoder가 사실상 일을 하지 않고 있다는 뜻이다.")


def report_reference() -> None:
    section("5. 무작위 기준값 (후보 5개 중 정답 1개)")
    print(f"  Top-1 Accuracy : {RANDOM_TOP1:.4f}")
    print(f"  MRR            : {RANDOM_MRR:.4f}")
    print(f"  nDCG@5         : {RANDOM_NDCG5:.4f}")
    print(f"  AUC            : {RANDOM_AUC:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="앞에서 N행만 본다. 0이면 전체 (기본 0)",
    )
    args = parser.parse_args()

    train_path = Path(args.train).expanduser()
    val_path = Path(args.validation).expanduser()

    for path in (train_path, val_path):
        if not path.exists():
            print(f"파일을 찾을 수 없습니다: {path}")
            return 1

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    if args.sample:
        train_df = train_df.head(args.sample)
        val_df = val_df.head(args.sample)

    print(f"train      : {train_path}  ({len(train_df):,} rows)")
    print(f"validation : {val_path}  ({len(val_df):,} rows)")

    report_codebook_usage(train_df, "train")
    report_identifiability(train_df, "train")
    report_collisions(train_df, "train")
    report_collisions(val_df, "validation")
    report_popularity_baseline(train_df, val_df)
    report_reference()

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
