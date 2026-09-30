"""Semantic ID가 추천에 쓸 만한 정보를 담고 있는지 진단한다.

Transformer의 성능이 낮을 때, 원인이 모델에 있는지 입력(SID)에 있는지
갈라내기 위한 스크립트다. 모델과 checkpoint를 전혀 쓰지 않고 데이터만 본다.
GPU도 쓰지 않는다.

    cd Transformer
    python -m analysis.diagnose_sid \
        --train datasets/ebnerd/train_sequences_1pos4neg.parquet \
        --validation datasets/ebnerd/validation_sequences_1pos4neg_half.parquet

Test 데이터는 쓰지 않는다.

확인하는 것
  1. codebook 사용률      선언된 vocab 대비 실제로 쓰이는 코드 수
  2. SID 해상도           엔트로피, 유효 SID 개수, 쏠림
  3. prefix 단계별 충돌   c1 / c1c2 / c1c2c3까지 정답과 오답이 같은 비율
  4. 레벨별 구분력        각 레벨까지만 봤을 때의 기대 Top-1
  5. 삼중항 다중도        같은 (c1,c2,c3)에 기사가 몇 개 뭉쳐 있는지
  6. history overlap      정답/오답이 history에 등장하는 비율
  7. trivial baseline     popularity / history overlap 만으로 낸 성능
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


CAND_LEVELS = ("candidate_c1", "candidate_c2", "candidate_c3")
HIST_LEVELS = ("history_c1", "history_c2", "history_c3", "history_c4")

# configs/transformer_ebnerd.gin에 선언된 값.
# 사용률(%)을 내기 위한 분모로만 쓴다.
DECLARED_VOCAB = {"c1": 25, "c2": 128, "c3": 512, "c4": 28}

# 자릿수 충돌을 피하기 위한 자리올림.
# c2 < 1000, c3 < 1000, c4 < 1000이므로 안전하다.
RADIX = 1000

# 후보 5개 중 정답 1개일 때의 무작위 기준값
RANDOM_TOP1 = 0.2
RANDOM_MRR = float(np.mean([1.0 / r for r in range(1, 6)]))
RANDOM_NDCG5 = float(np.mean([1.0 / np.log2(r + 1) for r in range(1, 6)]))
RANDOM_AUC = 0.5


class Tee:
    """화면과 파일에 같은 내용을 동시에 쓴다."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ----------------------------------------------------------------------
# 공통 변환
# ----------------------------------------------------------------------

def stack_candidates(df: pd.DataFrame) -> np.ndarray:
    # [N, 5, 3]
    levels = [np.stack(df[col].to_numpy()) for col in CAND_LEVELS]
    return np.stack(levels, axis=-1).astype(np.int64)


def labels_array(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["candidate_labels"].to_numpy()).astype(np.int64)


def prefix_keys(candidates: np.ndarray, depth: int) -> np.ndarray:
    """(c1), (c1,c2), (c1,c2,c3)를 각각 하나의 정수 key로 접는다."""
    key = candidates[..., 0].copy()

    for level in range(1, depth):
        key = key * RADIX + candidates[..., level]

    return key


def positive_index(labels: np.ndarray) -> np.ndarray:
    return labels.argmax(axis=1)


def score_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """후보 점수에서 Top-1 / MRR / nDCG@5 / AUC를 낸다.

    동점은 무작위로 갈린다고 보고 평균 순위로 처리한다.
    """
    rows = np.arange(len(scores))
    pos = positive_index(labels)
    positive_score = scores[rows, pos]

    greater = (scores > positive_score[:, None]).sum(axis=1)
    equal = (scores == positive_score[:, None]).sum(axis=1)  # 자기 자신 포함

    rank = greater + (equal + 1) / 2.0

    return {
        # 1등이 동점 k개면 1/k 확률로 맞는다
        "top1": float(np.mean((greater == 0) / equal)),
        "mrr": float(np.mean(1.0 / rank)),
        "ndcg@5": float(np.mean(1.0 / np.log2(rank + 1))),
        "auc": float(np.mean((4 - greater - (equal - 1) * 0.5) / 4.0)),
    }


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(
        f"  {name:<26} "
        f"Top-1 {metrics['top1']:.4f} | "
        f"MRR {metrics['mrr']:.4f} | "
        f"nDCG@5 {metrics['ndcg@5']:.4f} | "
        f"AUC {metrics['auc']:.4f}"
    )


# ----------------------------------------------------------------------
# 1. codebook 사용률
# ----------------------------------------------------------------------

def report_codebook_usage(df: pd.DataFrame, name: str) -> None:
    section(f"1. codebook 사용률  [{name}]")
    print("  선언된 vocab 대비 실제로 등장하는 코드 수.")
    print("  사용률이 낮으면 RQ-VAE codebook collapse다.")
    print()
    print(f"  {'column':<16}{'used':>8}{'vocab':>8}{'rate':>10}{'min':>7}{'max':>7}")
    print("  " + "-" * 56)

    for col in CAND_LEVELS + HIST_LEVELS:
        if col not in df.columns:
            continue

        arrays = [np.asarray(v) for v in df[col].to_numpy()]
        arrays = [a for a in arrays if a.size]

        if not arrays:
            continue

        values = np.concatenate(arrays)
        used = np.unique(values)

        level = col.split("_")[-1]
        declared = DECLARED_VOCAB.get(level)
        ratio = f"{len(used) / declared:.1%}" if declared else "-"

        print(
            f"  {col:<16}{len(used):>8}{declared if declared else '-':>8}"
            f"{ratio:>10}{used.min():>7}{used.max():>7}"
        )


# ----------------------------------------------------------------------
# 2. SID 해상도 / 정보량
# ----------------------------------------------------------------------

def report_information_content(df: pd.DataFrame, name: str) -> None:
    section(f"2. SID 해상도 / 정보량  [{name}]")

    candidates = stack_candidates(df)
    labels = labels_array(df)
    keys = prefix_keys(candidates, 3)

    rows = np.arange(len(keys))
    positive_keys = keys[rows, positive_index(labels)]

    counts = Counter(positive_keys.tolist())
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)
    probs = freqs / freqs.sum()

    entropy = float(-(probs * np.log2(probs)).sum())
    n_distinct = len(freqs)

    print(f"  전체 candidate 슬롯에 등장한 (c1,c2,c3) : "
          f"{len(np.unique(keys.reshape(-1))):,}개")
    print(f"  positive로 등장한 (c1,c2,c3)            : {n_distinct:,}개")
    print(f"  positive 총 개수                        : {len(positive_keys):,}개")
    print()
    print(f"  엔트로피       : {entropy:.2f} bits")
    print(f"  균등분포였다면 : {np.log2(n_distinct):.2f} bits")
    print(f"  유효 SID 개수  : {2 ** entropy:,.0f}개  (2^엔트로피)")
    print()
    print("  '유효 SID 개수'는 실제로 구분에 쓰이는 SID의 수다.")
    print("  기사 수보다 훨씬 작으면 SID가 그만큼 뭉뚱그려진 것이다.")
    print()

    cumulative = np.cumsum(probs)
    print("  상위 SID가 차지하는 positive 비율")
    for k in (10, 50, 100, 500, 1000, 5000):
        if k <= n_distinct:
            print(f"    상위 {k:>6}개 : {cumulative[k - 1]:.1%}")

    print()
    print("  단계별 엔트로피 (positive 기준)")
    positive_sids = candidates[rows, positive_index(labels)]

    for level, col in enumerate(CAND_LEVELS):
        values = positive_sids[:, level]
        level_counts = np.bincount(values)
        level_probs = level_counts[level_counts > 0] / len(values)
        level_entropy = float(-(level_probs * np.log2(level_probs)).sum())
        used = int((level_counts > 0).sum())

        print(
            f"    {col:<14} {level_entropy:>5.2f} bits | "
            f"사용 {used:>4}개 | 균등이면 {np.log2(used):>5.2f} bits"
        )


# ----------------------------------------------------------------------
# 3~4. prefix 단계별 충돌 + 레벨별 구분력
# ----------------------------------------------------------------------

def report_prefix_collisions(df: pd.DataFrame, name: str) -> None:
    section(f"3. prefix 단계별 정답-오답 충돌 + 레벨별 구분력  [{name}]")

    candidates = stack_candidates(df)
    labels = labels_array(df)
    rows = np.arange(len(candidates))
    pos = positive_index(labels)

    print("  정답과 같은 prefix를 가진 오답이 있으면")
    print("  모델은 그 레벨에서 둘을 구분할 수 없다.")
    print()
    print(f"  {'level':<12}{'collided':>14}{'rate':>10}{'expected Top-1':>18}")
    print("  " + "-" * 54)

    labels_by_depth = {1: "c1", 2: "c1c2", 3: "c1c2c3"}

    for depth in (1, 2, 3):
        keys = prefix_keys(candidates, depth)
        positive_key = keys[rows, pos]

        tie_counts = (keys == positive_key[:, None]).sum(axis=1)
        collided = int((tie_counts > 1).sum())
        total = len(keys)

        # 그 레벨까지만 보고 동점은 무작위로 갈랐을 때의 기대 정확도
        expected_top1 = float(np.mean(1.0 / tie_counts))

        print(
            f"  {labels_by_depth[depth]:<12}{collided:>14,}"
            f"{collided / total:>10.2%}{expected_top1:>18.4f}"
        )

    print()

    # c1c2c3 레벨의 동점 분포
    keys = prefix_keys(candidates, 3)
    positive_key = keys[rows, pos]
    tie_counts = (keys == positive_key[:, None]).sum(axis=1)
    total = len(keys)

    print("  c1c2c3 레벨의 동점 후보 수 분포")
    for k in range(1, 6):
        n = int((tie_counts == k).sum())
        if n:
            tag = "  (충돌 없음)" if k == 1 else ""
            print(f"    {k}개: {n:>9,}  ({n / total:>6.2%}){tag}")

    oracle = float(np.mean(1.0 / tie_counts))
    print()
    print(f"  >>> 이 데이터에서 가능한 Top-1 최대값 : {oracle:.4f}  ({oracle:.2%})")
    print("      완벽한 모델이 동점만 동전던지기로 갈랐을 때의 값이다.")


# ----------------------------------------------------------------------
# 5. 같은 (c1,c2,c3)에 기사가 몇 개 뭉쳐 있는가
# ----------------------------------------------------------------------

def report_triple_multiplicity(df: pd.DataFrame, name: str) -> None:
    section(f"5. 동일 (c1,c2,c3)에 매핑된 기사 수 분포  [{name}]")

    if not all(col in df.columns for col in HIST_LEVELS):
        print("  history에 c4가 없어 계산할 수 없습니다.")
        return

    print("  candidate는 (c1,c2,c3)만 쓰고 c4는 history에만 있다.")
    print("  history의 (c1,c2,c3,c4)를 모아 삼중항별 c4 개수를 세면,")
    print("  한 삼중항에 실제 기사가 몇 개 뭉쳐 있는지 알 수 있다.")
    print()

    parts = []
    for col in HIST_LEVELS:
        arrays = [np.asarray(v, dtype=np.int64) for v in df[col].to_numpy()]
        arrays = [a for a in arrays if a.size]
        parts.append(np.concatenate(arrays))

    c1, c2, c3, c4 = parts
    quad = ((c1 * RADIX + c2) * RADIX + c3) * RADIX + c4

    unique_quads = np.unique(quad)
    triples_of_quads = unique_quads // RADIX

    unique_triples, per_triple = np.unique(triples_of_quads, return_counts=True)

    print(f"  history에 등장한 기사 (c1,c2,c3,c4) : {len(unique_quads):,}개")
    print(f"  history에 등장한 삼중항 (c1,c2,c3)  : {len(unique_triples):,}개")
    print(f"  삼중항 하나당 평균 기사 수          : {per_triple.mean():.2f}개")
    print(f"  최대                                : {per_triple.max()}개")
    print()
    print("  삼중항당 기사 수 분포")

    dist = Counter(per_triple.tolist())
    for k in sorted(dist):
        if k <= 10 or k == max(dist):
            n = dist[k]
            print(f"    {k:>3}개 기사 : {n:>8,} 삼중항 ({n / len(unique_triples):>6.1%})")

    collapsed = int((per_triple > 1).sum())
    print()
    print(
        f"  기사 2개 이상이 같은 삼중항을 쓰는 경우 : "
        f"{collapsed:,} / {len(unique_triples):,}  "
        f"({collapsed / len(unique_triples):.1%})"
    )
    print("  이 비율이 높으면 candidate 쪽에 c4를 추가할 실익이 있다.")


# ----------------------------------------------------------------------
# 6. history overlap
# ----------------------------------------------------------------------

def build_history_key_sets(df: pd.DataFrame, depth: int) -> list:
    """행마다 history에 등장한 prefix key 집합을 만든다."""
    cols = [np.asarray(v, dtype=np.int64) for v in df["history_c1"].to_numpy()]

    if depth == 1:
        return [set(a.tolist()) for a in cols]

    c2s = [np.asarray(v, dtype=np.int64) for v in df["history_c2"].to_numpy()]

    if depth == 2:
        return [
            set((a * RADIX + b).tolist())
            for a, b in zip(cols, c2s)
        ]

    c3s = [np.asarray(v, dtype=np.int64) for v in df["history_c3"].to_numpy()]

    return [
        set(((a * RADIX + b) * RADIX + c).tolist())
        for a, b, c in zip(cols, c2s, c3s)
    ]


def history_overlap_counts(df: pd.DataFrame, depth: int) -> np.ndarray:
    """[N,5] - 각 후보의 prefix가 history에 몇 번 등장하는지."""
    candidates = stack_candidates(df)
    keys = prefix_keys(candidates, depth)

    cols = [np.asarray(v, dtype=np.int64) for v in df["history_c1"].to_numpy()]
    c2s = [np.asarray(v, dtype=np.int64) for v in df["history_c2"].to_numpy()]
    c3s = [np.asarray(v, dtype=np.int64) for v in df["history_c3"].to_numpy()]

    out = np.zeros(keys.shape, dtype=np.int64)

    for i in range(len(keys)):
        a = cols[i]

        if depth == 1:
            hist = a
        elif depth == 2:
            hist = a * RADIX + c2s[i]
        else:
            hist = (a * RADIX + c2s[i]) * RADIX + c3s[i]

        counter = Counter(hist.tolist())

        for j in range(keys.shape[1]):
            out[i, j] = counter.get(int(keys[i, j]), 0)

    return out


def report_history_overlap(df: pd.DataFrame, name: str) -> Dict[int, np.ndarray]:
    section(f"6. history - candidate SID overlap  [{name}]")

    print("  정답이 오답보다 history와 많이 겹친다면,")
    print("  모델이 배운 것은 사실상 '본 적 있는 SID 다시 고르기'일 수 있다.")
    print("  그 경우 Transformer 없이 단순 매칭으로도 비슷한 성능이 나온다.")
    print()

    labels = labels_array(df)
    rows = np.arange(len(df))
    pos = positive_index(labels)

    print(
        f"  {'level':<10}{'pos seen%':>12}{'neg seen%':>12}{'diff':>10}"
        f"{'pos count':>12}{'neg count':>12}"
    )
    print("  " + "-" * 70)
    print("  pos seen% = 정답 prefix가 history에 한 번이라도 등장한 비율")
    print("  pos count = 정답 prefix가 history에 등장한 평균 횟수")
    print("  " + "-" * 70)

    counts_by_depth: Dict[int, np.ndarray] = {}
    depth_names = {1: "c1", 2: "c1c2", 3: "c1c2c3"}

    for depth in (1, 2, 3):
        counts = history_overlap_counts(df, depth)
        counts_by_depth[depth] = counts

        positive_counts = counts[rows, pos]

        negative_mask = np.ones(counts.shape, dtype=bool)
        negative_mask[rows, pos] = False
        negative_counts = counts[negative_mask].reshape(len(df), 4)

        pos_rate = float((positive_counts > 0).mean())
        neg_rate = float((negative_counts > 0).mean())

        print(
            f"  {depth_names[depth]:<10}{pos_rate:>12.2%}{neg_rate:>12.2%}"
            f"{pos_rate - neg_rate:>+10.2%}"
            f"{positive_counts.mean():>12.3f}{negative_counts.mean():>12.3f}"
        )

    print()
    print("  diff가 크면 클수록 단순 반복 신호가 강하다는 뜻이다.")

    return counts_by_depth


# ----------------------------------------------------------------------
# 7. trivial baseline
# ----------------------------------------------------------------------

def report_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_overlap: Dict[int, np.ndarray],
) -> None:
    section("7. trivial baseline  (validation 기준)")

    print("  학습한 Transformer가 이 값들을 크게 넘지 못하면,")
    print("  history encoder가 사실상 일을 하지 않고 있다는 뜻이다.")
    print()

    val_labels = labels_array(val_df)

    # popularity: train에서 positive로 등장한 횟수
    train_candidates = stack_candidates(train_df)
    train_labels = labels_array(train_df)
    train_keys = prefix_keys(train_candidates, 3)
    train_rows = np.arange(len(train_keys))

    counter = Counter(train_keys[train_rows, positive_index(train_labels)].tolist())

    val_candidates = stack_candidates(val_df)
    val_keys = prefix_keys(val_candidates, 3)

    popularity = np.vectorize(lambda k: counter.get(int(k), 0))(val_keys)
    popularity = popularity.astype(np.float64)

    print_metrics("무작위", {
        "top1": RANDOM_TOP1, "mrr": RANDOM_MRR,
        "ndcg@5": RANDOM_NDCG5, "auc": RANDOM_AUC,
    })
    print_metrics("popularity", score_metrics(popularity, val_labels))

    for depth in (1, 2, 3):
        name = {1: "history overlap c1", 2: "history overlap c1c2",
                3: "history overlap c1c2c3"}[depth]
        print_metrics(name, score_metrics(
            val_overlap[depth].astype(np.float64), val_labels
        ))

    combined = val_overlap[3].astype(np.float64) * 1000.0 + popularity
    print_metrics("overlap c1c2c3 + pop", score_metrics(combined, val_labels))

    print()
    print("  참고: V1 Transformer의 validation 3-seed 평균")
    print("        Top-1 0.2659 | MRR 0.5167 | nDCG@5 0.6360 | AUC 0.5807")


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 0. 사용 가능한 컬럼
# ----------------------------------------------------------------------

ARTICLE_ID_COLUMNS = (
    "candidate_article_ids",
    "target_article_ids",
    "history_article_ids",
)


def report_columns(df: pd.DataFrame, name: str) -> None:
    section(f"0. parquet 컬럼  [{name}]")

    for col in df.columns:
        print(f"  {col}")

    print()
    found = [c for c in ARTICLE_ID_COLUMNS if c in df.columns]

    if found:
        print(f"  기사 ID 컬럼 발견: {', '.join(found)}")
    else:
        print("  기사 ID 컬럼이 없습니다.")
        print("  삼중항당 기사 수는 history의 (c1,c2,c3,c4)로 근사합니다.")


# ----------------------------------------------------------------------
# 1~2. 기사 ID 기준 삼중항 다중도
# ----------------------------------------------------------------------

def report_article_multiplicity(df: pd.DataFrame, name: str) -> None:
    section(f"1-2. 기사 ID 기준 (c1,c2,c3) 매핑 분포  [{name}]")

    if "candidate_article_ids" not in df.columns:
        print("  candidate_article_ids 컬럼이 없어 계산할 수 없습니다.")
        print("  섹션 5(history c4 기준)를 대신 보세요.")
        return

    candidates = stack_candidates(df)
    keys = prefix_keys(candidates, 3).reshape(-1)

    article_ids = np.concatenate(
        [np.asarray(v).reshape(-1) for v in df["candidate_article_ids"].to_numpy()]
    )

    if len(article_ids) != len(keys):
        print(f"  기사 ID 개수({len(article_ids):,})와 "
              f"candidate 슬롯 수({len(keys):,})가 다릅니다. 건너뜁니다.")
        return

    # 문자열 ID일 수 있으므로 정수 코드로 factorize
    codes, uniques = pd.factorize(article_ids)

    n_articles = len(uniques)
    n_triples = len(np.unique(keys))

    print(f"  고유 candidate 기사 ID : {n_articles:,}개")
    print(f"  고유 (c1,c2,c3)        : {n_triples:,}개")
    print(f"  압축비                 : {n_articles / max(n_triples, 1):.1f} 기사 / 삼중항")
    print()

    # 삼중항별 서로 다른 기사 수
    pairs = np.unique(np.stack([keys, codes], axis=1), axis=0)
    triples_of_pairs = pairs[:, 0]
    _, per_triple = np.unique(triples_of_pairs, return_counts=True)

    print(f"  삼중항당 기사 수   평균 {per_triple.mean():.2f} | "
          f"중앙값 {int(np.median(per_triple))} | "
          f"최대 {per_triple.max()} | 최소 {per_triple.min()}")
    print()
    print("  분위수")
    for q in (50, 75, 90, 95, 99):
        print(f"    {q:>2}% : {np.percentile(per_triple, q):.0f}개")

    print()
    print("  분포")
    dist = Counter(per_triple.tolist())
    shown = 0
    for k in sorted(dist):
        if shown < 12 or k == max(dist):
            n = dist[k]
            print(f"    {k:>4}개 기사 : {n:>7,} 삼중항 ({n / len(per_triple):>6.1%})")
            shown += 1

    collapsed = int((per_triple > 1).sum())
    print()
    print(f"  기사 2개 이상이 같은 삼중항 : {collapsed:,} / {len(per_triple):,} "
          f"({collapsed / len(per_triple):.1%})")
    print()
    print("  candidate는 (c1,c2,c3)만 쓰므로, 같은 삼중항의 기사들은")
    print("  모델이 원리적으로 구분할 수 없다. 압축비가 클수록 상한이 낮다.")


# ----------------------------------------------------------------------
# 7. popularity baseline 원인 규명
# ----------------------------------------------------------------------

def report_popularity_diagnosis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:
    section("7. popularity baseline 진단")

    print("  계산 방식")
    print("    1) train의 각 impression에서 positive의 (c1,c2,c3)를 센다")
    print("    2) validation의 후보 5개에 그 횟수를 점수로 매긴다")
    print("    3) 동점은 평균 순위로 처리한다")
    print()

    train_candidates = stack_candidates(train_df)
    train_labels = labels_array(train_df)
    train_keys = prefix_keys(train_candidates, 3)
    train_rows = np.arange(len(train_keys))
    train_pos_keys = train_keys[train_rows, positive_index(train_labels)]

    counter = Counter(train_pos_keys.tolist())

    val_candidates = stack_candidates(val_df)
    val_labels = labels_array(val_df)
    val_keys = prefix_keys(val_candidates, 3)
    val_rows = np.arange(len(val_keys))
    pos = positive_index(val_labels)

    scores = np.vectorize(lambda k: counter.get(int(k), 0))(val_keys).astype(np.float64)

    positive_scores = scores[val_rows, pos]
    negative_mask = np.ones(scores.shape, dtype=bool)
    negative_mask[val_rows, pos] = False
    negative_scores = scores[negative_mask].reshape(len(val_df), 4)

    print("  validation 후보의 popularity 점수")
    print(f"    정답 평균 {positive_scores.mean():>10.1f} | "
          f"중앙값 {np.median(positive_scores):>8.1f} | "
          f"0인 비율 {(positive_scores == 0).mean():>6.1%}")
    print(f"    오답 평균 {negative_scores.mean():>10.1f} | "
          f"중앙값 {np.median(negative_scores):>8.1f} | "
          f"0인 비율 {(negative_scores == 0).mean():>6.1%}")
    print()

    if negative_scores.mean() > positive_scores.mean():
        print("  >>> 오답이 정답보다 인기가 높습니다.")
        print("      negative가 노출 많은 기사에서 뽑혔다는 신호입니다.")
        print("      이 경우 popularity는 오답을 가리키므로 AUC가 0.5 아래로 갑니다.")
    else:
        print("  >>> 정답이 오답보다 인기가 높습니다.")

    print()

    # 동점 구조
    all_same = (scores == scores[:, :1]).all(axis=1)
    all_zero = (scores == 0).all(axis=1)
    print(f"  후보 5개 점수가 모두 같은 impression : "
          f"{all_same.mean():.1%}  (그중 모두 0인 경우 {all_zero.mean():.1%})")
    print("    이런 impression은 무작위와 같아 0.2 / AUC 0.5로 기여합니다.")
    print()

    # train에 없던 SID
    unseen = ~np.isin(val_keys[val_rows, pos], np.array(list(counter.keys())))
    print(f"  validation 정답 SID가 train positive에 없던 비율 : {unseen.mean():.1%}")
    print("    높으면 train/validation이 시간으로 갈렸고 SID 분포가 이동한 것입니다.")
    print()

    # 대안: 노출 기준 popularity
    exposure = Counter(train_keys.reshape(-1).tolist())
    exposure_scores = np.vectorize(
        lambda k: exposure.get(int(k), 0)
    )(val_keys).astype(np.float64)

    print("  다른 정의로 다시 계산")
    print_metrics("positive 빈도 기준", score_metrics(scores, val_labels))
    print_metrics("노출(전체 슬롯) 기준", score_metrics(exposure_scores, val_labels))
    print_metrics("부호 반전", score_metrics(-scores, val_labels))
    print()

    # metric 코드 자체의 sanity check
    rng = np.random.default_rng(0)
    random_scores = rng.random(scores.shape)
    shuffled = score_metrics(random_scores, val_labels)

    print("  metric 코드 sanity check (무작위 점수)")
    print_metrics("무작위 점수", shuffled)
    print(f"    기대값: Top-1 {RANDOM_TOP1:.4f} | MRR {RANDOM_MRR:.4f} | "
          f"nDCG@5 {RANDOM_NDCG5:.4f} | AUC {RANDOM_AUC:.4f}")
    print("    이 두 줄이 일치하면 metric 계산 자체는 정상입니다.")


def report_split_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """train과 validation이 같은 기사/SID를 쓰는지 본다.

    뉴스는 기사가 계속 바뀐다. validation 시점의 기사가 train에 없었다면
    모델은 기사를 외울 수 없고 SID의 의미 구조로만 일반화해야 한다.
    그 비율이 성능 상한을 좌우한다.
    """
    section("8. train - validation 집합 비교")

    def candidate_sets(df):
        candidates = stack_candidates(df)
        labels = labels_array(df)
        rows = np.arange(len(df))
        pos = positive_index(labels)

        keys = prefix_keys(candidates, 3)

        out = {
            "triple_all": set(keys.reshape(-1).tolist()),
            "triple_pos": set(keys[rows, pos].tolist()),
        }

        if "candidate_article_ids" in df.columns:
            ids = np.stack(df["candidate_article_ids"].to_numpy())
            out["article_all"] = set(ids.reshape(-1).tolist())
            out["article_pos"] = set(ids[rows, pos].tolist())

        return out

    train_sets = candidate_sets(train_df)
    val_sets = candidate_sets(val_df)

    rows_out = [
        ("candidate 전체 (c1,c2,c3)", "triple_all"),
        ("정답 (c1,c2,c3)", "triple_pos"),
        ("candidate 전체 기사 ID", "article_all"),
        ("정답 기사 ID", "article_pos"),
    ]

    print(f"  {'항목':<26}{'train':>10}{'val':>10}{'공통':>10}"
          f"{'val 중 신규':>14}")
    print("  " + "-" * 72)

    for label, key in rows_out:
        if key not in train_sets or key not in val_sets:
            continue

        a, b = train_sets[key], val_sets[key]
        shared = a & b
        new_in_val = len(b) - len(shared)

        print(
            f"  {label:<20}{len(a):>10,}{len(b):>10,}{len(shared):>10,}"
            f"{new_in_val / max(len(b), 1):>14.1%}"
        )

    print()

    # impression 단위로 보면 몇 %가 새 기사인가
    if "candidate_article_ids" in val_df.columns:
        val_candidates = stack_candidates(val_df)
        val_labels = labels_array(val_df)
        rows = np.arange(len(val_df))
        ids = np.stack(val_df["candidate_article_ids"].to_numpy())
        positive_ids = ids[rows, positive_index(val_labels)]

        train_articles = train_sets.get("article_all", set())
        unseen = np.array([int(i) not in train_articles for i in positive_ids])

        print(f"  validation impression 중 정답 기사가 train에 아예 없던 비율")
        print(f"    {unseen.mean():.1%}  ({int(unseen.sum()):,} / {len(unseen):,})")
        print()
        print("  이 값이 높으면 모델은 기사를 외울 수 없고")
        print("  SID의 의미 구조로만 새 기사를 맞혀야 한다.")


def run_all(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    report_columns(train_df, "train")

    report_codebook_usage(train_df, "train")
    report_codebook_usage(val_df, "validation")

    report_article_multiplicity(train_df, "train")
    report_article_multiplicity(val_df, "validation")

    report_information_content(train_df, "train")
    report_information_content(val_df, "validation")

    report_prefix_collisions(train_df, "train")
    report_prefix_collisions(val_df, "validation")

    report_triple_multiplicity(train_df, "train")
    report_triple_multiplicity(val_df, "validation")

    report_history_overlap(train_df, "train")
    val_overlap = report_history_overlap(val_df, "validation")

    report_baselines(train_df, val_df, val_overlap)
    report_popularity_diagnosis(train_df, val_df)
    report_split_overlap(train_df, val_df)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument(
        "--out",
        default="analysis_out/diagnose_sid.txt",
        help="결과를 저장할 파일 (기본 analysis_out/diagnose_sid.txt)",
    )
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

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    if args.sample:
        train_df = train_df.head(args.sample)
        val_df = val_df.head(args.sample)

    with out_path.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, handle)):
            print(f"train      : {train_path}  ({len(train_df):,} rows)")
            print(f"validation : {val_path}  ({len(val_df):,} rows)")
            print("Test 데이터는 사용하지 않습니다.")
            run_all(train_df, val_df)
            print()

    print(f"결과 저장: {out_path.resolve()}")
    return 0


from analysis.live_output import enable_line_buffering

enable_line_buffering()

if __name__ == "__main__":
    raise SystemExit(main())
