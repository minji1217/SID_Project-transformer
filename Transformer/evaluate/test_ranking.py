"""
evaluate/ranking.py(학습 루프용 벡터화 구현)가
evaluate/metrics.py(테스트용 기준 구현)와 같은 값을 내는지 검증한다.

실행:
    cd Transformer
    python -m evaluate.test_ranking
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from evaluate.metrics import evaluate_ranking
from evaluate.ranking import compute_ranking_metrics


NUM_CANDIDATES = 5


def make_case(
    num_impressions: int,
    seed: int,
    tie_probability: float,
):
    # 동점이 섞인 랜덤 candidate score를 만든다.
    # tie_probability가 높을수록 동점 후보가 많이 생겨
    # 동점 처리 로직까지 비교할 수 있다.
    rng = np.random.default_rng(seed)

    scores = rng.normal(
        loc=0.0,
        scale=3.0,
        size=(num_impressions, NUM_CANDIDATES),
    )

    # 일부 값을 정수로 반올림해 인위적으로 동점을 만든다
    tie_mask = rng.random(scores.shape) < tie_probability
    scores[tie_mask] = np.round(scores[tie_mask])

    labels = np.zeros(
        (num_impressions, NUM_CANDIDATES),
        dtype=np.int64,
    )
    positive_index = rng.integers(
        low=0,
        high=NUM_CANDIDATES,
        size=num_impressions,
    )
    labels[np.arange(num_impressions), positive_index] = 1

    return scores, labels


def reference_metrics(scores: np.ndarray, labels: np.ndarray):
    # metrics.py가 기대하는 long format DataFrame으로 변환
    num_impressions = scores.shape[0]

    rows = {
        "sample_index": np.repeat(
            np.arange(num_impressions),
            NUM_CANDIDATES,
        ),
        "label": labels.reshape(-1),
        "candidate_score": scores.reshape(-1),
    }

    return evaluate_ranking(
        df=pd.DataFrame(rows),
        group_column="sample_index",
    )


def vectorized_metrics(scores: np.ndarray, labels: np.ndarray):
    output = compute_ranking_metrics(
        candidate_scores=torch.from_numpy(scores),
        candidate_labels=torch.from_numpy(labels),
        k=5,
    )

    return {
        "top1_accuracy": float(output.top1.mean().item()),
        "auc": float(output.auc.mean().item()),
        "mrr": float(output.mrr.mean().item()),
        "ndcg@5": float(output.ndcg.mean().item()),
    }


def main() -> int:
    cases = [
        ("동점 없음", 2000, 0, 0.0),
        ("동점 약간", 2000, 1, 0.15),
        ("동점 많음", 2000, 2, 0.60),
        ("동점 극단", 500, 3, 0.95),
    ]

    # float32 누적 오차를 감안한 허용 오차
    tolerance = 1e-6
    failed = False

    print("=" * 62)
    print("ranking.py  vs  metrics.py  일치 검증")
    print("=" * 62)

    for name, num_impressions, seed, tie_probability in cases:
        scores, labels = make_case(
            num_impressions=num_impressions,
            seed=seed,
            tie_probability=tie_probability,
        )

        reference = reference_metrics(scores, labels)
        vectorized = vectorized_metrics(scores, labels)

        print()
        print(f"[{name}]  impressions={num_impressions}")

        for key in ("top1_accuracy", "auc", "mrr", "ndcg@5"):
            difference = abs(reference[key] - vectorized[key])
            ok = difference <= tolerance

            if not ok:
                failed = True

            print(
                f"  {key:<14} "
                f"metrics.py={reference[key]:.10f}  "
                f"ranking.py={vectorized[key]:.10f}  "
                f"diff={difference:.2e}  "
                f"{'OK' if ok else 'MISMATCH'}"
            )

    print()
    print("=" * 62)

    if failed:
        print("FAILED: 두 구현의 값이 다릅니다.")
        return 1

    print("PASSED: 모든 경우에서 값이 일치합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
