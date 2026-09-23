"""
Top-1 Accuracy: 최종 1위가 클릭 기사인지 확인
AUC: positive가 negative보다 위에 오는 비율
MRR: 첫 positive가 얼마나 위에 있는지 확인
nDCG@5 / nDCG@10: top-K 전체 순위 품질
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_path(path: str) -> Path:
    # "~/..." 표기를 풀고, 상대경로는 Transformer 기준 절대경로로 변환
    path_obj = Path(path).expanduser()
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    # positive가 negative보다 높은 score를 갖는 비율
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return None

    comparisons = positive_scores[:, None] - negative_scores[None, :]
    wins = (comparisons > 0).sum()
    ties = (comparisons == 0).sum()
    total_pairs = len(positive_scores) * len(negative_scores)

    return float((wins + 0.5 * ties) / total_pairs)


def reciprocal_rank(labels: np.ndarray, ranks: np.ndarray) -> Optional[float]:
    # 첫 positive의 reciprocal rank 계산
    positive_ranks = ranks[labels == 1]

    if len(positive_ranks) == 0:
        return None

    first_positive_rank = int(positive_ranks.min())
    return float(1.0 / first_positive_rank)


def ndcg_at_k(labels: np.ndarray, ranks: np.ndarray, k: int) -> Optional[float]:
    # top-k 기준 nDCG 계산
    num_positive = int((labels == 1).sum())

    if num_positive == 0:
        return None

    dcg = 0.0

    for label, rank in zip(labels, ranks):
        if label == 1 and rank <= k:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_count = min(num_positive, k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))

    if idcg == 0:
        return None

    return float(dcg / idcg)


def top1_accuracy(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    # 가장 높은 score의 후보가 positive인지 확인
    if (labels == 1).sum() == 0:
        return None

    top_index = int(np.argmax(scores))
    return float(labels[top_index] == 1)


def make_ranks(scores: np.ndarray) -> np.ndarray:
    # score가 높을수록 작은 rank 부여
    sorted_indices = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[sorted_indices] = np.arange(1, len(scores) + 1)
    return ranks


def evaluate_ranking(df: pd.DataFrame, group_column: str) -> Dict[str, float]:
    # impression별 ranking metric 계산
    top1_values = []
    auc_values = []
    mrr_values = []
    ndcg5_values = []
    ndcg10_values = []

    num_impressions = 0

    for _, group in df.groupby(group_column, sort=False):
        labels = group["label"].to_numpy(dtype=np.int64)
        scores = group["candidate_score"].to_numpy(dtype=np.float64)

        if len(labels) != 5:
            raise ValueError(f"Each sample must contain exactly 5 candidates, but found {len(labels)}.")

        num_positive = int((labels == 1).sum())

        if num_positive != 1:
            raise ValueError(f"Each sample must contain exactly 1 positive candidate, but found {num_positive}.")

        ranks = make_ranks(scores)

        top1 = top1_accuracy(labels, scores)
        auc = auc_from_scores(labels, scores)
        mrr = reciprocal_rank(labels, ranks)
        ndcg5 = ndcg_at_k(labels, ranks, k=5)
        ndcg10 = ndcg_at_k(labels, ranks, k=10)

        num_impressions += 1

        if top1 is not None:
            top1_values.append(top1)

        if auc is not None:
            auc_values.append(auc)

        if mrr is not None:
            mrr_values.append(mrr)

        if ndcg5 is not None:
            ndcg5_values.append(ndcg5)

        if ndcg10 is not None:
            ndcg10_values.append(ndcg10)

    return {
        "num_impressions": num_impressions,
        "top1_accuracy": float(np.mean(top1_values)) if top1_values else float("nan"),
        "auc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "mrr": float(np.mean(mrr_values)) if mrr_values else float("nan"),
        "ndcg@5": float(np.mean(ndcg5_values)) if ndcg5_values else float("nan"),
        "ndcg@10": float(np.mean(ndcg10_values)) if ndcg10_values else float("nan"),
    }


def evaluate(prediction_path: Path) -> Dict[str, float]:
    # prediction parquet을 읽어 전체 ranking metric 계산
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found:\n{prediction_path}")

    df = pd.read_parquet(prediction_path)

    required_columns = ["label", "candidate_score"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "sample_index" in df.columns:
        group_column = "sample_index"
    elif "impression_id" in df.columns:
        group_column = "impression_id"
    else:
        raise ValueError("Prediction file must contain sample_index or impression_id.")

    df["label"] = df["label"].astype(int)
    return evaluate_ranking(df=df, group_column=group_column)


def print_metrics(metrics: Dict[str, float]) -> None:
    print()
    print("Test Ranking Performance")
    print(f"Impressions    : {metrics['num_impressions']:,}")
    print(f"Top-1 Accuracy : {metrics['top1_accuracy']:.6f}")
    print(f"AUC            : {metrics['auc']:.6f}")
    print(f"MRR            : {metrics['mrr']:.6f}")
    print(f"nDCG@5         : {metrics['ndcg@5']:.6f}")
    print(f"nDCG@10        : {metrics['ndcg@10']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Top-1 Accuracy, AUC, MRR, nDCG@5 and nDCG@10.")
    parser.add_argument("--prediction_path", type=str, default="out/transformer/ebnerd/test_candidate_scores.parquet")
    parser.add_argument("--output_path", type=str, default="out/transformer/ebnerd/test_metrics.json")
    args = parser.parse_args()

    prediction_path = resolve_path(args.prediction_path)
    output_path = resolve_path(args.output_path)

    metrics = evaluate(prediction_path=prediction_path)
    print_metrics(metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print()
    print("Metrics saved:", output_path)


if __name__ == "__main__":
    main()