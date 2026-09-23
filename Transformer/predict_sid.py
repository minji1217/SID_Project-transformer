from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import gin
import pandas as pd
import torch

from torch.utils.data import DataLoader

from data.sequence import NewsSequenceDataset, collate_news_sequences
from modules.model import NewsEncoderDecoderTransformer


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_article_id(article_id):
    if article_id is None:
        return None

    try:
        if pd.isna(article_id):
            return None
    except (TypeError, ValueError):
        pass

    return str(article_id)


def load_checkpoint(checkpoint_path: Path, model: NewsEncoderDecoderTransformer, device: torch.device) -> dict:
    # 저장된 Transformer checkpoint 불러오기
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # train_transformer.py에서 저장한 checkpoint 형식
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        model.load_state_dict(checkpoint)

    return checkpoint


@torch.no_grad()
def predict(
    model: NewsEncoderDecoderTransformer,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    # 후보 5개의 score를 계산하고 Top-1 Accuracy와 후보별 결과 반환
    model.eval()

    prediction_rows = []

    # Metric 누적값
    num_impressions = 0
    num_correct = 0

    positive_score_sum = 0.0
    negative_score_sum = 0.0
    positive_prob_sum = 0.0
    negative_prob_sum = 0.0

    num_positive_candidates = 0
    num_negative_candidates = 0

    sample_index = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Input → device
        history_sids = batch["history_sids"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
        candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

        # Forward
        model_output = model(history_sids=history_sids, history_mask=history_mask, candidate_sids=candidate_sids)
        candidate_scores = model_output.candidate_scores

        # 후보 5개 사이의 상대 probability
        candidate_probs = torch.softmax(candidate_scores, dim=1)

        # Top-1 prediction
        predicted_indices = candidate_scores.argmax(dim=1)
        target_indices = candidate_labels.argmax(dim=1)
        correct_mask = predicted_indices == target_indices

        batch_size = candidate_sids.shape[0]
        num_candidates = candidate_sids.shape[1]

        # Ranking
        sorted_indices = torch.argsort(candidate_scores, dim=1, descending=True)
        ranks = torch.empty_like(sorted_indices)
        rank_values = torch.arange(1, num_candidates + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        ranks.scatter_(dim=1, index=sorted_indices, src=rank_values)

        # Metric 계산
        positive_mask = candidate_labels == 1
        negative_mask = candidate_labels == 0

        batch_num_positive = int(positive_mask.sum().item())
        batch_num_negative = int(negative_mask.sum().item())

        positive_score_sum += float(candidate_scores[positive_mask].sum().item())
        negative_score_sum += float(candidate_scores[negative_mask].sum().item())
        positive_prob_sum += float(candidate_probs[positive_mask].sum().item())
        negative_prob_sum += float(candidate_probs[negative_mask].sum().item())

        num_positive_candidates += batch_num_positive
        num_negative_candidates += batch_num_negative
        num_correct += int(correct_mask.sum().item())
        num_impressions += batch_size

        # 후보별 prediction 결과 저장
        for i in range(batch_size):
            impression_id = batch["impression_ids"][i]
            user_id = batch["user_ids"][i]
            impression_time = batch["impression_times"][i]
            article_ids = batch["candidate_article_ids"][i]

            if article_ids is None:
                article_ids = [None] * num_candidates

            if len(article_ids) != num_candidates:
                raise ValueError(
                    f"candidate_article_ids length mismatch at sample {sample_index}: "
                    f"{len(article_ids)} vs {num_candidates}"
                )

            is_correct = bool(correct_mask[i].item())

            for candidate_idx in range(num_candidates):
                sid = candidate_sids[i, candidate_idx].detach().cpu().tolist()
                article_id = normalize_article_id(article_ids[candidate_idx])

                prediction_rows.append(
                    {
                        "sample_index": sample_index,
                        "batch_index": batch_idx,
                        "impression_id": impression_id,
                        "user_id": user_id,
                        "impression_time": impression_time,
                        "article_id": article_id,
                        "c1": int(sid[0]),
                        "c2": int(sid[1]),
                        "c3": int(sid[2]),
                        "label": float(candidate_labels[i, candidate_idx].item()),
                        "candidate_score": float(candidate_scores[i, candidate_idx].item()),
                        "candidate_probability": float(candidate_probs[i, candidate_idx].item()),
                        "c1_log_prob": float(model_output.c1_log_probs[i, candidate_idx].item()),
                        "c2_log_prob": float(model_output.c2_log_probs[i, candidate_idx].item()),
                        "c3_log_prob": float(model_output.c3_log_probs[i, candidate_idx].item()),
                        "rank": int(ranks[i, candidate_idx].item()),
                        "is_top1": bool(predicted_indices[i].item() == candidate_idx),
                        "top1_correct": is_correct,
                    }
                )

            sample_index += 1

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            print(f"Processed batch {batch_idx + 1:,}/{num_batches:,}")

    # Final metrics
    top1_accuracy = num_correct / num_impressions if num_impressions > 0 else 0.0
    positive_score = positive_score_sum / num_positive_candidates if num_positive_candidates > 0 else float("nan")
    negative_score = negative_score_sum / num_negative_candidates if num_negative_candidates > 0 else float("nan")
    positive_probability = positive_prob_sum / num_positive_candidates if num_positive_candidates > 0 else float("nan")
    negative_probability = negative_prob_sum / num_negative_candidates if num_negative_candidates > 0 else float("nan")

    metrics = {
        "num_impressions": num_impressions,
        "num_correct": num_correct,
        "top1_accuracy": top1_accuracy,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
        "num_positive_candidates": num_positive_candidates,
        "num_negative_candidates": num_negative_candidates,
    }

    predictions_df = pd.DataFrame(prediction_rows)
    return predictions_df, metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    # Test dataset 최종 평가 결과 출력
    print()
    print("Test Evaluation")
    print("Impressions:", f"{int(metrics['num_impressions']):,}")
    print("Correct Top-1:", f"{int(metrics['num_correct']):,}")
    print("Top-1 Accuracy:", f"{metrics['top1_accuracy']:.4%}")

    print()
    print(f"Score | Positive={metrics['positive_score']:.4f} | Negative={metrics['negative_score']:.4f}")
    print(f"Probability | Positive={metrics['positive_probability']:.6f} | Negative={metrics['negative_probability']:.6f}")

    print()
    print(
        f"Candidates | Positive={int(metrics['num_positive_candidates']):,} | "
        f"Negative={int(metrics['num_negative_candidates']):,}"
    )


def main() -> None:
    # CLI argument
    parser = argparse.ArgumentParser(
        description="Evaluate 1-positive 4-negative candidate news ranking using Semantic-ID Transformer scores."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="out/transformer/ebnerd/checkpoint_best.pt")
    parser.add_argument("--test_path", type=str, default="datasets/ebnerd/test_sequences_1pos4neg.parquet")
    parser.add_argument("--output_path", type=str, default="out/transformer/ebnerd/test_candidate_scores.parquet")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    # 최종 설정으로 학습한 checkpoint를 불러오려면
    # 모델 구조가 학습 때와 같아야 한다.
    # 예: --gin-binding "NewsEncoderDecoderTransformer.d_model = 512"
    parser.add_argument(
        "--gin-binding",
        type=str,
        action="append",
        default=[],
        metavar="BINDING",
    )

    args = parser.parse_args()

    # Path
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    test_path = resolve_path(args.test_path)
    output_path = resolve_path(args.output_path)

    # File check
    if not config_path.exists():
        raise FileNotFoundError(f"Gin config not found:\n{config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found:\n{test_path}")

    # Gin
    gin.parse_config_file(str(config_path), skip_unknown=True)

    # config 파일을 먼저 읽고, 그 뒤에 덮어쓴다.
    if args.gin_binding:
        print("Gin overrides:")

        for binding in args.gin_binding:
            print("  ", binding)

        gin.parse_config(args.gin_binding, skip_unknown=True)

    # Device
    device = get_device()

    print()
    print("Candidate Ranking Evaluation")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Test path:", test_path)
    print("Checkpoint:", checkpoint_path)

    # Test Dataset
    test_dataset = NewsSequenceDataset(parquet_path=str(test_path))

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
    )

    print("Test samples:", f"{len(test_dataset):,}")

    # Model
    model = NewsEncoderDecoderTransformer().to(device)

    # Checkpoint
    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, model=model, device=device)
    print("Checkpoint loaded.")

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("Epoch:", checkpoint["epoch"])

    if isinstance(checkpoint, dict) and "validation_loss" in checkpoint:
        print("Checkpoint validation loss:", checkpoint["validation_loss"])

    if isinstance(checkpoint, dict) and "validation_top1_accuracy" in checkpoint:
        print("Checkpoint validation Top-1:", f"{checkpoint['validation_top1_accuracy']:.4%}")

    # Test prediction
    predictions_df, metrics = predict(model=model, dataloader=test_loader, device=device)

    # Save candidate predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(output_path, index=False)

    # Final result
    print_metrics(metrics)

    print()
    print("Candidate scores saved:", output_path)


if __name__ == "__main__":
    main()