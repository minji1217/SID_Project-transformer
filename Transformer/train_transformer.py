from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gin
import numpy as np
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.sequence import NewsSequenceDataset, collate_news_sequences
from evaluate.ranking import compute_ranking_metrics
from modules.model import NewsEncoderDecoderTransformer
from modules.loss import TransformerLoss

try:
    # run_summary.json에 기준 config의 지문과 적용된 설정을 남기기 위해 쓴다.
    # sweep 폴더가 없어도 학습 자체는 동작해야 하므로 실패를 허용한다.
    from sweep.hashing import file_sha256, parse_gin_bindings
except ImportError:
    file_sha256 = None
    parse_gin_bindings = None


BASE_DIR = Path(__file__).resolve().parent

# main()이 채우고 train()이 run_summary.json에 그대로 넣는다.
# 어떤 config와 어떤 override로 돌린 결과인지 파일만 보고 알 수 있게 한다.
RUN_CONTEXT: Dict[str, Any] = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str) -> Path:
    # 상대경로를 Transformer 프로젝트 기준 절대경로로 변환
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp(
    use_amp: bool,
    amp_dtype: str,
    device: torch.device,
) -> Tuple[bool, Optional[torch.dtype]]:
    # AMP(Automatic Mixed Precision) 사용 여부와 실제 dtype을 결정
    #
    # AMP를 켜면 forward/backward 계산을 16bit로 수행해
    # Ampere 이상 GPU에서 1.5~2배 빨라진다.
    # log_softmax / cross_entropy는 PyTorch가 자동으로 fp32로
    # 처리하므로 candidate score 계산의 수치 안정성은 유지된다.
    if not use_amp:
        return False, None

    if device.type != "cuda":
        print("[AMP] CUDA device가 아니므로 AMP를 사용하지 않습니다.")
        return False, None

    dtype_table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }

    key = str(amp_dtype).strip().lower()

    if key not in dtype_table:
        raise ValueError(
            "amp_dtype must be one of "
            f"{sorted(set(dtype_table))}. "
            f"Received: {amp_dtype}"
        )

    dtype = dtype_table[key]

    # bfloat16 미지원 GPU(T4 등)에서는 float16으로 자동 전환
    if (
        dtype is torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        print(
            "[AMP] 이 GPU는 bfloat16을 지원하지 않아 "
            "float16으로 전환합니다."
        )
        dtype = torch.float16

    return True, dtype


def create_grad_scaler(amp_dtype: Optional[torch.dtype]):
    # GradScaler는 float16에서만 필요하다.
    # bfloat16은 표현 범위가 fp32와 같아 scaling이 필요 없다.
    if amp_dtype is not torch.float16:
        return None

    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def create_metric_dict() -> Dict[str, float]:
    # 한 epoch 동안 누적할 학습/평가 통계
    return {
        "preference_loss_sum": 0.0,
        "positive_score_sum": 0.0,
        "negative_score_sum": 0.0,
        "positive_prob_sum": 0.0,
        "negative_prob_sum": 0.0,
        "auc_sum": 0.0,
        "mrr_sum": 0.0,
        "ndcg5_sum": 0.0,
        "top1_correct": 0,
        "num_impressions": 0,
        "num_positive_candidates": 0,
        "num_negative_candidates": 0,
    }


@torch.no_grad()
def update_metrics(
    metrics: Dict[str, float],
    loss_output,
    model_output,
    candidate_labels: torch.Tensor,
) -> None:
    # 한 batch의 loss, score, probability, Top-1 결과 누적
    candidate_labels = candidate_labels.float()
    candidate_scores = model_output.candidate_scores
    batch_size = candidate_scores.shape[0]

    positive_mask = candidate_labels == 1
    negative_mask = candidate_labels == 0

    num_positive = int(positive_mask.sum().item())
    num_negative = int(negative_mask.sum().item())

    # Loss
    metrics["preference_loss_sum"] += loss_output.preference_loss.item() * batch_size

    # Raw candidate score
    metrics["positive_score_sum"] += float(candidate_scores[positive_mask].sum().item())
    metrics["negative_score_sum"] += float(candidate_scores[negative_mask].sum().item())

    # 후보 5개 사이의 상대확률
    candidate_probs = torch.softmax(candidate_scores, dim=1)
    metrics["positive_prob_sum"] += float(candidate_probs[positive_mask].sum().item())
    metrics["negative_prob_sum"] += float(candidate_probs[negative_mask].sum().item())

    # Ranking metric (Top-1 / AUC / MRR / nDCG@5)
    # evaluate/ranking.py는 evaluate/metrics.py와 동일한 정의를 사용한다.
    # 동점 처리까지 일치하는지는 evaluate/test_ranking.py가 검증한다.
    ranking = compute_ranking_metrics(
        candidate_scores=candidate_scores,
        candidate_labels=candidate_labels,
        k=5,
    )

    metrics["top1_correct"] += int(ranking.top1.sum().item())
    metrics["auc_sum"] += float(ranking.auc.sum().item())
    metrics["mrr_sum"] += float(ranking.mrr.sum().item())
    metrics["ndcg5_sum"] += float(ranking.ndcg.sum().item())

    # Count
    metrics["num_impressions"] += batch_size
    metrics["num_positive_candidates"] += num_positive
    metrics["num_negative_candidates"] += num_negative


def finalize_metrics(
    metrics: Dict[str, float],
    loss_fn: TransformerLoss,
) -> Dict[str, float]:
    # epoch 누적값을 평균 metric으로 변환
    num_impressions = int(metrics["num_impressions"])
    num_positive = int(metrics["num_positive_candidates"])
    num_negative = int(metrics["num_negative_candidates"])

    if num_impressions == 0:
        raise ValueError("No impressions were processed.")

    preference_loss = metrics["preference_loss_sum"] / num_impressions
    total_loss = loss_fn.lambda_preference * preference_loss

    positive_score = (
        metrics["positive_score_sum"] / num_positive
        if num_positive > 0
        else float("nan")
    )

    negative_score = (
        metrics["negative_score_sum"] / num_negative
        if num_negative > 0
        else float("nan")
    )

    positive_prob = (
        metrics["positive_prob_sum"] / num_positive
        if num_positive > 0
        else float("nan")
    )

    negative_prob = (
        metrics["negative_prob_sum"] / num_negative
        if num_negative > 0
        else float("nan")
    )

    top1_accuracy = metrics["top1_correct"] / num_impressions

    auc = metrics["auc_sum"] / num_impressions
    mrr = metrics["mrr_sum"] / num_impressions
    ndcg_at_5 = metrics["ndcg5_sum"] / num_impressions

    # Positive-Negative score gap
    # 값 자체는 모델의 전체 확신도 수준에 따라 달라지므로
    # 서로 다른 config 사이의 직접 비교보다는 진단용으로 본다.
    score_gap = positive_score - negative_score

    return {
        "total_loss": total_loss,
        "preference_loss": preference_loss,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "score_gap": score_gap,
        "positive_prob": positive_prob,
        "negative_prob": negative_prob,
        "top1_accuracy": top1_accuracy,
        "auc": auc,
        "mrr": mrr,
        "ndcg@5": ndcg_at_5,
        "top1_correct": int(metrics["top1_correct"]),
        "num_impressions": num_impressions,
        "num_positive_candidates": num_positive,
        "num_negative_candidates": num_negative,
    }


def train_one_epoch(
    model: NewsEncoderDecoderTransformer,
    loss_fn: TransformerLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: Optional[float] = 1.0,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    scaler=None,
) -> Dict[str, float]:
    # training dataset 전체를 한 번 학습
    model.train()
    metrics = create_metric_dict()

    for batch in dataloader:
        # Batch → GPU
        history_sids = batch["history_sids"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
        candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

        # Gradient 초기화
        optimizer.zero_grad(set_to_none=True)

        # Forward
        # amp_enabled=False이면 autocast는 아무 일도 하지 않는다.
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            model_output = model(
                history_sids=history_sids,
                history_mask=history_mask,
                candidate_sids=candidate_sids,
            )

            # Loss
            loss_output = loss_fn(
                candidate_scores=model_output.candidate_scores,
                candidate_labels=candidate_labels,
            )

        if scaler is not None:
            # float16 AMP: gradient를 scaling해서 underflow를 막는다.
            scaler.scale(loss_output.total_loss).backward()

            # Gradient clipping 전에 scaling을 되돌려야
            # clip이 실제 gradient 크기 기준으로 동작한다.
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )

            # Parameter update
            scaler.step(optimizer)
            scaler.update()

        else:
            # fp32 또는 bfloat16 AMP
            # Backpropagation
            loss_output.total_loss.backward()

            # Gradient clipping
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )

            # Parameter update
            optimizer.step()

        # Metric update
        update_metrics(
            metrics=metrics,
            loss_output=loss_output,
            model_output=model_output,
            candidate_labels=candidate_labels,
        )

    return finalize_metrics(
        metrics=metrics,
        loss_fn=loss_fn,
    )


@torch.no_grad()
def evaluate(
    model: NewsEncoderDecoderTransformer,
    loss_fn: TransformerLoss,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    # validation dataset에서 loss와 Top-1 성능 계산
    model.eval()
    metrics = create_metric_dict()

    for batch in dataloader:
        # Batch → GPU
        history_sids = batch["history_sids"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
        candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

        # Forward
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            model_output = model(
                history_sids=history_sids,
                history_mask=history_mask,
                candidate_sids=candidate_sids,
            )

            # Loss
            loss_output = loss_fn(
                candidate_scores=model_output.candidate_scores,
                candidate_labels=candidate_labels,
            )

        # Metric
        update_metrics(
            metrics=metrics,
            loss_output=loss_output,
            model_output=model_output,
            candidate_labels=candidate_labels,
        )

    return finalize_metrics(
        metrics=metrics,
        loss_fn=loss_fn,
    )


def print_metrics(
    split_name: str,
    metrics: Dict[str, float],
) -> None:
    # train/validation 주요 metric 출력
    print(
        f"{split_name} Loss={metrics['total_loss']:.6f} | "
        f"Preference={metrics['preference_loss']:.6f}"
    )

    print(
        f"{split_name} Top-1 Accuracy={metrics['top1_accuracy']:.4%} | "
        f"Correct={int(metrics['top1_correct']):,}/"
        f"{int(metrics['num_impressions']):,}"
    )

    print(
        f"{split_name} Ranking | "
        f"MRR={metrics['mrr']:.6f} | "
        f"nDCG@5={metrics['ndcg@5']:.6f} | "
        f"AUC={metrics['auc']:.6f}"
    )

    print(
        f"{split_name} Score Gap={metrics['score_gap']:.4f}"
    )

    print(
        f"{split_name} Score | "
        f"Positive={metrics['positive_score']:.4f} | "
        f"Negative={metrics['negative_score']:.4f}"
    )

    print(
        f"{split_name} Probability | "
        f"Positive={metrics['positive_prob']:.6f} | "
        f"Negative={metrics['negative_prob']:.6f}"
    )

    print(
        f"{split_name} Candidates | "
        f"Positive={int(metrics['num_positive_candidates']):,} | "
        f"Negative={int(metrics['num_negative_candidates']):,}"
    )


def save_checkpoint(
    path: Path,
    model: NewsEncoderDecoderTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_top1_accuracy: float,
    include_optimizer: bool = True,
) -> None:
    # model/optimizer 상태와 gin 설정 저장
    #
    # optimizer 상태(AdamW의 moment 2개)는 모델 크기의 약 2배라
    # 파일 크기의 대부분을 차지한다.
    # 학습 재개가 필요 없는 탐색 단계에서는 제외할 수 있다.
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "validation_loss": validation_loss,
        "validation_top1_accuracy": validation_top1_accuracy,
        "gin_config": gin.config_str(),
    }

    if include_optimizer:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    torch.save(checkpoint, path)


def write_epoch_history(
    path: Path,
    epoch_records: list,
) -> None:
    # epoch마다의 train/validation 지표를 CSV로 남긴다.
    # 학습 곡선 확인과 best epoch 위치 판단에 사용한다.
    if not epoch_records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(epoch_records[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for record in epoch_records:
            writer.writerow(record)


def write_run_summary(
    path: Path,
    summary: Dict[str, Any],
) -> None:
    # 이 run의 설정과 best epoch 지표를 JSON으로 남긴다.
    # sweep이 이 파일만 읽어서 단계 요약 CSV를 만든다.
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


@gin.configurable
def train(
    train_path: str,
    validation_path: str,
    save_dir: str,
    batch_size: int = 128,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    gradient_clip_norm: Optional[float] = 1.0,
    num_workers: int = 0,
    prefetch_factor: int = 4,
    seed: int = 42,
    early_stopping_patience: Optional[int] = None,
    use_amp: bool = False,
    amp_dtype: str = "bfloat16",
    save_every_epoch: bool = True,
    save_optimizer_state: bool = True,
) -> None:
    # train/validation dataset을 이용해 Transformer 전체 학습
    set_seed(seed)
    device = get_device()

    print()
    print("Transformer Training")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # AMP
    amp_enabled, resolved_amp_dtype = resolve_amp(
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        device=device,
    )

    print(
        "AMP             :",
        (
            str(resolved_amp_dtype).replace("torch.", "")
            if amp_enabled
            else "disabled (float32)"
        ),
    )
    print("DataLoader workers:", num_workers)
    print(
        "Checkpoint      :",
        (
            "every epoch + best + final"
            if save_every_epoch
            else "best + final only"
        ),
        "|",
        (
            "with optimizer state"
            if save_optimizer_state
            else "model only"
        ),
    )

    # Path
    train_path = resolve_path(train_path)
    validation_path = resolve_path(validation_path)
    save_dir = resolve_path(save_dir)

    print("Train path      :", train_path)
    print("Validation path :", validation_path)
    print("Save directory  :", save_dir)

    # File check
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found:\n{train_path}")

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n{validation_path}"
        )

    # Dataset
    train_dataset = NewsSequenceDataset(
        parquet_path=str(train_path)
    )

    validation_dataset = NewsSequenceDataset(
        parquet_path=str(validation_path)
    )

    print("Train samples      :", f"{len(train_dataset):,}")
    print("Validation samples :", f"{len(validation_dataset):,}")

    # DataLoader
    # num_workers > 0일 때만 worker 관련 옵션을 줄 수 있다.
    # persistent_workers는 epoch마다 worker를 다시 만드는 비용을 없애고,
    # prefetch_factor는 worker가 미리 준비해 둘 batch 수를 정한다.
    dataloader_kwargs: Dict[str, Any] = {}

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
        **dataloader_kwargs,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
        **dataloader_kwargs,
    )

    # Model / Loss / Optimizer
    model = NewsEncoderDecoderTransformer().to(device)
    loss_fn = TransformerLoss().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scaler = create_grad_scaler(resolved_amp_dtype)

    # 이 run이 실제로 얼마나 GPU 메모리를 쓰는지 측정한다.
    # 성능이 같은 설정 중에서 고를 때의 기준이 되고,
    # 뒤 단계에서 batch를 키울 여유가 있는지 판단하는 근거가 된다.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Parameter count
    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters     :", f"{total_parameters:,}")
    print("Trainable parameters :", f"{trainable_parameters:,}")

    # Before training validation
    print()
    print("Before Training")

    initial_validation_metrics = evaluate(
        model=model,
        loss_fn=loss_fn,
        dataloader=validation_loader,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=resolved_amp_dtype,
    )

    print_metrics(
        "Validation",
        initial_validation_metrics,
    )

    # Best model tracking
    # checkpoint_best는 Validation Top-1 Accuracy 기준
    best_validation_top1 = -float("inf")
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # 결과 파일용 상태
    best_epoch = 0
    best_record: Optional[Dict[str, Any]] = None
    epoch_records: list = []
    stopped_early = False
    training_start_time = time.time()

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch = 0
    validation_loss = float("inf")
    validation_top1 = 0.0

    # Epoch loop
    for epoch in range(1, num_epochs + 1):
        print()
        print(f"Epoch {epoch}/{num_epochs}")

        epoch_start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
            amp_enabled=amp_enabled,
            amp_dtype=resolved_amp_dtype,
            scaler=scaler,
        )

        print_metrics(
            "Train",
            train_metrics,
        )

        # Validation
        validation_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            dataloader=validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=resolved_amp_dtype,
        )

        print_metrics(
            "Validation",
            validation_metrics,
        )

        validation_loss = validation_metrics["total_loss"]
        validation_top1 = validation_metrics["top1_accuracy"]

        # epoch 단위 기록
        record: Dict[str, Any] = {
            "epoch": epoch,
            "epoch_seconds": round(
                time.time() - epoch_start_time,
                2,
            ),
        }

        for key, value in validation_metrics.items():
            record[f"val_{key}"] = value

        for key, value in train_metrics.items():
            record[f"train_{key}"] = value

        epoch_records.append(record)

        # Every epoch checkpoint
        # 파라미터 탐색 중에는 save_every_epoch=False로 두어
        # epoch마다 쌓이는 checkpoint 용량을 없앤다.
        # checkpoint_best.pt와 checkpoint_final.pt는 항상 저장된다.
        if save_every_epoch:
            save_checkpoint(
                path=save_dir / f"checkpoint_epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_top1_accuracy=validation_top1,
                include_optimizer=save_optimizer_state,
            )

        # Best checkpoint
        # Validation Top-1 Accuracy가 가장 높은 epoch를 best로 선택
        if validation_top1 > best_validation_top1:
            best_validation_top1 = validation_top1
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            best_record = record

            save_checkpoint(
                path=save_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_top1_accuracy=validation_top1,
                include_optimizer=save_optimizer_state,
            )

            print(
                f"✓ Best checkpoint updated "
                f"(Validation Top-1={best_validation_top1:.4%}, "
                f"Loss={best_validation_loss:.6f})"
            )

        else:
            epochs_without_improvement += 1

            print(
                "Validation Top-1 did not improve."
            )

        # Early stopping
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping. "
                f"No validation Top-1 improvement for "
                f"{early_stopping_patience} epochs."
            )
            stopped_early = True
            break

    # Final checkpoint
    final_checkpoint_path = (
        save_dir / "checkpoint_final.pt"
    )

    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        validation_loss=validation_loss,
        validation_top1_accuracy=validation_top1,
        include_optimizer=save_optimizer_state,
    )

    # 결과 파일
    # best epoch는 Validation Top-1 Accuracy 기준으로 선택된 epoch이며,
    # 아래 지표들은 모두 그 epoch에서 측정된 값이다.
    epochs_run = len(epoch_records)

    history_path = save_dir / "epoch_history.csv"
    write_epoch_history(history_path, epoch_records)

    run_summary: Dict[str, Any] = {
        "status": "SUCCESS",
        "save_dir": str(save_dir),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "amp_enabled": amp_enabled,
        "amp_dtype": (
            str(resolved_amp_dtype).replace("torch.", "")
            if amp_enabled
            else None
        ),
        "seed": seed,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "num_epochs_configured": num_epochs,
        "num_epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "total_seconds": round(
            time.time() - training_start_time,
            2,
        ),
        "mean_epoch_seconds": (
            round(
                sum(r["epoch_seconds"] for r in epoch_records)
                / epochs_run,
                2,
            )
            if epochs_run > 0
            else None
        ),
        "selection_metric": "val_top1_accuracy",
        # max_memory_allocated는 tensor가 실제로 점유한 최댓값이고,
        # max_memory_reserved는 caching allocator가 확보한 총량이다.
        # OOM 여유를 볼 때는 reserved 쪽이 nvidia-smi 값에 가깝다.
        "peak_gpu_memory_mb": (
            round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                2,
            )
            if device.type == "cuda"
            else None
        ),
        "peak_gpu_memory_reserved_mb": (
            round(
                torch.cuda.max_memory_reserved(device) / (1024 ** 2),
                2,
            )
            if device.type == "cuda"
            else None
        ),
        "best_metrics": best_record or {},
        "base_config_path": RUN_CONTEXT.get("base_config_path"),
        "base_config_sha256": RUN_CONTEXT.get("base_config_sha256"),
        "config_hash": RUN_CONTEXT.get("config_hash"),
        "gin_overrides": RUN_CONTEXT.get("gin_overrides", []),
        "effective_bindings": (
            parse_gin_bindings(gin.config_str())
            if parse_gin_bindings is not None
            else None
        ),
        "gin_config": gin.config_str(),
    }

    # sweep이 넘겨준 추가 정보(stage 이름, 소스 지문 등)
    for key, value in RUN_CONTEXT.get("extra", {}).items():
        run_summary.setdefault(key, value)

    summary_path = save_dir / "run_summary.json"
    write_run_summary(summary_path, run_summary)

    print()
    print("Training Finished")
    print(
        "Best validation Top-1:",
        f"{best_validation_top1:.4%}",
    )
    print(
        "Loss at best checkpoint:",
        f"{best_validation_loss:.6f}",
    )
    print(
        "Best checkpoint:",
        save_dir / "checkpoint_best.pt",
    )
    print(
        "Final checkpoint:",
        final_checkpoint_path,
    )
    print(
        "Run summary:",
        summary_path,
    )
    print(
        "Epoch history:",
        history_path,
    )


def main() -> None:
    # --config으로 gin 파일을 받아 학습 시작
    parser = argparse.ArgumentParser(
        description=(
            "Train News Semantic-ID Encoder-Decoder Transformer"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    # gin 설정을 명령줄에서 덮어쓴다.
    # 예: --gin-binding "train.learning_rate = 0.0002"
    # 파라미터 탐색에서 config 파일을 복제하지 않고 값만 바꾸기 위해 사용한다.
    parser.add_argument(
        "--gin-binding",
        type=str,
        action="append",
        default=[],
        metavar="BINDING",
    )

    # sweep이 run_summary.json에 추가로 남기고 싶은 값을 담은 JSON 파일
    parser.add_argument(
        "--summary-extra",
        type=str,
        default=None,
        metavar="JSON_PATH",
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Gin config not found:\n{config_path}"
        )

    print(
        "Gin config:",
        config_path,
    )

    # run_summary.json에 남길 실행 맥락
    RUN_CONTEXT["base_config_path"] = str(config_path)
    RUN_CONTEXT["gin_overrides"] = list(args.gin_binding)

    if file_sha256 is not None:
        RUN_CONTEXT["base_config_sha256"] = file_sha256(config_path)

    if args.summary_extra:
        extra_path = resolve_path(args.summary_extra)

        if not extra_path.exists():
            raise FileNotFoundError(
                f"summary-extra file not found:\n{extra_path}"
            )

        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        RUN_CONTEXT["config_hash"] = extra.pop("config_hash", None)
        RUN_CONTEXT["extra"] = extra

    gin.parse_config_file(
        str(config_path)
    )

    # config 파일을 먼저 읽고, 그 뒤에 덮어쓴다.
    if args.gin_binding:
        print("Gin overrides:")

        for binding in args.gin_binding:
            print("  ", binding)

        gin.parse_config(args.gin_binding)

    train()


if __name__ == "__main__":
    main()