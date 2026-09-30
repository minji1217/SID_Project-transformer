"""c123 mean과 weighted [1,1,1]이 정말 같은 실행인지 확인한다.

hidden weight sweep의 A가 기존 c123 mean을 재현하지 못했다.
D를 best로 확정하기 전에 그 차이가 어디서 오는지부터 본다.

보는 것

  1. backbone embedding / encoder 초기값
  2. pool_proj / pool_score / candidate_projection / bilinear 초기값
  3. 첫 train batch의 impression_id / candidate order
  4. 첫 batch forward의 user vector
  5. candidate vector
  6. candidate score
  7. 첫 batch loss
  8. gradient
  9. 같은 모델을 두 번 돌렸을 때의 차이 (GPU 비결정성)
 10. optimizer step을 N번 돌렸을 때 언제 갈라지는가
 11. 이미 끝난 두 run의 epoch 곡선 비교

학습하지 않는다. checkpoint를 만들지 않는다.

    cd Transformer
    python -m analysis.compare_direct_runs \
        --mean-run sweep_out/ebnerd_v2/priority2_direct/bilinear_c123/seed42 \
        --weighted-run sweep_out/ebnerd_v2/priority2_direct/hidden_weight_sweep/A_1.00_1.00_1.00/seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gin
import numpy as np
import pandas as pd
import torch

from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.sequence import NewsSequenceDataset, collate_news_sequences
from modules.direct_scorer import BACKBONE_TRAINABLE_PREFIXES, DirectScorer
from modules.loss import TransformerLoss
from modules.model import NewsEncoderDecoderTransformer
from sweep.run_stage import format_binding
from train_transformer import set_seed


BASE_DIR = Path(__file__).resolve().parent.parent

# 이 값보다 작으면 같다고 본다.
EXACT = 0.0


def section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def resolve_path(path: str) -> Path:
    path_obj = Path(path).expanduser()
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verdict(deviation: float) -> str:
    return "동일" if deviation == EXACT else "다름"


def report(label: str, deviation: float, width: int = 46) -> bool:
    same = deviation == EXACT
    mark = "OK  " if same else "DIFF"
    print(f"  {mark}  {label:<{width}} 최대편차 {deviation:.6e}")
    return same


def max_dev(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).abs().max().item())


# ---------------------------------------------------------------- 환경


def report_determinism() -> Dict[str, Any]:
    section("0. 결정론 설정")

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch": torch.__version__,
    }

    try:
        info["deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
    except Exception:
        info["deterministic_algorithms"] = None

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["matmul_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        info["cudnn_tf32"] = bool(torch.backends.cudnn.allow_tf32)

    for key, value in info.items():
        print(f"  {key:<28} {value}")

    print()
    print("  train_transformer.set_seed()는 RNG만 시드하고")
    print("  cudnn.deterministic / use_deterministic_algorithms를 켜지 않습니다.")
    print("  GPU에서는 같은 설정을 두 번 돌려도 결과가 달라질 수 있습니다.")

    if info.get("matmul_tf32"):
        print()
        print("  TF32가 켜져 있습니다. matmul이 fp32보다 낮은 정밀도로 돌아가")
        print("  연산 순서 차이가 더 크게 드러납니다.")

    return info


# ---------------------------------------------------------------- 모델


def build_pair(
    config_path: Path,
    bindings: List[str],
    train_path: Path,
    validation_path: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> Tuple[DirectScorer, DirectScorer, Dict[str, Any], Dict[str, Any]]:
    """두 모델을 학습 스크립트와 같은 순서로 만든다.

    set_seed -> Dataset -> DataLoader -> Model
    """
    gin.parse_config_file(str(config_path), skip_unknown=True)

    if bindings:
        gin.parse_config(bindings, skip_unknown=True)

    def make(mode: str, weights: Tuple[float, float, float]):
        set_seed(seed)

        train_dataset = NewsSequenceDataset(parquet_path=str(train_path))
        validation_dataset = NewsSequenceDataset(parquet_path=str(validation_path))

        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_news_sequences,
            pin_memory=(device.type == "cuda"),
        )

        backbone = NewsEncoderDecoderTransformer().to(device)

        model = DirectScorer(
            backbone=backbone, scorer_type="bilinear", history_levels=3,
            history_mode=mode, level_weights=weights,
        ).to(device)

        batch = next(iter(loader))

        return model, batch

    model_mean, batch_mean = make("mean", (1.0, 1.0, 1.0))
    model_weighted, batch_weighted = make("weighted", (1.0, 1.0, 1.0))

    return model_mean, model_weighted, batch_mean, batch_weighted


def compare_parameters(
    a: DirectScorer, b: DirectScorer
) -> Tuple[bool, Dict[str, Any]]:
    section("1-2. 초기값 비교")

    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())

    if set(pa) != set(pb):
        only_a = sorted(set(pa) - set(pb))
        only_b = sorted(set(pb) - set(pa))
        print(f"  parameter 집합이 다릅니다.")
        print(f"    mean에만     : {only_a}")
        print(f"    weighted에만 : {only_b}")
        return False, {"same_keys": False}

    print(f"  parameter tensor {len(pa)}개, 집합 동일")
    print()

    groups = {
        "1. backbone embedding": lambda n: n.startswith((
            "backbone.c1_embedding", "backbone.c2_embedding",
            "backbone.c3_embedding", "backbone.c4_embedding",
        )),
        "1. backbone encoder": lambda n: n.startswith((
            "backbone.encoder.", "backbone.encoder_dummy_embedding",
        )),
        "2. pool_proj": lambda n: n.startswith("pool_proj"),
        "2. pool_score": lambda n: n.startswith("pool_score"),
        "2. candidate_projection": lambda n: n.startswith("candidate_projection"),
        "2. bilinear": lambda n: n.startswith("bilinear"),
    }

    details: Dict[str, float] = {}
    all_same = True

    for label, matches in groups.items():
        names = [n for n in pa if matches(n)]

        if not names:
            print(f"  ----  {label:<46} (해당 없음)")
            continue

        deviation = max(max_dev(pa[n], pb[n]) for n in names)
        details[label] = deviation
        all_same &= report(f"{label}  ({len(names)} tensor)", deviation)

    # 그 외 전부
    covered = {n for n in pa if any(m(n) for m in groups.values())}
    rest = sorted(set(pa) - covered)

    if rest:
        deviation = max(max_dev(pa[n], pb[n]) for n in rest)
        details["기타"] = deviation
        all_same &= report(f"기타  ({len(rest)} tensor)", deviation)

    return all_same, details


def compare_batch(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    section("3. 첫 train batch 비교")

    same = True
    details: Dict[str, Any] = {}

    ids_a = list(a["impression_ids"])
    ids_b = list(b["impression_ids"])

    match = ids_a == ids_b
    details["impression_ids"] = match
    same &= match

    print(f"  {'OK  ' if match else 'DIFF'}  impression_id 순서 "
          f"({len(ids_a)}개)")

    if not match:
        overlap = len(set(map(str, ids_a)) & set(map(str, ids_b)))
        print(f"        겹치는 impression {overlap}/{len(ids_a)}개")
        print(f"        mean     앞 5개: {[str(x) for x in ids_a[:5]]}")
        print(f"        weighted 앞 5개: {[str(x) for x in ids_b[:5]]}")

    for key in ("history_sids", "history_mask", "candidate_sids",
                "candidate_labels"):
        equal = torch.equal(a[key], b[key])
        details[key] = equal
        same &= equal
        print(f"  {'OK  ' if equal else 'DIFF'}  {key:<46} "
              f"shape {tuple(a[key].shape)}")

    return same, details


@torch.no_grad()
def compare_forward(
    model_a: DirectScorer, model_b: DirectScorer,
    batch: Dict[str, Any], device: torch.device,
) -> Tuple[bool, Dict[str, float]]:
    """같은 batch를 두 모델에 넣는다. 데이터 차이를 배제하고 경로만 본다."""
    section("4-7. 같은 batch를 두 모델에 넣었을 때")

    model_a.eval()
    model_b.eval()

    history_sids = batch["history_sids"].to(device)
    history_mask = batch["history_mask"].to(device)
    candidate_sids = batch["candidate_sids"].to(device)
    candidate_labels = batch["candidate_labels"].to(device)

    out_a = model_a(history_sids=history_sids, history_mask=history_mask,
                    candidate_sids=candidate_sids)
    out_b = model_b(history_sids=history_sids, history_mask=history_mask,
                    candidate_sids=candidate_sids)

    loss_fn = TransformerLoss().to(device)

    loss_a = loss_fn(candidate_scores=out_a.candidate_scores,
                     candidate_labels=candidate_labels)
    loss_b = loss_fn(candidate_scores=out_b.candidate_scores,
                     candidate_labels=candidate_labels)

    details = {
        "4. user vector": max_dev(out_a.user_vector, out_b.user_vector),
        "5. candidate vector": max_dev(
            out_a.candidate_vectors, out_b.candidate_vectors),
        "6. candidate score": max_dev(
            out_a.candidate_scores, out_b.candidate_scores),
        "7. loss": abs(
            float(loss_a.total_loss.item()) - float(loss_b.total_loss.item())),
    }

    same = True

    for label, deviation in details.items():
        same &= report(label, deviation)

    print()
    print(f"        mean     loss {float(loss_a.total_loss):.10f}")
    print(f"        weighted loss {float(loss_b.total_loss):.10f}")

    # Top-1이 갈리는 impression 수
    pred_a = out_a.candidate_scores.argmax(dim=1)
    pred_b = out_b.candidate_scores.argmax(dim=1)
    flipped = int((pred_a != pred_b).sum().item())

    print()
    print(f"        1등 예측이 바뀐 impression : {flipped} / "
          f"{pred_a.shape[0]}")

    details["flipped_top1"] = flipped

    return same, details


def compare_gradients(
    model_a: DirectScorer, model_b: DirectScorer,
    batch: Dict[str, Any], device: torch.device,
) -> Tuple[bool, float]:
    section("8. 같은 batch의 gradient")

    history_sids = batch["history_sids"].to(device)
    history_mask = batch["history_mask"].to(device)
    candidate_sids = batch["candidate_sids"].to(device)
    candidate_labels = batch["candidate_labels"].to(device)

    loss_fn = TransformerLoss().to(device)

    grads = []

    for model in (model_a, model_b):
        model.train()
        model.zero_grad(set_to_none=True)

        out = model(history_sids=history_sids, history_mask=history_mask,
                    candidate_sids=candidate_sids)
        loss = loss_fn(candidate_scores=out.candidate_scores,
                       candidate_labels=candidate_labels)
        loss.total_loss.backward()

        grads.append({
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        })

    ga, gb = grads

    if set(ga) != set(gb):
        print("  gradient가 있는 parameter 집합이 다릅니다.")
        return False, float("nan")

    deviation = max(max_dev(ga[n], gb[n]) for n in ga)
    same = report(f"gradient  ({len(ga)} tensor)", deviation)

    return same, deviation


def check_self_determinism(
    model: DirectScorer, batch: Dict[str, Any], device: torch.device,
    repeats: int,
) -> Dict[str, float]:
    """같은 모델, 같은 batch를 여러 번 돌려 본다.

    forward가 매번 같은 값을 내는지, backward는 어떤지 본다.
    여기서 차이가 나면 코드 경로와 무관하게 GPU가 비결정적인 것이다.
    """
    section(f"9. 같은 모델을 {repeats}번 돌렸을 때 (GPU 비결정성)")

    history_sids = batch["history_sids"].to(device)
    history_mask = batch["history_mask"].to(device)
    candidate_sids = batch["candidate_sids"].to(device)
    candidate_labels = batch["candidate_labels"].to(device)

    loss_fn = TransformerLoss().to(device)

    scores: List[Tensor] = []
    grads: List[Tensor] = []

    for _ in range(repeats):
        model.eval()

        with torch.no_grad():
            out = model(history_sids=history_sids, history_mask=history_mask,
                        candidate_sids=candidate_sids)
            scores.append(out.candidate_scores.detach().clone())

        model.train()
        model.zero_grad(set_to_none=True)

        out = model(history_sids=history_sids, history_mask=history_mask,
                    candidate_sids=candidate_sids)
        loss = loss_fn(candidate_scores=out.candidate_scores,
                       candidate_labels=candidate_labels)
        loss.total_loss.backward()

        grads.append(
            torch.cat([
                param.grad.detach().reshape(-1)
                for _, param in sorted(model.named_parameters())
                if param.grad is not None
            ]).clone()
        )

    forward_dev = max(max_dev(scores[0], s) for s in scores[1:])
    backward_dev = max(max_dev(grads[0], g) for g in grads[1:])

    report("forward (candidate score)", forward_dev)
    report("backward (gradient)", backward_dev)

    print()

    if forward_dev == 0.0 and backward_dev == 0.0:
        print("  같은 모델은 매번 같은 값을 냅니다.")
        print("  차이가 있다면 코드 경로 때문입니다.")
    else:
        print("  같은 모델인데도 값이 달라집니다.")
        print("  GPU 커널이 비결정적입니다. 두 run의 차이를 코드 경로 탓으로")
        print("  돌릴 수 없습니다. 재현 검증의 허용치를 이 크기 이상으로")
        print("  잡아야 합니다.")

    return {"forward": forward_dev, "backward": backward_dev}


def run_n_steps(
    mode: str,
    config_path: Path, bindings: List[str], train_path: Path,
    validation_path: Path, device: torch.device, seed: int,
    batch_size: int, learning_rate: float, steps: int,
) -> Tuple[List[float], List[Dict[str, Tensor]]]:
    """한 경로로 step을 N번 돌린다.

    두 모델을 만들어 두고 번갈아 batch를 뽑으면 안 된다. DataLoader의
    sampler가 generator라서 permutation이 iter() 시점이 아니라 첫
    next() 시점에 정해지기 때문이다. 그렇게 하면 두 모델이 서로 다른
    데이터를 받는다. 실제 학습은 프로세스가 분리되어 있으므로, 여기서도
    한 경로를 끝까지 돌린 뒤 다음 경로를 처음부터 돌린다.
    """
    set_seed(seed)

    train_dataset = NewsSequenceDataset(parquet_path=str(train_path))
    NewsSequenceDataset(parquet_path=str(validation_path))

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
    )

    backbone = NewsEncoderDecoderTransformer().to(device)
    model = DirectScorer(
        backbone=backbone, scorer_type="bilinear", history_levels=3,
        history_mode=mode, level_weights=(1.0, 1.0, 1.0),
    ).to(device)

    params = [
        p for _, p in
        model.backbone_trainable_parameters() + model.new_parameters()
    ]
    optimizer = AdamW(params, lr=learning_rate, weight_decay=0.0)

    loss_fn = TransformerLoss().to(device)

    losses: List[float] = []
    snapshots: List[Dict[str, Tensor]] = []

    loader_iter = iter(loader)

    for _ in range(steps):
        batch = next(loader_iter)
        model.train()

        out = model(
            history_sids=batch["history_sids"].to(device),
            history_mask=batch["history_mask"].to(device),
            candidate_sids=batch["candidate_sids"].to(device),
        )
        loss = loss_fn(
            candidate_scores=out.candidate_scores,
            candidate_labels=batch["candidate_labels"].to(device),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.total_loss.backward()
        optimizer.step()

        losses.append(float(loss.total_loss.item()))
        snapshots.append({
            name: param.detach().clone()
            for name, param in model.named_parameters()
        })

    return losses, snapshots


def divergence_over_steps(
    config_path: Path, bindings: List[str], train_path: Path,
    validation_path: Path, device: torch.device, seed: int,
    batch_size: int, learning_rate: float, steps: int,
) -> List[Dict[str, Any]]:
    """optimizer step을 돌리며 두 경로가 언제 갈라지는지 본다."""
    section(f"10. optimizer step {steps}번 동안의 벌어짐")

    losses_a, snaps_a = run_n_steps(
        "mean", config_path, bindings, train_path, validation_path,
        device, seed, batch_size, learning_rate, steps,
    )
    losses_b, snaps_b = run_n_steps(
        "weighted", config_path, bindings, train_path, validation_path,
        device, seed, batch_size, learning_rate, steps,
    )

    rows: List[Dict[str, Any]] = []

    print(f"  {'step':>5} {'loss(mean)':>15} {'loss(weighted)':>15} "
          f"{'loss 차이':>12} {'parameter 차이':>15}")
    print("  " + "-" * 68)

    for index in range(steps):
        pa, pb = snaps_a[index], snaps_b[index]
        param_dev = max(max_dev(pa[n], pb[n]) for n in pa)
        loss_diff = abs(losses_a[index] - losses_b[index])

        rows.append({
            "step": index + 1,
            "loss_mean": losses_a[index],
            "loss_weighted": losses_b[index],
            "loss_diff": loss_diff,
            "param_max_dev": param_dev,
        })

        print(f"  {index + 1:>5} {losses_a[index]:>15.10f} "
              f"{losses_b[index]:>15.10f} {loss_diff:>12.3e} "
              f"{param_dev:>15.3e}")

    print()

    if all(row["loss_diff"] == 0.0 for row in rows):
        print("  모든 step에서 loss가 같습니다. 두 경로는 같은 실행입니다.")
    else:
        first = next(r for r in rows if r["loss_diff"] > 0)
        print(f"  step {first['step']}에서 처음 갈라집니다 "
              f"(loss 차이 {first['loss_diff']:.3e}).")

    return rows


def compare_runs(mean_run: Optional[Path], weighted_run: Optional[Path]) -> None:
    section("11. 이미 끝난 두 run의 epoch 곡선")

    if not mean_run or not weighted_run:
        print("  run 경로를 주지 않아 건너뜁니다.")
        return

    for path in (mean_run, weighted_run):
        if not (path / "training_history.csv").exists():
            print(f"  training_history.csv가 없습니다: {path}")
            return

    def load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path / "training_history.csv", encoding="utf-8-sig")
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        return df

    a, b = load(mean_run), load(weighted_run)

    print(f"  {'epoch':>5} | {'train loss':>24} | {'val Top-1':>24}")
    print(f"  {'':>5} | {'mean':>11} {'weighted':>12} | "
          f"{'mean':>11} {'weighted':>12}")
    print("  " + "-" * 62)

    for epoch in sorted(set(a["epoch"]) & set(b["epoch"])):
        ra = a[a["epoch"] == epoch].iloc[0]
        rb = b[b["epoch"] == epoch].iloc[0]

        print(
            f"  {epoch:>5} | {ra['train_loss']:>11.6f} "
            f"{rb['train_loss']:>12.6f} | "
            f"{ra['validation_top1']:>11.6f} {rb['validation_top1']:>12.6f}"
        )

    first_a = a.iloc[0]
    first_b = b.iloc[0]

    print()
    print(f"  epoch 1 train loss 차이 : "
          f"{abs(first_a['train_loss'] - first_b['train_loss']):.3e}")
    print(f"  epoch 1 val Top-1 차이  : "
          f"{abs(first_a['validation_top1'] - first_b['validation_top1']):.3e}")
    print()
    print("  epoch 1부터 train loss가 다르면 학습 중에 갈라진 것입니다.")
    print("  train loss는 같은데 val Top-1만 다르면 평가 쪽 문제입니다.")


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="c123 mean과 weighted [1,1,1]이 같은 실행인지 확인한다."
    )
    parser.add_argument(
        "--priority0-dir",
        default="sweep_out/ebnerd_v2/priority0_shuffled/seed42",
    )
    parser.add_argument("--config", default="configs/transformer_ebnerd.gin")
    parser.add_argument(
        "--train-path",
        default="datasets/ebnerd/train_sequences_1pos4neg.parquet",
    )
    parser.add_argument("--mean-run", default=None)
    parser.add_argument("--weighted-run", default=None)
    parser.add_argument("--out", default=None)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)

    args = parser.parse_args()

    priority0_dir = resolve_path(args.priority0_dir)
    config_path = resolve_path(args.config)
    train_path = resolve_path(args.train_path)

    meta_path = priority0_dir / "run_metadata.json"

    if not meta_path.exists():
        print(f"Priority 0 run_metadata.json이 없습니다: {meta_path}")
        return 1

    p0_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    validation_path = resolve_path(p0_meta["dataset"])

    bindings = p0_meta.get("gin_bindings") or []

    if not bindings and p0_meta.get("config"):
        config = p0_meta["config"]
        bindings = [format_binding(k, config[k]) for k in sorted(config)]

    device = get_device()

    section("c123 mean vs weighted [1,1,1]")
    print(f"  train      : {train_path}")
    print(f"  validation : {validation_path}")
    print(f"  device     : {device}")
    print(f"  gin binding: {len(bindings)}개")
    print()
    print("  학습하지 않습니다. checkpoint를 만들지 않습니다.")

    determinism = report_determinism()

    model_mean, model_weighted, batch_mean, batch_weighted = build_pair(
        config_path, bindings, train_path, validation_path,
        device, args.seed, args.batch_size,
    )

    params_same, param_details = compare_parameters(model_mean, model_weighted)
    batch_same, batch_details = compare_batch(batch_mean, batch_weighted)

    forward_same, forward_details = compare_forward(
        model_mean, model_weighted, batch_mean, device
    )
    grad_same, grad_dev = compare_gradients(
        model_mean, model_weighted, batch_mean, device
    )

    self_dev = check_self_determinism(
        model_mean, batch_mean, device, args.repeats
    )

    steps = divergence_over_steps(
        config_path, bindings, train_path, validation_path, device,
        args.seed, args.batch_size, args.learning_rate, args.steps,
    )

    compare_runs(
        resolve_path(args.mean_run) if args.mean_run else None,
        resolve_path(args.weighted_run) if args.weighted_run else None,
    )

    # ---- 결론
    section("정리")

    print(f"  1-2. 초기값          : {verdict(max(param_details.values()))}")
    print(f"  3.   첫 batch        : "
          f"{'동일' if batch_same else '다름'}")
    print(f"  4.   user vector     : "
          f"{verdict(forward_details['4. user vector'])}")
    print(f"  5.   candidate vector: "
          f"{verdict(forward_details['5. candidate vector'])}")
    print(f"  6.   candidate score : "
          f"{verdict(forward_details['6. candidate score'])}")
    print(f"  7.   loss            : {verdict(forward_details['7. loss'])}")
    print(f"  8.   gradient        : {verdict(grad_dev)}")
    print(f"  9.   같은 모델 반복  : forward {verdict(self_dev['forward'])}, "
          f"backward {verdict(self_dev['backward'])}")

    print()

    if self_dev["forward"] > 0 or self_dev["backward"] > 0:
        print("  => 같은 모델을 두 번 돌려도 값이 달라집니다.")
        print("     GPU 비결정성이 있으므로, 두 run의 차이를 코드 경로")
        print("     때문이라고 단정할 수 없습니다. 재현 허용치를 이보다")
        print("     크게 잡거나, 결정론 설정을 켜고 다시 재야 합니다.")
    elif params_same and batch_same and forward_same and grad_same:
        print("  => 초기값, 데이터, forward, gradient가 전부 같습니다.")
        print("     한 step 기준으로 두 경로는 완전히 같은 실행입니다.")
        print("     그래도 최종 결과가 다르면 여러 step 누적 결과를 보세요")
        print("     (섹션 10).")
    else:
        print("  => 차이가 나는 지점이 있습니다. 위 섹션에서 어디인지 보세요.")

    if steps:
        last = steps[-1]
        print()
        print(f"  step {last['step']}까지 parameter 최대 차이 "
              f"{last['param_max_dev']:.3e}")

    # ---- 저장
    if args.out:
        out_dir = resolve_path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "compare_direct_runs.json").write_text(
            json.dumps({
                "determinism": determinism,
                "parameters": param_details,
                "batch": batch_details,
                "forward": forward_details,
                "gradient_max_dev": grad_dev,
                "self_determinism": self_dev,
                "steps": steps,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            }, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print()
        print(f"  저장: {out_dir / 'compare_direct_runs.json'}")

    return 0


from analysis.live_output import enable_line_buffering

enable_line_buffering()

if __name__ == "__main__":
    raise SystemExit(main())
