"""이미 학습된 P2 checkpoint로 validation 예측만 다시 저장한다.

학습하지 않는다. checkpoint를 새로 만들지 않는다. weight를 건드리지
않는다. 저장된 checkpoint를 읽어 forward만 다시 돌린다.

쓰는 이유: 처음 예측을 저장할 때 candidate_article_id를 빠뜨렸다.
score fusion 진단에서 두 score의 후보 정렬을 기사 ID로 대조하려면
그 컬럼이 필요하다.

checkpoint에는 이 run이 쓴 gin binding과 구조 설정이 같이 들어 있으므로
그대로 복원한다. 복원한 설정이 원래 run의 run_metadata.json과 다르면
멈춘다.

    cd Transformer
    python -m analysis.repredict_direct \
        --run sweep_out/ebnerd_v2/priority2_direct/regularization_sweep/do0.00_wd0.0000/seed42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import gin
import pandas as pd
import torch

from torch.utils.data import DataLoader

from data.sequence import NewsSequenceDataset, collate_news_sequences
from modules.direct_scorer import DirectScorer
from modules.model import NewsEncoderDecoderTransformer
from analysis.run_priority2_direct import (
    GROUP_COLUMN,
    offline_metrics,
    predict_validation,
    resolve_path,
)


BASE_DIR = Path(__file__).resolve().parent.parent

# 학습되는 값이 아니라 level 가중치를 담아둔 상수 버퍼다. 생성자에서
# level_weights로 다시 만들어지므로 오래된 checkpoint에 없어도 된다.
# 대신 복원된 값이 metadata와 맞는지 따로 확인한다.
ALLOWED_MISSING = {"level_weight_vector"}

OUTPUT_NAME = "validation_predictions.parquet"
BACKUP_NAME = "validation_predictions_without_article_id.parquet"


def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P2 checkpoint로 validation 예측만 다시 저장한다."
    )
    parser.add_argument("--run", required=True, help="P2 run 폴더 (seed42)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    started_at = time.time()

    run_dir = resolve_path(args.run)
    checkpoint_path = run_dir / "checkpoint_best.pt"
    metadata_path = run_dir / "run_metadata.json"
    output_path = run_dir / OUTPUT_NAME

    section("validation 예측 다시 저장")

    for path, label in (
        (checkpoint_path, "checkpoint_best.pt"),
        (metadata_path, "run_metadata.json"),
    ):
        if not path.exists():
            print(f"{label}가 없습니다: {path}")
            return 1

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    validation_path = resolve_path(metadata["validation_path"])

    if not validation_path.exists():
        print(f"validation parquet이 없습니다: {validation_path}")
        return 1

    bindings: List[str] = (
        checkpoint.get("gin_bindings") or metadata.get("gin_bindings") or []
    )

    device = get_device()

    print(f"  run        : {run_dir}")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  validation : {validation_path}")
    print(f"  device     : {device}")
    print(f"  scorer     : {metadata['scorer_type']}")
    print(f"  history    : {metadata.get('history_codes', '?')} "
          f"({metadata.get('history_mode', 'mean')}, "
          f"{metadata.get('history_levels', '?')} level)")
    print(f"  level 가중 : {metadata.get('level_weights')}")
    print(f"  beta4      : {metadata.get('beta4', 0.0)} "
          f"(사용 {metadata.get('use_c4_identity', False)})")
    print()
    print("  학습하지 않습니다. weight를 바꾸지 않습니다.")

    # ---- 모델 복원
    section("모델 복원")

    gin.parse_config_file(
        str(resolve_path(metadata["config"])), skip_unknown=True
    )

    if bindings:
        gin.parse_config(bindings, skip_unknown=True)
        print(f"  gin binding {len(bindings)}개 적용")

    backbone = NewsEncoderDecoderTransformer().to(device)

    model = DirectScorer(
        backbone=backbone,
        scorer_type=metadata["scorer_type"],
        history_levels=int(metadata.get("history_levels", 4)),
        history_mode=metadata.get("history_mode", "mean"),
        level_weights=tuple(metadata.get("level_weights", (1.0, 1.0, 1.0))),
        use_c4_identity=bool(metadata.get("use_c4_identity", False)),
        beta4=float(metadata.get("beta4", 0.0)),
    ).to(device)

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )

    unexpected_missing = [n for n in missing if n not in ALLOWED_MISSING]

    if unexpected_missing or unexpected:
        print(f"  state_dict 불일치")
        print(f"    missing    : {unexpected_missing}")
        print(f"    unexpected : {list(unexpected)}")
        print("  구조가 저장 당시와 다릅니다. 중단합니다.")
        return 1

    if missing:
        print(f"  checkpoint에 없던 상수 버퍼: {sorted(missing)}")
        print(f"    생성자가 다시 만든 값을 씁니다.")

    print(f"  나머지 state_dict는 그대로 불렀습니다 (unexpected 0)")
    print(f"  checkpoint epoch: {checkpoint.get('epoch')}")

    # level 가중치가 metadata와 맞는지 확인한다.
    expected_weights = tuple(
        float(w) for w in metadata.get("level_weights", (1.0, 1.0, 1.0))
    )
    actual_weights = tuple(
        float(w) for w in model.level_weight_vector.detach().cpu().tolist()
    )

    if actual_weights != expected_weights:
        print(f"  level 가중치가 다릅니다: "
              f"{actual_weights} vs {expected_weights}")
        return 1

    print(f"  level 가중치 확인: {actual_weights}")

    # ---- 예측
    section("예측")

    dataset = NewsSequenceDataset(parquet_path=str(validation_path))

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
    )

    predictions = predict_validation(model, loader, device)

    print(f"  {len(predictions):,} row")
    print(f"  컬럼: {list(predictions.columns)}")

    if "candidate_article_id" not in predictions.columns:
        print("  candidate_article_id가 여전히 없습니다. 중단합니다.")
        return 1

    missing_ids = int(predictions["candidate_article_id"].isna().sum())

    print(f"  candidate_article_id 비어 있는 행: {missing_ids:,}")

    if missing_ids:
        print("  일부 행에 기사 ID가 없습니다. 중단합니다.")
        return 1

    # ---- 기존 결과와 같은지 확인
    section("기존 예측과 같은가")

    offline = offline_metrics(predictions)
    stored = json.loads(
        (run_dir / "best_metrics.json").read_text(encoding="utf-8")
    )["validation"]

    pairs = [
        ("Top-1", "top1_accuracy", "top1_accuracy"),
        ("MRR", "mrr", "mrr"),
        ("nDCG@5", "ndcg@5", "ndcg@5"),
        ("AUC", "auc", "auc"),
    ]

    worst = 0.0

    print(f"  {'지표':<10} {'저장된 값':>14} {'다시 계산':>14} {'차이':>12}")
    print("  " + "-" * 54)

    for display, offline_key, stored_key in pairs:
        got = offline[offline_key]
        want = stored[stored_key]
        delta = abs(got - want)
        worst = max(worst, delta)
        print(f"  {display:<10} {want:>14.10f} {got:>14.10f} {delta:>12.3e}")

    print()

    if worst > 1e-6:
        print(f"  !! 기존 결과와 다릅니다 (최대 {worst:.3e}).")
        print("     구조나 checkpoint가 저장 당시와 달라졌을 수 있습니다.")
        print("     기존 파일을 건드리지 않고 중단합니다.")
        return 1

    print("  기존 결과와 일치합니다. 같은 모델, 같은 예측입니다.")

    # ---- 저장
    section("저장")

    if output_path.exists():
        backup_path = run_dir / BACKUP_NAME

        if backup_path.exists():
            print(f"  백업이 이미 있습니다: {backup_path}")
            print("  기존 파일을 덮어쓰지 않고 중단합니다.")
            return 1

        output_path.rename(backup_path)
        print(f"  기존 파일 보관 : {backup_path}")

    predictions.to_parquet(output_path, index=False)

    print(f"  새 파일 저장   : {output_path}")
    print(f"  추가된 컬럼    : candidate_article_id")
    print()
    print(f"  총 소요 {time.time() - started_at:.0f}초")

    return 0


from analysis.live_output import enable_line_buffering

enable_line_buffering()

if __name__ == "__main__":
    raise SystemExit(main())
