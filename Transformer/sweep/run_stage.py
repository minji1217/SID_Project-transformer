"""
단계별 파라미터 탐색 실행기.

한 번에 한 단계만 실행하고 멈춘다.
단계가 끝나면 summary.csv와 selected_auto.json을 남기므로,
사람이 결과를 확인한 뒤 다음 단계로 넘어갈 수 있다.

    cd Transformer
    python -m sweep.run_stage --config configs/transformer_mind.gin --stage 1
    # summary.csv 확인
    python -m sweep.run_stage --config configs/transformer_mind.gin --stage 1 --accept-auto
    python -m sweep.run_stage --config configs/transformer_mind.gin --stage 2

event_parameter_sweep/run_grid.py와 같은 원칙을 따른다.
  - 완료된 run은 _COMPLETE.json으로 표시하고 절대 덮어쓰지 않는다
  - 실패한 run의 폴더는 진단을 위해 남긴다
  - 같은 config는 다시 학습하지 않고 이전 결과를 재사용한다
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sweep.selection import (
    check_row_warnings,
    check_stage_warnings,
    rank_rows,
)
from sweep.stages import (
    METRIC_PRIORITY,
    STAGES,
    get_stage,
    stage_dir_name,
)


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = BASE_DIR / "train_transformer.py"

# summary.csv에서 앞쪽에 보여줄 지표 (우선순위 순)
METRIC_COLUMNS = [rule.key for rule in METRIC_PRIORITY]


def format_gin_value(value: Any) -> str:
    # gin이 읽을 수 있는 리터럴로 바꾼다
    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, str):
        return f'"{value}"'

    return repr(value)


def format_binding(key: str, value: Any) -> str:
    return f"{key} = {format_gin_value(value)}"


# "Class.param = value" 형태의 gin binding 한 줄
_BINDING_PATTERN = re.compile(
    r"^\s*([A-Za-z_][\w]*\.[A-Za-z_][\w]*)\s*=\s*(.+?)\s*$"
)


def load_base_bindings(path: Path) -> Dict[str, Any]:
    # 기준 config에 이미 들어 있는 값을 읽는다.
    #
    # 예를 들어 2단계의 max_history_length=20은 기준 config의 값과 같으므로
    # 1단계에서 이미 학습한 config와 동일하다.
    # 이런 경우를 같은 run으로 인식해 다시 학습하지 않기 위해 필요하다.
    #
    # 값을 읽지 못하는 줄은 건너뛴다. 그래도 동작에는 문제가 없고,
    # 중복 제거가 조금 덜 될 뿐이다.
    bindings: Dict[str, Any] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]

        matched = _BINDING_PATTERN.match(line)

        if matched is None:
            continue

        try:
            bindings[matched.group(1)] = ast.literal_eval(
                matched.group(2)
            )
        except (ValueError, SyntaxError):
            continue

    return bindings


def strip_defaults(
    bindings: Dict[str, Any],
    base_bindings: Dict[str, Any],
) -> Dict[str, Any]:
    # 기준 config와 값이 같은 binding은 빼고 돌려준다.
    # 학습에는 원래 binding을 그대로 넘기고,
    # 이 결과는 run의 신원(hash)을 정할 때만 쓴다.
    return {
        key: value
        for key, value in bindings.items()
        if key not in base_bindings or base_bindings[key] != value
    }


def config_hash(bindings: Dict[str, Any]) -> str:
    # 같은 설정이면 같은 폴더를 쓰게 해서 중복 학습을 막는다.
    payload = json.dumps(
        {key: bindings[key] for key in sorted(bindings)},
        sort_keys=True,
        default=str,
    )

    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def short_name(key: str) -> str:
    # "NewsEncoderDecoderTransformer.d_model" -> "d_model"
    return key.rsplit(".", 1)[-1]


def describe(bindings: Dict[str, Any]) -> str:
    return ", ".join(
        f"{short_name(key)}={bindings[key]}"
        for key in sorted(bindings)
    )


def load_carried_configs(
    sweep_dir: Path,
    stage_number: int,
) -> List[Dict[str, Any]]:
    # 1단계는 base config 그대로 시작한다
    if stage_number == 1:
        return [{}]

    previous_dir = sweep_dir / stage_dir_name(stage_number - 1)
    selected_path = previous_dir / "selected.json"

    if not selected_path.exists():
        auto_path = previous_dir / "selected_auto.json"

        hint = (
            f"\n이전 단계의 자동 선택 결과가 있습니다: {auto_path}\n"
            "내용을 확인한 뒤 selected.json으로 복사하거나, "
            "이전 단계를 --accept-auto로 다시 실행하세요."
            if auto_path.exists()
            else f"\n먼저 {stage_number - 1}단계를 실행하세요."
        )

        raise FileNotFoundError(
            f"이전 단계의 확정 파일이 없습니다: {selected_path}{hint}"
        )

    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    configs = payload.get("configs")

    if not configs:
        raise ValueError(
            f"{selected_path}에 configs가 비어 있습니다."
        )

    return configs


def run_training(
    run_dir: Path,
    base_config: Path,
    bindings: Dict[str, Any],
) -> None:
    # train_transformer.py를 별도 프로세스로 실행한다.
    # gin은 전역 상태라 한 프로세스에서 여러 config를 연달아
    # 적용하면 설정이 섞일 수 있어 run마다 프로세스를 분리한다.
    run_dir.mkdir(parents=True, exist_ok=False)

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(base_config),
    ]

    for key in sorted(bindings):
        command += ["--gin-binding", format_binding(key, bindings[key])]

    command += [
        "--gin-binding",
        format_binding("train.save_dir", str(run_dir)),
    ]

    (run_dir / "bindings.json").write_text(
        json.dumps(bindings, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    log_path = run_dir / "train_log.txt"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    if process.returncode != 0:
        # 실패한 폴더는 지우지 않는다.
        # _COMPLETE.json이 없으므로 완료된 run과 구분된다.
        raise RuntimeError(
            f"학습이 실패했습니다 (exit code {process.returncode}).\n"
            f"로그: {log_path}"
        )


def flatten_summary(
    run_dir: Path,
    bindings: Dict[str, Any],
    stage_name: str,
) -> Dict[str, Any]:
    summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )

    row: Dict[str, Any] = {
        "stage": stage_name,
        "config_hash": run_dir.name,
        "config": describe(bindings) or "(base config)",
    }

    for key, value in bindings.items():
        row[short_name(key)] = value

    for key in (
        "best_epoch",
        "num_epochs_run",
        "stopped_early",
        "mean_epoch_seconds",
        "total_seconds",
        "seed",
        "total_parameters",
        "amp_dtype",
        "gpu",
    ):
        row[key] = summary.get(key)

    for key, value in (summary.get("best_metrics") or {}).items():
        if key in ("epoch", "epoch_seconds"):
            continue
        row[key] = value

    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    # 지표를 앞쪽에 오게 열 순서를 정리한다
    leading = ["rank", "stage", "config", "config_hash"]
    trailing = ["warnings", "decision"]

    seen = set(leading) | set(trailing)
    middle: List[str] = []

    for key in METRIC_COLUMNS:
        if any(key in row for row in rows) and key not in seen:
            middle.append(key)
            seen.add(key)

    for row in rows:
        for key in row:
            if key not in seen:
                middle.append(key)
                seen.add(key)

    fieldnames = (
        [key for key in leading if any(key in row for row in rows)]
        + middle
        + [key for key in trailing if any(key in row for row in rows)]
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer 단계별 파라미터 탐색",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="기준 gin config (예: configs/transformer_mind.gin)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        help=f"실행할 단계 번호 (1 ~ {len(STAGES)})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="sweep_out/default",
        help="결과를 모아 둘 폴더",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=12,
        help="탐색용 최대 epoch (기본 12, 최종 학습은 30)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="탐색용 early stopping patience (기본 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--keep-optimizer-state",
        action="store_true",
        help="checkpoint에 optimizer 상태를 포함한다 (용량 약 3배)",
    )
    parser.add_argument(
        "--accept-auto",
        action="store_true",
        help=(
            "자동 선택 결과를 그대로 확정한다. "
            "사람이 확인하지 않고 다음 단계로 넘어갈 때 사용한다."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실행할 run 목록만 출력하고 학습은 하지 않는다.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    stage = get_stage(args.stage)
    sweep_dir = Path(args.out)

    if not sweep_dir.is_absolute():
        sweep_dir = BASE_DIR / sweep_dir

    base_config = Path(args.config)

    if not base_config.is_absolute():
        base_config = BASE_DIR / base_config

    if not base_config.exists():
        print(f"gin config를 찾을 수 없습니다: {base_config}")
        return 1

    runs_dir = sweep_dir / "runs"
    stage_dir = sweep_dir / stage_dir_name(args.stage)

    base_bindings = load_base_bindings(base_config)

    # 탐색 예산과 저장 정책은 모든 run에 동일하게 적용한다
    budget_bindings: Dict[str, Any] = {
        "train.num_epochs": args.num_epochs,
        "train.early_stopping_patience": args.patience,
        "train.seed": args.seed,
        "train.save_every_epoch": False,
        "train.save_optimizer_state": bool(args.keep_optimizer_state),
    }

    carried_configs = load_carried_configs(sweep_dir, args.stage)

    print("=" * 78)
    print(f"Stage {args.stage}/{len(STAGES)}  {stage.name}")
    print("=" * 78)
    print(stage.description)
    print(f"Base config : {base_config}")
    print(f"Sweep dir   : {sweep_dir}")
    print(f"이어받은 config: {len(carried_configs)}개")
    print(f"이 단계 변형   : {len(stage.variants)}개")
    print(f"예정 run       : {len(carried_configs) * len(stage.variants)}개")
    print(f"탐색 예산      : 최대 {args.num_epochs} epoch, patience {args.patience}")
    print("=" * 78)

    planned: List[Dict[str, Any]] = []

    for carried in carried_configs:
        for variant in stage.variants:
            bindings = dict(carried)
            bindings.update(variant)

            full_bindings = dict(bindings)
            full_bindings.update(budget_bindings)

            planned.append(
                {
                    "params": bindings,
                    "full": full_bindings,
                    "hash": config_hash(
                        strip_defaults(full_bindings, base_bindings)
                    ),
                }
            )

    if args.dry_run:
        for index, item in enumerate(planned, 1):
            run_dir = runs_dir / item["hash"]
            status = (
                "재사용"
                if (run_dir / "_COMPLETE.json").exists()
                else "실행"
            )
            print(f"{index:>3}. [{status}] {describe(item['params'])}")
        return 0

    rows: List[Dict[str, Any]] = []

    for index, item in enumerate(planned, 1):
        run_dir = runs_dir / item["hash"]
        complete_marker = run_dir / "_COMPLETE.json"

        print()
        print("-" * 78)
        print(f"[{index}/{len(planned)}] {describe(item['params'])}")

        if complete_marker.exists():
            print(f"이미 완료된 config라 재사용합니다: {run_dir.name}")

        elif run_dir.exists():
            print(
                "완료 표시가 없는 폴더가 이미 있습니다. "
                "덮어쓰지 않고 중단합니다.\n"
                f"경로: {run_dir}\n"
                "이전 실행이 실패한 흔적이라면 폴더를 확인하고 직접 지우세요."
            )
            return 1

        else:
            run_training(
                run_dir=run_dir,
                base_config=base_config,
                bindings=item["full"],
            )

            complete_marker.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "stage": stage.name,
                        "bindings": item["full"],
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            print(f"완료: {run_dir.name}")

        row = flatten_summary(
            run_dir=run_dir,
            bindings=item["params"],
            stage_name=stage.name,
        )
        row["_params"] = item["params"]
        rows.append(row)

    # 선택
    ordered = rank_rows(rows)
    winner = ordered[0][0]

    for position, (row, trace) in enumerate(ordered, 1):
        row["rank"] = position
        row["decision"] = " | ".join(trace)
        row["warnings"] = " | ".join(check_row_warnings(row))

    stage_warnings = check_stage_warnings(rows, winner)

    summary_path = stage_dir / "summary.csv"
    write_csv(summary_path, [row for row, _ in ordered])

    selected = [
        row["_params"]
        for row, _ in ordered[: stage.top_k]
    ]

    auto_payload = {
        "stage": args.stage,
        "stage_name": stage.name,
        "top_k": stage.top_k,
        "configs": selected,
        "stage_warnings": stage_warnings,
        "note": (
            "이 파일은 자동 선택 결과입니다. "
            "확정하려면 selected.json으로 복사하세요."
        ),
    }

    auto_path = stage_dir / "selected_auto.json"
    auto_path.write_text(
        json.dumps(auto_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(f"Stage {args.stage} 완료")
    print("=" * 78)

    for position, (row, _) in enumerate(ordered, 1):
        marker = "->" if position <= stage.top_k else "  "
        print(
            f"{marker} {position}. {row['config']}\n"
            f"      Top-1={row.get('val_top1_accuracy')} "
            f"MRR={row.get('val_mrr')} "
            f"nDCG@5={row.get('val_ndcg@5')} "
            f"AUC={row.get('val_auc')}"
        )

        if row.get("warnings"):
            print(f"      [경고] {row['warnings']}")

    if stage_warnings:
        print()
        print("단계 경고:")
        for warning in stage_warnings:
            print(f"  - {warning}")

    print()
    print(f"요약 CSV   : {summary_path}")
    print(f"자동 선택   : {auto_path}")

    selected_path = stage_dir / "selected.json"

    if args.accept_auto:
        selected_path.write_text(
            json.dumps(auto_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"확정 완료   : {selected_path}")
        print(f"다음: python -m sweep.run_stage --config {args.config} --stage {args.stage + 1}")

    else:
        print()
        print("summary.csv를 확인한 뒤 다음 중 하나를 하세요:")
        print(f"  1) 자동 선택 그대로 확정: 같은 명령에 --accept-auto 추가")
        print(f"  2) 직접 고르기: selected_auto.json을 편집해 "
              f"{selected_path.name}으로 저장")

    return 0


if __name__ == "__main__":
    sys.exit(main())
