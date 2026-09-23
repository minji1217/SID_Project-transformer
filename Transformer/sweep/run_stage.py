"""
단계별 파라미터 탐색 실행기.

한 번에 한 단계만 실행하고 멈춘다.
단계가 끝나면 summary.csv와 selected_auto.json을 남기므로,
사람이 결과를 확인한 뒤 다음 단계로 넘어갈 수 있다.

    cd Transformer
    python -m sweep.run_stage --config configs/transformer_ebnerd.gin --stage 1
    # summary.csv 확인
    python -m sweep.run_stage --config configs/transformer_ebnerd.gin --stage 1 --accept-auto
    python -m sweep.run_stage --config configs/transformer_ebnerd.gin --stage 2

event_parameter_sweep/run_grid.py와 같은 원칙을 따른다.
  - 완료된 run은 _COMPLETE.json으로 표시하고 절대 덮어쓰지 않는다
  - 실패한 run의 폴더는 진단을 위해 남긴다
  - 같은 config는 다시 학습하지 않고 이전 결과를 재사용한다

Test 데이터는 어떤 단계의 선택에도 사용하지 않는다.
모든 판단은 Validation 지표로만 한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sweep.failures import build_failure_record
from sweep.hashing import (
    compute_config_hash,
    file_sha256,
    load_base_bindings,
    source_fingerprint,
)
from sweep.selection import (
    NotEnoughSuccessfulRuns,
    check_row_warnings,
    check_stage_warnings,
    select_stage_top_k,
)
from sweep.stages import (
    METRIC_PRIORITY,
    STAGES,
    get_stage,
    is_final_stage,
    stage_dir_name,
)


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = BASE_DIR / "train_transformer.py"

COMPLETE_MARKER = "_COMPLETE.json"
FAILED_MARKER = "_FAILED.json"

# summary.csv에서 앞쪽에 보여줄 지표 (우선순위 순)
METRIC_COLUMNS = [rule.key for rule in METRIC_PRIORITY]

# summary.csv에 내보내지 않는 내부 값
INTERNAL_COLUMNS = {"_params"}


def format_gin_value(value: Any) -> str:
    # gin이 읽을 수 있는 리터럴로 바꾼다
    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, str):
        return f'"{value}"'

    return repr(value)


def format_binding(key: str, value: Any) -> str:
    return f"{key} = {format_gin_value(value)}"


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
    # 1단계는 기준 config 그대로 시작한다
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
        raise ValueError(f"{selected_path}에 configs가 비어 있습니다.")

    return configs


def backup_run_dir(run_dir: Path) -> Path:
    # 실패한 폴더를 지우지 않고 시각이 붙은 이름으로 옮긴다.
    # 같은 오류가 반복되는지 나중에 비교할 수 있어야 한다.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = run_dir.with_name(f"{run_dir.name}.failed.{timestamp}")

    run_dir.rename(backup_path)

    return backup_path


def run_training(
    run_dir: Path,
    base_config: Path,
    bindings: Dict[str, Any],
    summary_extra: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    # train_transformer.py를 별도 프로세스로 실행한다.
    #
    # gin은 전역 상태라 한 프로세스에서 여러 config를 연달아 적용하면
    # 설정이 섞인다. run마다 프로세스를 분리해야 안전하다.
    #
    # 성공하면 None, 실패하면 실패 기록을 돌려준다.
    run_dir.mkdir(parents=True, exist_ok=False)

    extra_path = run_dir / "summary_extra.json"
    extra_path.write_text(
        json.dumps(summary_extra, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    (run_dir / "bindings.json").write_text(
        json.dumps(bindings, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(base_config),
        "--summary-extra",
        str(extra_path),
    ]

    for key in sorted(bindings):
        command += ["--gin-binding", format_binding(key, bindings[key])]

    command += [
        "--gin-binding",
        format_binding("train.save_dir", str(run_dir)),
    ]

    log_path = run_dir / "train_log.txt"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    if process.returncode == 0 and (run_dir / "run_summary.json").exists():
        return None

    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    return build_failure_record(
        log_text=log_text,
        return_code=process.returncode,
        log_path=str(log_path),
    )


def success_row(
    run_dir: Path,
    params: Dict[str, Any],
    stage_name: str,
) -> Dict[str, Any]:
    summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )

    row: Dict[str, Any] = {
        "stage": stage_name,
        "status": "SUCCESS",
        "config_hash": run_dir.name,
        "config": describe(params) or "(base config)",
    }

    for key, value in params.items():
        row[short_name(key)] = value

    for key in (
        "best_epoch",
        "num_epochs_run",
        "stopped_early",
        "mean_epoch_seconds",
        "total_seconds",
        "seed",
        "total_parameters",
        "trainable_parameters",
        "peak_gpu_memory_mb",
        "peak_gpu_memory_reserved_mb",
        "amp_dtype",
        "gpu",
        "base_config_sha256",
    ):
        row[key] = summary.get(key)

    for key, value in (summary.get("best_metrics") or {}).items():
        if key in ("epoch", "epoch_seconds"):
            continue
        row[key] = value

    return row


def failure_row(
    run_dir: Path,
    params: Dict[str, Any],
    stage_name: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "stage": stage_name,
        "status": "FAILED",
        "config_hash": run_dir.name,
        "config": describe(params) or "(base config)",
        "error_type": record.get("error_type"),
        "return_code": record.get("return_code"),
        "message": record.get("message"),
        "log_path": record.get("log_path"),
    }

    for key, value in params.items():
        row[short_name(key)] = value

    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    # 지표를 앞쪽에 오게 열 순서를 정리한다
    leading = ["rank", "stage", "status", "config", "config_hash"]
    trailing = ["error_type", "message", "log_path", "warnings", "decision"]

    seen = set(leading) | set(trailing) | INTERNAL_COLUMNS
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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        help=f"실행할 단계 번호 (1 ~ {len(STAGES)})",
    )
    parser.add_argument("--out", type=str, default="sweep_out/default")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-optimizer-state",
        action="store_true",
        help="checkpoint에 optimizer 상태를 포함한다 (용량 약 3배)",
    )
    parser.add_argument(
        "--accept-auto",
        action="store_true",
        help="자동 선택 결과를 그대로 확정하고 다음 단계로 넘어간다.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="첫 실패에서 즉시 중단한다. 기본은 기록 후 계속 진행한다.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "이전에 실패한 run을 다시 실행한다. "
            "기존 폴더는 지우지 않고 시각이 붙은 이름으로 옮긴다."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    stage = get_stage(args.stage)
    final_stage = is_final_stage(args.stage)

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
    base_sha256 = file_sha256(base_config)
    fingerprint = source_fingerprint(BASE_DIR)

    # 탐색 예산과 저장 정책은 모든 run에 동일하게 적용한다
    budget_bindings: Dict[str, Any] = {
        "train.num_epochs": args.num_epochs,
        "train.early_stopping_patience": args.patience,
        "train.seed": args.seed,
        "train.save_every_epoch": False,
        "train.save_optimizer_state": bool(args.keep_optimizer_state),
    }

    try:
        carried_configs = load_carried_configs(sweep_dir, args.stage)
    except (FileNotFoundError, ValueError) as error:
        print("=" * 78)
        print("이전 단계의 결과를 읽을 수 없습니다.")
        print("=" * 78)
        print(error)
        return 1

    print("=" * 78)
    print(f"Stage {args.stage}/{len(STAGES)}  {stage.name}")
    print("=" * 78)
    print(stage.description)
    print(f"Base config   : {base_config}")
    print(f"Config SHA-256: {base_sha256[:16]}...")
    print(f"Source        : {fingerprint['source_sha256']}", end="")
    print(
        f" (git {fingerprint['git_commit'][:8]}"
        f"{'+dirty' if fingerprint['git_dirty'] else ''})"
        if fingerprint["git_commit"]
        else ""
    )
    print(f"Sweep dir     : {sweep_dir}")
    print(f"이어받은 config: {len(carried_configs)}개")
    print(f"이 단계 변형   : {len(stage.variants)}개")
    print(f"예정 run       : {len(carried_configs) * len(stage.variants)}개")
    print(f"탐색 예산      : 최대 {args.num_epochs} epoch, patience {args.patience}")
    print(
        f"다음 단계 유지 : 상위 {stage.top_k}개"
        + ("  (Final Top-3)" if final_stage else "")
    )
    print("=" * 78)

    planned: List[Dict[str, Any]] = []

    for carried in carried_configs:
        for variant in stage.variants:
            params = dict(carried)
            params.update(variant)

            full_bindings = dict(params)
            full_bindings.update(budget_bindings)

            planned.append(
                {
                    "params": params,
                    "full": full_bindings,
                    "hash": compute_config_hash(
                        base_config_path=base_config,
                        bindings=full_bindings,
                        base_bindings=base_bindings,
                    ),
                }
            )

    if args.dry_run:
        for index, item in enumerate(planned, 1):
            run_dir = runs_dir / item["hash"]

            if (run_dir / COMPLETE_MARKER).exists():
                status = "재사용"
            elif (run_dir / FAILED_MARKER).exists():
                status = "재시도" if args.retry_failed else "실패기록"
            else:
                status = "실행"

            print(f"{index:>3}. [{status}] {describe(item['params'])}")

        print()
        print(f"총 {len(planned)}개")
        return 0

    rows: List[Dict[str, Any]] = []
    params_by_hash: Dict[str, Dict[str, Any]] = {}

    for index, item in enumerate(planned, 1):
        run_dir = runs_dir / item["hash"]
        params_by_hash[item["hash"]] = item["params"]

        complete_marker = run_dir / COMPLETE_MARKER
        failed_marker = run_dir / FAILED_MARKER

        print()
        print("-" * 78)
        print(f"[{index}/{len(planned)}] {describe(item['params'])}")

        # 이전에 실패한 run을 다시 돌리는 경우
        if args.retry_failed and run_dir.exists() and not complete_marker.exists():
            backup_path = backup_run_dir(run_dir)
            print(f"이전 실패 결과를 보관했습니다: {backup_path.name}")

        if complete_marker.exists():
            print(f"이미 완료된 config라 재사용합니다: {run_dir.name}")
            rows.append(success_row(run_dir, item["params"], stage.name))
            continue

        if failed_marker.exists():
            record = json.loads(failed_marker.read_text(encoding="utf-8"))
            print(
                f"이전에 실패한 run입니다 ({record.get('error_type')}). "
                "다시 돌리려면 --retry-failed를 쓰세요."
            )
            rows.append(
                failure_row(run_dir, item["params"], stage.name, record)
            )
            continue

        if run_dir.exists():
            # 완료 표시도 실패 표시도 없는 폴더.
            # 학습 도중에 중단된 흔적이므로 덮어쓰지 않는다.
            record = {
                "status": "FAILED",
                "error_type": "UNKNOWN",
                "return_code": None,
                "message": (
                    "완료 표시가 없는 폴더가 이미 존재합니다. "
                    "학습 도중 중단된 것으로 보입니다."
                ),
                "log_path": str(run_dir / "train_log.txt"),
            }
            failed_marker.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"완료 표시가 없는 폴더입니다: {run_dir}")

            if args.fail_fast:
                return 1

            rows.append(
                failure_row(run_dir, item["params"], stage.name, record)
            )
            continue

        summary_extra = {
            "config_hash": item["hash"],
            "stage": stage.name,
            "stage_number": args.stage,
            "source_fingerprint": fingerprint,
        }

        record = run_training(
            run_dir=run_dir,
            base_config=base_config,
            bindings=item["full"],
            summary_extra=summary_extra,
        )

        if record is None:
            complete_marker.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "stage": stage.name,
                        "config_hash": item["hash"],
                        "base_config_sha256": base_sha256,
                        "source_fingerprint": fingerprint,
                        "bindings": item["full"],
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            print(f"완료: {run_dir.name}")
            rows.append(success_row(run_dir, item["params"], stage.name))
            continue

        # 실패
        failed_marker.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"실패 [{record['error_type']}]: {record['message']}")
        print(f"로그: {record['log_path']}")

        if args.fail_fast:
            print("--fail-fast가 지정되어 중단합니다.")
            write_csv(
                stage_dir / "summary.csv",
                rows + [failure_row(run_dir, item["params"], stage.name, record)],
            )
            return 1

        print("나머지 run을 계속 진행합니다.")
        rows.append(failure_row(run_dir, item["params"], stage.name, record))

    # 선택
    summary_path = stage_dir / "summary.csv"

    try:
        ordered, failed_rows, selected_rows = select_stage_top_k(
            rows,
            stage.top_k,
        )

    except NotEnoughSuccessfulRuns as error:
        # 순위를 매길 수 없어도 무엇이 실패했는지는 남긴다
        for row in rows:
            row["warnings"] = " | ".join(check_row_warnings(row))

        write_csv(summary_path, rows)

        print()
        print("=" * 78)
        print("자동 선택을 할 수 없습니다.")
        print("=" * 78)
        print(error)
        print()
        print(f"요약 CSV: {summary_path}")
        return 1

    for position, (row, trace) in enumerate(ordered, 1):
        row["rank"] = position
        row["decision"] = " | ".join(trace)
        row["warnings"] = " | ".join(check_row_warnings(row))

    for row in failed_rows:
        row["rank"] = None
        row["warnings"] = "실패한 run이라 순위 선정에서 제외됨"

    stage_warnings = check_stage_warnings(
        [row for row, _ in ordered],
        ordered[0][0],
    )

    write_csv(summary_path, [row for row, _ in ordered] + failed_rows)

    selected = [
        params_by_hash[row["config_hash"]]
        for row in selected_rows
    ]

    auto_payload: Dict[str, Any] = {
        "stage": args.stage,
        "stage_name": stage.name,
        "top_k": stage.top_k,
        "is_final_stage": final_stage,
        "configs": selected,
        "stage_warnings": stage_warnings,
        "failed_run_count": len(failed_rows),
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

    if failed_rows:
        print()
        print(f"실패한 run {len(failed_rows)}개 (순위 선정 제외):")
        for row in failed_rows:
            print(f"  - [{row.get('error_type')}] {row['config']}")

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

        if final_stage:
            print()
            print("마지막 단계입니다. Final Top-3로 seed 검증을 실행하세요:")
            print(
                f"  python -m sweep.run_seed_robustness "
                f"--config {args.config} --sweep-out {args.out} "
                f"--seeds 42 123 2026 --num-epochs 30 --patience 5"
            )
        else:
            print(
                f"다음: python -m sweep.run_stage --config {args.config} "
                f"--stage {args.stage + 1} --out {args.out}"
            )

    else:
        print()
        print("summary.csv를 확인한 뒤 다음 중 하나를 하세요:")
        print("  1) 자동 선택 그대로 확정: 같은 명령에 --accept-auto 추가")
        print(
            f"  2) 직접 고르기: selected_auto.json을 편집해 "
            f"{selected_path.name}으로 저장"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
