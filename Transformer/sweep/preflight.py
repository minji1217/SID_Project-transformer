"""
긴 sweep을 시작하기 전에 환경과 데이터를 점검한다.

84 run을 돌리다가 30번째에서 데이터 컬럼이 없어 실패하는 일을 막는 것이 목적이다.

    cd Transformer
    python -m sweep.preflight --config configs/transformer_ebnerd.gin

기본은 앞쪽 일부 sample만 본다. 전체를 확인하려면 --full-data-check를 쓴다.
검사를 모두 통과하면 PREFLIGHT PASSED를 출력하고 0으로 종료한다.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sweep.hashing import load_base_bindings


BASE_DIR = Path(__file__).resolve().parent.parent

REQUIRED_MODULES = (
    "torch",
    "transformers",
    "gin",
    "numpy",
    "pandas",
    "pyarrow",
)

REQUIRED_COLUMNS = (
    "history_c1",
    "history_c2",
    "history_c3",
    "history_c4",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_labels",
)

NUM_CANDIDATES = 5
NUM_POSITIVES = 1

# 디스크 여유가 이보다 적으면 경고
MIN_FREE_GIB = 5.0


class Report:
    # 검사 결과를 모아서 마지막에 한 번에 판정한다.
    # 첫 실패에서 멈추면 남은 문제를 한 번에 볼 수 없다.

    def __init__(self) -> None:
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def ok(self, name: str, detail: str = "") -> None:
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name: str, detail: str) -> None:
        self.warnings.append(f"{name}: {detail}")
        print(f"  [WARN] {name} — {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} — {detail}")

    def section(self, title: str) -> None:
        print()
        print(title)


def check_imports(report: Report) -> None:
    report.section("1. Python 패키지")

    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            report.fail(name, f"import 실패 ({error})")
            continue

        version = getattr(module, "__version__", "?")
        report.ok(name, str(version))


def check_cuda(report: Report, amp_dtype: str) -> None:
    report.section("2~3. CUDA와 AMP")

    try:
        import torch
    except ImportError:
        report.fail("torch", "import 실패라 CUDA를 확인할 수 없습니다.")
        return

    if not torch.cuda.is_available():
        report.warn(
            "CUDA",
            "사용할 수 없습니다. CPU로도 동작하지만 sweep에는 매우 느립니다.",
        )
        return

    report.ok("CUDA", torch.cuda.get_device_name(0))

    requested = str(amp_dtype).strip().lower()

    if requested in ("bfloat16", "bf16"):
        if torch.cuda.is_bf16_supported():
            report.ok("AMP bfloat16", "지원됨")
        else:
            report.warn(
                "AMP bfloat16",
                "이 GPU는 지원하지 않아 학습 시 float16으로 자동 전환됩니다.",
            )
    else:
        report.ok("AMP", f"{requested} 사용 설정")


def read_sample(
    path: Path,
    columns: Tuple[str, ...],
    sample_rows: int,
    full: bool,
):
    # 필요한 컬럼만, 필요한 만큼만 읽는다.
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(str(path))
    available = set(parquet_file.schema_arrow.names)
    usable = [column for column in columns if column in available]

    frames = []

    for batch in parquet_file.iter_batches(
        batch_size=sample_rows,
        columns=usable,
    ):
        frames.append(batch.to_pandas())

        if not full:
            break

    import pandas as pd

    if not frames:
        return pd.DataFrame(columns=usable), available, parquet_file.metadata.num_rows

    return (
        pd.concat(frames, ignore_index=True),
        available,
        parquet_file.metadata.num_rows,
    )


def check_dataset(
    report: Report,
    label: str,
    path: Path,
    vocab_sizes: Dict[str, int],
    sample_rows: int,
    full: bool,
) -> None:
    if not path.exists():
        hint = ""

        # 상대경로인데 datasets 링크가 없으면 그것이 원인일 가능성이 높다
        if not (BASE_DIR / "datasets").exists():
            hint = (
                "\n           datasets 링크가 없습니다. "
                "ln -s ~/shared/datasets "
                f"{BASE_DIR / 'datasets'}"
            )

        report.fail(f"{label} 파일", f"찾을 수 없습니다: {path}{hint}")
        return

    report.ok(f"{label} 파일", str(path))

    try:
        frame, available, total_rows = read_sample(
            path=path,
            columns=REQUIRED_COLUMNS,
            sample_rows=sample_rows,
            full=full,
        )
    except Exception as error:
        report.fail(f"{label} 읽기", f"{type(error).__name__}: {error}")
        return

    missing = [
        column for column in REQUIRED_COLUMNS if column not in available
    ]

    if missing:
        report.fail(f"{label} 컬럼", f"없는 컬럼: {missing}")
        return

    report.ok(
        f"{label} 컬럼",
        f"필수 {len(REQUIRED_COLUMNS)}개 모두 존재, 전체 {total_rows:,}행",
    )

    scope = "전체" if full else f"앞 {len(frame):,}행"

    # 후보 개수와 positive 개수
    bad_count = 0
    bad_positive = 0

    for _, row in frame.iterrows():
        labels = list(row["candidate_labels"])

        if len(labels) != NUM_CANDIDATES:
            bad_count += 1

        if sum(1 for value in labels if float(value) == 1.0) != NUM_POSITIVES:
            bad_positive += 1

    if bad_count:
        report.fail(
            f"{label} 후보 수",
            f"{scope} 중 {bad_count}행이 {NUM_CANDIDATES}개가 아닙니다.",
        )
    else:
        report.ok(f"{label} 후보 수", f"{scope} 모두 {NUM_CANDIDATES}개")

    if bad_positive:
        report.fail(
            f"{label} positive 수",
            f"{scope} 중 {bad_positive}행이 positive {NUM_POSITIVES}개가 아닙니다.",
        )
    else:
        report.ok(f"{label} positive 수", f"{scope} 모두 {NUM_POSITIVES}개")

    # SID vocab 범위
    column_to_vocab = {
        "history_c1": "c1_vocab_size",
        "history_c2": "c2_vocab_size",
        "history_c3": "c3_vocab_size",
        "history_c4": "c4_vocab_size",
        "candidate_c1": "c1_vocab_size",
        "candidate_c2": "c2_vocab_size",
        "candidate_c3": "c3_vocab_size",
    }

    for column, vocab_key in column_to_vocab.items():
        vocab_size = vocab_sizes.get(vocab_key)

        if vocab_size is None:
            report.warn(
                f"{label} {column}",
                f"{vocab_key}를 config에서 읽지 못해 범위를 확인하지 못했습니다.",
            )
            continue

        observed_max = -1
        observed_min = None

        for value in frame[column]:
            for item in value:
                item = int(item)

                if item > observed_max:
                    observed_max = item

                if observed_min is None or item < observed_min:
                    observed_min = item

        if observed_max < 0:
            report.warn(f"{label} {column}", "값이 없습니다.")
            continue

        if observed_min is not None and observed_min < 0:
            report.fail(
                f"{label} {column}",
                f"음수 값이 있습니다 (min={observed_min}).",
            )
            continue

        if observed_max >= vocab_size:
            report.fail(
                f"{label} {column}",
                f"{scope} 최댓값 {observed_max} >= {vocab_key} {vocab_size}. "
                "학습 중 embedding index 오류가 납니다.",
            )
        else:
            report.ok(
                f"{label} {column}",
                f"{scope} 범위 [{observed_min}, {observed_max}] < {vocab_size}",
            )


def check_datasets_link(report: Report) -> None:
    # 공용 서버에서는 데이터를 각자 폴더에 복사하지 않고
    # Transformer/datasets를 공용 폴더로 연결해서 쓴다.
    #
    # 링크가 없으면 gin의 상대경로가 풀리지 않는다.
    # "파일을 찾을 수 없습니다"만 보고는 원인을 알기 어려우므로
    # 링크 상태를 따로 알려준다.
    link = BASE_DIR / "datasets"

    if link.is_symlink():
        target = link.resolve()

        if target.exists():
            report.ok("datasets 링크", f"-> {target}")
        else:
            report.fail(
                "datasets 링크",
                f"링크가 가리키는 곳이 없습니다: {target}",
            )

        return

    if link.is_dir():
        report.ok("datasets 폴더", f"{link} (링크가 아닌 실제 폴더)")
        return

    report.warn(
        "datasets",
        f"{link}가 없습니다. gin이 상대경로를 쓴다면 아래를 실행하세요:\n"
        f"           ln -s ~/shared/datasets {link}",
    )


def check_architecture(report: Report, bindings: Dict[str, Any]) -> None:
    report.section("9. 모델 설정")

    d_model = bindings.get("NewsEncoderDecoderTransformer.d_model")
    num_heads = bindings.get("NewsEncoderDecoderTransformer.num_heads")

    if d_model is None or num_heads is None:
        report.warn(
            "d_model % num_heads",
            "config에서 값을 읽지 못했습니다.",
        )
        return

    if d_model % num_heads != 0:
        report.fail(
            "d_model % num_heads",
            f"d_model={d_model}이 num_heads={num_heads}로 나누어떨어지지 않습니다.",
        )
    else:
        report.ok(
            "d_model % num_heads",
            f"d_model={d_model}, num_heads={num_heads}",
        )


def check_output_dir(report: Report, out_dir: Path) -> None:
    report.section("10~11. 출력 폴더와 디스크")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report.ok("출력 폴더 쓰기", str(out_dir))

    except OSError as error:
        report.fail("출력 폴더 쓰기", f"{out_dir}: {error}")
        return

    usage = shutil.disk_usage(str(out_dir))
    free_gib = usage.free / (1024 ** 3)

    if free_gib < MIN_FREE_GIB:
        report.warn(
            "디스크 여유",
            f"{free_gib:.1f} GiB 남았습니다. checkpoint 저장이 실패할 수 있습니다.",
        )
    else:
        report.ok("디스크 여유", f"{free_gib:.1f} GiB")


def check_forward(report: Report, config_path: Path) -> None:
    report.section("12. 합성 데이터 forward와 loss")

    try:
        import gin
        import torch

        from modules.loss import TransformerLoss
        from modules.model import NewsEncoderDecoderTransformer

    except ImportError as error:
        report.fail("모델 import", str(error))
        return

    try:
        gin.parse_config_file(str(config_path), skip_unknown=True)
        model = NewsEncoderDecoderTransformer()
        loss_fn = TransformerLoss()

    except Exception as error:
        report.fail("모델 생성", f"{type(error).__name__}: {error}")
        return

    total_parameters = sum(p.numel() for p in model.parameters())
    report.ok("모델 생성", f"파라미터 {total_parameters:,}개")

    try:
        batch_size = 2
        history_length = 3

        history_sids = torch.stack(
            [
                torch.randint(0, model.c1_vocab_size, (batch_size, history_length)),
                torch.randint(0, model.c2_vocab_size, (batch_size, history_length)),
                torch.randint(0, model.c3_vocab_size, (batch_size, history_length)),
                torch.randint(0, model.c4_vocab_size, (batch_size, history_length)),
            ],
            dim=-1,
        )

        history_mask = torch.ones(
            (batch_size, history_length),
            dtype=torch.long,
        )

        # 후보 5개의 (c1,c2,c3)는 서로 달라야 한다
        candidate_sids = torch.stack(
            [
                torch.stack(
                    [
                        torch.tensor(
                            [
                                index % model.c1_vocab_size,
                                index % model.c2_vocab_size,
                                index % model.c3_vocab_size,
                            ]
                        )
                        for index in range(NUM_CANDIDATES)
                    ]
                )
                for _ in range(batch_size)
            ]
        )

        candidate_labels = torch.zeros(
            (batch_size, NUM_CANDIDATES),
            dtype=torch.float32,
        )
        candidate_labels[:, 0] = 1.0

        model.eval()

        with torch.no_grad():
            output = model(
                history_sids=history_sids,
                history_mask=history_mask,
                candidate_sids=candidate_sids,
            )

            loss_output = loss_fn(
                candidate_scores=output.candidate_scores,
                candidate_labels=candidate_labels,
            )

        score_shape = tuple(output.candidate_scores.shape)

        if score_shape != (batch_size, NUM_CANDIDATES):
            report.fail(
                "forward 출력 shape",
                f"{score_shape} != {(batch_size, NUM_CANDIDATES)}",
            )
            return

        report.ok("forward", f"candidate_scores shape {score_shape}")
        report.ok("loss", f"{float(loss_output.total_loss.item()):.6f}")

    except Exception as error:
        report.fail("forward/loss", f"{type(error).__name__}: {error}")


def resolve_path(value: str) -> Path:
    # "~/shared/datasets/..." 표기를 지원한다
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sweep 실행 전 환경과 데이터 점검",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default="sweep_out/preflight",
        help="쓰기 권한과 디스크 용량을 확인할 폴더",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2000,
        help="기본 검사에서 확인할 행 수",
    )
    parser.add_argument(
        "--full-data-check",
        action="store_true",
        help="전체 데이터의 min/max와 label 구조를 확인한다 (느림).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = resolve_path(args.config)

    if not config_path.exists():
        print(f"gin config를 찾을 수 없습니다: {config_path}")
        return 1

    bindings = load_base_bindings(config_path)

    report = Report()

    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)
    print(f"Config : {config_path}")
    print(
        "Data   : "
        + ("전체 검사" if args.full_data_check else f"앞 {args.sample_rows:,}행 검사")
    )

    check_imports(report)

    check_cuda(
        report,
        amp_dtype=str(bindings.get("train.amp_dtype", "bfloat16")),
    )

    vocab_sizes = {
        key: bindings.get(f"NewsEncoderDecoderTransformer.{key}")
        for key in (
            "c1_vocab_size",
            "c2_vocab_size",
            "c3_vocab_size",
            "c4_vocab_size",
        )
    }

    report.section("4~8. 데이터")

    check_datasets_link(report)

    for label, binding_key in (
        ("Train", "train.train_path"),
        ("Validation", "train.validation_path"),
    ):
        raw_path = bindings.get(binding_key)

        if raw_path is None:
            report.fail(f"{label} 경로", f"config에 {binding_key}가 없습니다.")
            continue

        check_dataset(
            report=report,
            label=label,
            path=resolve_path(str(raw_path)),
            vocab_sizes=vocab_sizes,
            sample_rows=args.sample_rows,
            full=args.full_data_check,
        )

    check_architecture(report, bindings)
    check_output_dir(report, resolve_path(args.out))
    check_forward(report, config_path)

    print()
    print("=" * 78)

    if report.warnings:
        print(f"경고 {len(report.warnings)}개:")
        for warning in report.warnings:
            print(f"  - {warning}")
        print()

    if report.failures:
        print(f"PREFLIGHT FAILED — 문제 {len(report.failures)}개")
        for failure in report.failures:
            print(f"  - {failure}")
        print("=" * 78)
        return 1

    print("PREFLIGHT PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
