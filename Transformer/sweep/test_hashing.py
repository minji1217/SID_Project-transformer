"""
run hash가 학습 결과를 바꾸는 값에만 반응하는지 검사한다.

    cd Transformer
    python -m sweep.test_hashing
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from sweep.hashing import (
    HASH_EXCLUDED_BINDINGS,
    compute_config_hash,
    load_base_bindings,
    parse_gin_bindings,
    strip_defaults,
)
from sweep.testing import Checker


BASE_CONFIG_TEXT = """
NewsSequenceDataset.max_history_length = 20
NewsEncoderDecoderTransformer.d_model = 384
NewsEncoderDecoderTransformer.num_heads = 6
NewsEncoderDecoderTransformer.use_sep = True

train.learning_rate = 0.0001   # 주석은 무시되어야 한다
train.batch_size = 128
train.num_epochs = 30
train.early_stopping_patience = 5
train.seed = 42
train.save_dir = "out/ebnerd"
"""

BASELINE_BINDINGS: Dict[str, Any] = {
    "train.learning_rate": 0.0002,
    "train.num_epochs": 12,
    "train.early_stopping_patience": 3,
    "train.seed": 42,
    "train.save_every_epoch": False,
    "train.save_optimizer_state": False,
}


def write_config(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    checker = Checker("Hash 테스트")

    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)

        base_config = write_config(
            directory, "base.gin", BASE_CONFIG_TEXT
        )

        baseline = compute_config_hash(base_config, BASELINE_BINDINGS)

        # --------------------------------------------------------------
        checker.section("gin 파일 파싱")

        parsed = load_base_bindings(base_config)
        checker.equals(
            "주석이 붙은 값도 읽음",
            parsed.get("train.learning_rate"),
            0.0001,
        )
        checker.equals(
            "bool 값",
            parsed.get("NewsEncoderDecoderTransformer.use_sep"),
            True,
        )
        checker.equals(
            "문자열 값",
            parsed.get("train.save_dir"),
            "out/ebnerd",
        )
        checker.equals(
            "gin.config_str() 형식도 같은 파서로 처리",
            parse_gin_bindings("train.batch_size = 64").get("train.batch_size"),
            64,
        )

        # --------------------------------------------------------------
        checker.section("같은 입력이면 같은 hash")

        checker.equals(
            "재계산",
            compute_config_hash(base_config, dict(BASELINE_BINDINGS)),
            baseline,
        )
        checker.equals(
            "binding 순서가 달라도 동일",
            compute_config_hash(
                base_config,
                dict(reversed(list(BASELINE_BINDINGS.items()))),
            ),
            baseline,
        )

        # --------------------------------------------------------------
        checker.section("달라져야 하는 경우")

        changed_base = write_config(
            directory,
            "base_changed.gin",
            BASE_CONFIG_TEXT.replace(
                "NewsEncoderDecoderTransformer.d_ff = 0", ""
            ).replace("train.batch_size = 128", "train.batch_size = 256"),
        )
        checker.check(
            "base config 내용 변경",
            compute_config_hash(changed_base, BASELINE_BINDINGS) != baseline,
        )

        for name, key, value in [
            ("seed 변경", "train.seed", 123),
            ("num_epochs 변경", "train.num_epochs", 13),
            ("patience 변경", "train.early_stopping_patience", 4),
            ("learning_rate 변경", "train.learning_rate", 0.0005),
            (
                "d_model 변경",
                "NewsEncoderDecoderTransformer.d_model",
                512,
            ),
            (
                "use_sep 변경",
                "NewsEncoderDecoderTransformer.use_sep",
                False,
            ),
        ]:
            bindings = dict(BASELINE_BINDINGS)
            bindings[key] = value

            checker.check(
                name,
                compute_config_hash(base_config, bindings) != baseline,
                f"{key}={value}",
            )

        # --------------------------------------------------------------
        checker.section("같아야 하는 경우")

        for name, key, value in [
            ("save_dir", "train.save_dir", "/somewhere/else"),
            ("save_every_epoch", "train.save_every_epoch", True),
            ("save_optimizer_state", "train.save_optimizer_state", True),
        ]:
            bindings = dict(BASELINE_BINDINGS)
            bindings[key] = value

            checker.check(
                f"{name}는 hash에 영향 없음",
                compute_config_hash(base_config, bindings) == baseline,
                f"{key}={value}",
            )

        checker.equals(
            "제외 대상 목록",
            HASH_EXCLUDED_BINDINGS,
            frozenset(
                {
                    "train.save_dir",
                    "train.save_every_epoch",
                    "train.save_optimizer_state",
                }
            ),
        )

        # --------------------------------------------------------------
        checker.section("기준 config와 같은 값은 생략한 것과 동일")

        base_bindings = load_base_bindings(base_config)

        with_default = dict(BASELINE_BINDINGS)
        with_default["NewsSequenceDataset.max_history_length"] = 20

        checker.check(
            "max_history_length=20(기본값)을 명시해도 같은 hash",
            compute_config_hash(base_config, with_default) == baseline,
            "탐색 단계마다 같은 학습을 반복하지 않기 위해 필요하다.",
        )

        stripped = strip_defaults(with_default, base_bindings)
        checker.check(
            "strip_defaults가 기본값을 제거",
            "NewsSequenceDataset.max_history_length" not in stripped,
            f"stripped={sorted(stripped)}",
        )

        different_value = dict(BASELINE_BINDINGS)
        different_value["NewsSequenceDataset.max_history_length"] = 30

        checker.check(
            "기본값과 다른 값은 hash가 달라짐",
            compute_config_hash(base_config, different_value) != baseline,
        )

    return checker.finish()


if __name__ == "__main__":
    sys.exit(main())
