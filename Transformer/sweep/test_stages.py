"""
Stage 구성이 의도대로 생성되는지 검사한다.

    cd Transformer
    python -m sweep.test_stages
"""

from __future__ import annotations

import sys

from sweep.stages import STAGES, get_stage, is_final_stage, total_run_estimate
from sweep.testing import Checker


EXPECTED_VARIANT_COUNTS = [8, 7, 3, 4, 12, 12]
EXPECTED_TOP_K = [2, 2, 2, 2, 2, 3]

D_MODEL_KEY = "NewsEncoderDecoderTransformer.d_model"
NUM_HEADS_KEY = "NewsEncoderDecoderTransformer.num_heads"


def main() -> int:
    checker = Checker("Stage 구성 테스트")

    checker.section("Stage 수와 변형 수")
    checker.equals("Stage 수", len(STAGES), 6)

    for index, expected in enumerate(EXPECTED_VARIANT_COUNTS, 1):
        stage = get_stage(index)
        checker.equals(
            f"Stage {index} ({stage.name}) variants",
            len(stage.variants),
            expected,
        )

    checker.section("top_k")

    for index, expected in enumerate(EXPECTED_TOP_K, 1):
        stage = get_stage(index)
        checker.equals(f"Stage {index} top_k", stage.top_k, expected)

    checker.check(
        "Stage 6이 마지막 단계",
        is_final_stage(6) and not is_final_stage(5),
    )

    checker.section("총 run 수")
    checker.equals("total_run_estimate()", total_run_estimate(), 84)

    checker.section("Architecture 조합")
    arch_variants = get_stage(2).variants
    pairs = [
        (variant[D_MODEL_KEY], variant[NUM_HEADS_KEY])
        for variant in arch_variants
    ]

    checker.check(
        "(256, 6) 없음",
        (256, 6) not in pairs,
        f"pairs={pairs}",
    )
    checker.check(
        "(512, 6) 없음",
        (512, 6) not in pairs,
        f"pairs={pairs}",
    )
    checker.check(
        "모든 조합에서 d_model % num_heads == 0",
        all(d_model % num_heads == 0 for d_model, num_heads in pairs),
        f"pairs={pairs}",
    )
    checker.equals("유효 조합 수", len(pairs), 7)

    checker.section("Stage별 gin binding 이름")

    expected_bindings = {
        1: {
            "NewsSequenceDataset.max_history_length",
            "NewsEncoderDecoderTransformer.use_sep",
        },
        2: {D_MODEL_KEY, NUM_HEADS_KEY},
        3: {"NewsEncoderDecoderTransformer.num_layers"},
        4: {"NewsEncoderDecoderTransformer.d_ff"},
        5: {"train.learning_rate", "train.batch_size"},
        6: {
            "NewsEncoderDecoderTransformer.dropout_rate",
            "train.weight_decay",
        },
    }

    for stage_number, expected in expected_bindings.items():
        stage = get_stage(stage_number)
        actual = set()

        for variant in stage.variants:
            actual.update(variant.keys())

        checker.equals(f"Stage {stage_number} binding", actual, expected)

    checker.section("Stage별 값 범위")

    history_values = sorted(
        {
            variant["NewsSequenceDataset.max_history_length"]
            for variant in get_stage(1).variants
        }
    )
    checker.equals("max_history_length", history_values, [10, 20, 30, 50])

    sep_values = sorted(
        {
            variant["NewsEncoderDecoderTransformer.use_sep"]
            for variant in get_stage(1).variants
        }
    )
    checker.equals("use_sep", sep_values, [False, True])

    lr_values = sorted(
        {variant["train.learning_rate"] for variant in get_stage(5).variants}
    )
    checker.equals(
        "learning_rate",
        lr_values,
        [0.00005, 0.0001, 0.0002, 0.0005],
    )

    batch_values = sorted(
        {variant["train.batch_size"] for variant in get_stage(5).variants}
    )
    checker.equals("batch_size", batch_values, [64, 128, 256])

    dropout_values = sorted(
        {
            variant["NewsEncoderDecoderTransformer.dropout_rate"]
            for variant in get_stage(6).variants
        }
    )
    checker.equals("dropout_rate", dropout_values, [0.0, 0.1, 0.2, 0.3])

    wd_values = sorted(
        {variant["train.weight_decay"] for variant in get_stage(6).variants}
    )
    checker.equals("weight_decay", wd_values, [0.0, 0.01, 0.05])

    return checker.finish()


if __name__ == "__main__":
    sys.exit(main())
