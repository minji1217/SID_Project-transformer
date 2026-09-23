"""
단계별 파라미터 탐색의 정의.

전체 조합을 다 도는 grid search는 불가능하다.
  4(max_history) x 2(use_sep) x 7(d_model,num_heads) x 3(num_layers)
  x 4(d_ff) x 4(lr) x 3(batch) x 4(dropout) x 3(weight_decay)
  = 96,768 가지

대신 서로 강하게 얽힌 파라미터를 같은 단계에서 함께 탐색하고,
각 단계에서 상위 2개 branch를 다음 단계로 넘긴다.
마지막 단계에서는 seed 검증용으로 Final Top-3를 남긴다.

중복 제거 전 기준 총 84 run이다.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple


# ---------------------------------------------------------------- 지표 우선순위
#
# 단계별로 상위 config를 고를 때 이 순서대로 본다.
# 앞 지표의 최고값과 tolerance 이내인 후보만 다음 지표로 내려간다.
#
# tolerance가 0이면 사실상 1순위 지표만 쓰이게 된다.
# Top-1 Accuracy는 (맞힌 수 / 전체)라 값이 촘촘해서
# 정확히 같은 값이 나오는 경우가 거의 없기 때문이다.
#
# 기본값 근거:
#   validation 5만 건, Top-1이 0.3 근처일 때
#   순수 표본오차(binomial standard error)는 약 0.002이다.
#   seed에 따른 변동은 그보다 크므로 그 2배인 0.005를 기본값으로 둔다.
#
# 실제 seed 실험으로 표준편차를 측정했다면 그 값으로 바꾼다.
# (sweep/DECISION_RULE.md 참고)
#
# 선정 우선순위에 넣지 않는 지표:
#   total_loss      lambda_preference=1이라 preference_loss와 값이 같다
#   negative_prob   positive_prob에서 계산되므로 구조적으로 중복이다
#   nDCG@10         후보가 5개라 nDCG@5와 항상 같은 값이다

class MetricRule(NamedTuple):
    key: str
    direction: str      # "max" 또는 "min"
    tolerance: float
    label: str


METRIC_PRIORITY: List[MetricRule] = [
    MetricRule("val_top1_accuracy", "max", 0.005, "Top-1 Accuracy"),
    MetricRule("val_mrr", "max", 0.003, "MRR"),
    MetricRule("val_ndcg@5", "max", 0.003, "nDCG@5"),
    MetricRule("val_auc", "max", 0.002, "AUC"),
    MetricRule("val_preference_loss", "min", 0.005, "Preference Loss"),
    MetricRule("val_positive_prob", "max", 0.002, "Positive Probability"),
    MetricRule("val_score_gap", "max", 0.0, "Positive-Negative Score Gap"),
]


# ------------------------------------------------------------------- 단계 정의

class Stage(NamedTuple):
    name: str
    description: str
    # 각 variant는 이 단계에서 시도할 gin binding 묶음이다.
    # 서로 얽힌 파라미터는 한 variant에 함께 넣어 Cartesian product로 탐색한다.
    variants: List[Dict[str, Any]]
    top_k: int


def _product(
    first_binding: str,
    first_values: List[Any],
    second_binding: str,
    second_values: List[Any],
) -> List[Dict[str, Any]]:
    # 두 파라미터의 Cartesian product
    return [
        {
            first_binding: first_value,
            second_binding: second_value,
        }
        for first_value in first_values
        for second_value in second_values
    ]


def _single(binding: str, values: List[Any]) -> List[Dict[str, Any]]:
    return [{binding: value} for value in values]


# d_model과 num_heads는 함께 정해야 한다.
# modules/model.py가 d_model % num_heads != 0이면 ValueError를 던지므로
# 3 x 3 = 9쌍 중 유효한 7쌍만 남긴다.
#   256: 4, 8      (256 % 6 != 0)
#   384: 4, 6, 8
#   512: 4, 8      (512 % 6 != 0)
_ARCH_VARIANTS: List[Dict[str, Any]] = [
    {
        "NewsEncoderDecoderTransformer.d_model": d_model,
        "NewsEncoderDecoderTransformer.num_heads": num_heads,
    }
    for d_model in (256, 384, 512)
    for num_heads in (4, 6, 8)
    if d_model % num_heads == 0
]


STAGES: List[Stage] = [
    Stage(
        name="max_history_use_sep",
        description=(
            "입력 길이 구조. history 길이와 SEP 토큰 사용 여부는 "
            "encoder 입력 시퀀스를 함께 결정하므로 같은 단계에서 탐색한다."
        ),
        variants=_product(
            "NewsSequenceDataset.max_history_length",
            [10, 20, 30, 50],
            "NewsEncoderDecoderTransformer.use_sep",
            [True, False],
        ),
        top_k=2,
    ),
    Stage(
        name="d_model_num_heads",
        description=(
            "attention 폭과 head 수. d_model % num_heads != 0이면 "
            "모델이 에러를 내므로 유효한 7쌍만 함께 탐색한다."
        ),
        variants=_ARCH_VARIANTS,
        top_k=2,
    ),
    Stage(
        name="num_layers",
        description="encoder/decoder layer 수.",
        variants=_single(
            "NewsEncoderDecoderTransformer.num_layers",
            [2, 4, 6],
        ),
        top_k=2,
    ),
    Stage(
        name="d_ff",
        description="FFN 내부 차원.",
        variants=_single(
            "NewsEncoderDecoderTransformer.d_ff",
            [768, 1024, 1536, 2048],
        ),
        top_k=2,
    ),
    Stage(
        name="learning_rate_batch_size",
        description=(
            "학습률과 batch 크기. 두 값은 학습 dynamics가 연결되어 있어 "
            "따로 정하면 잘못된 조합에 빠지므로 Cartesian product로 탐색한다."
        ),
        variants=_product(
            "train.learning_rate",
            [0.00005, 0.0001, 0.0002, 0.0005],
            "train.batch_size",
            [64, 128, 256],
        ),
        top_k=2,
    ),
    Stage(
        name="dropout_weight_decay",
        description=(
            "정규화 강도. dropout과 weight decay는 함께 과적합을 조절하므로 "
            "Cartesian product로 탐색한다. "
            "마지막 단계이므로 seed 검증용 Final Top-3를 남긴다."
        ),
        variants=_product(
            "NewsEncoderDecoderTransformer.dropout_rate",
            [0.0, 0.1, 0.2, 0.3],
            "train.weight_decay",
            [0.0, 0.01, 0.05],
        ),
        top_k=3,
    ),
]


def get_stage(stage_number: int) -> Stage:
    # stage_number는 1부터 시작한다.
    if not 1 <= stage_number <= len(STAGES):
        raise ValueError(
            f"stage must be between 1 and {len(STAGES)}. "
            f"Received: {stage_number}"
        )

    return STAGES[stage_number - 1]


def stage_dir_name(stage_number: int) -> str:
    stage = get_stage(stage_number)
    return f"stage_{stage_number:02d}_{stage.name}"


def is_final_stage(stage_number: int) -> bool:
    return stage_number == len(STAGES)


def total_run_estimate() -> int:
    # 중복 제거 전 기준 run 수
    total = 0
    carried = 1

    for stage in STAGES:
        total += carried * len(stage.variants)
        carried = min(stage.top_k, carried * len(stage.variants))

    return total
