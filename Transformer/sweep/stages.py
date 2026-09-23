"""
단계별 파라미터 탐색의 정의.

전체 조합을 다 도는 grid search는 96,768가지라 불가능하다.
  4(lr) x 7(d_model,num_heads) x 3(num_layers) x 4(dropout)
  x 4(max_history) x 3(batch_size) x 4(d_ff) x 3(weight_decay) x 2(use_sep)

대신 영향이 큰 파라미터부터 순서대로 탐색하고,
각 단계에서 상위 top_k개를 고정한 뒤 다음 단계로 넘어간다 (총 62 run).

각 단계에서 이전 단계가 남긴 config를 그대로 이어받기 때문에,
단계 순서가 결과에 영향을 준다.
그래서 영향이 크다고 알려진 learning_rate와 max_history_length를 앞에 두고,
top_k=2로 두 갈래를 들고 가서 한 번의 잘못된 선택으로 전체가 틀어지지 않게 한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple


# ---------------------------------------------------------------- 지표 우선순위
#
# 단계별로 상위 config를 고를 때 이 순서대로 본다.
# 앞 지표의 차이가 tolerance 이내면 "동률"로 보고 다음 지표로 내려간다.
#
# tolerance가 0이면 사실상 1순위 지표만 쓰이게 된다.
# Top-1 Accuracy는 (맞힌 수 / 전체)라 값이 촘촘해서
# 정확히 같은 값이 나오는 경우가 거의 없기 때문이다.
#
# 기본값 근거:
#   validation 5만 건, Top-1이 0.3 근처일 때
#   순수 표본오차(binomial standard error)는 약 0.002이다.
#   seed에 따른 변동은 그보다 크므로 그 2배 정도인 0.005를 기본값으로 둔다.
#
# 실제 seed 실험으로 표준편차를 측정했다면 그 값으로 바꾼다.
# (sweep/DECISION_RULE.md 참고)

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
    # d_model과 num_heads처럼 서로 얽힌 파라미터는 한 variant에 함께 넣는다.
    variants: List[Dict[str, Any]]
    top_k: int


def _single(binding: str, values: List[Any]) -> List[Dict[str, Any]]:
    return [{binding: value} for value in values]


# d_model과 num_heads는 함께 정해야 한다.
# modules/model.py가 d_model % num_heads != 0이면 ValueError를 던지므로
# 3 x 3 = 9쌍 중 유효한 7쌍만 남긴다.
#   256: 4, 8      (6은 나누어떨어지지 않음)
#   384: 4, 6, 8
#   512: 4, 8      (6은 나누어떨어지지 않음)
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
        name="learning_rate",
        description="학습률. 영향이 가장 커서 가장 먼저 정한다.",
        variants=_single(
            "train.learning_rate",
            [0.00005, 0.0001, 0.0002, 0.0005],
        ),
        top_k=2,
    ),
    Stage(
        name="max_history_length",
        description="사용할 최근 기사 수. 입력 길이가 바뀌어 영향이 크다.",
        variants=_single(
            "NewsSequenceDataset.max_history_length",
            [10, 20, 30, 50],
        ),
        top_k=2,
    ),
    Stage(
        name="d_model_num_heads",
        description=(
            "임베딩 차원과 attention head 수. "
            "d_model % num_heads != 0이면 모델이 에러를 내므로 함께 탐색한다."
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
        name="dropout",
        description="dropout 비율. 앞 단계에서 정해진 모델 크기에 맞춰 조절한다.",
        variants=_single(
            "NewsEncoderDecoderTransformer.dropout_rate",
            [0.0, 0.1, 0.2, 0.3],
        ),
        top_k=2,
    ),
    Stage(
        name="weight_decay",
        description="정규화 강도. dropout과 함께 과적합을 조절한다.",
        variants=_single(
            "train.weight_decay",
            [0.0, 0.01, 0.05],
        ),
        top_k=2,
    ),
    Stage(
        name="batch_size",
        description=(
            "batch 크기. learning_rate와 얽혀 있어 뒤쪽에 둔다. "
            "여기서부터는 갈래를 하나로 좁힌다."
        ),
        variants=_single(
            "train.batch_size",
            [64, 128, 256],
        ),
        top_k=1,
    ),
    Stage(
        name="use_sep",
        description="history 기사 사이 SEP 토큰 사용 여부. ablation 성격.",
        variants=_single(
            "NewsEncoderDecoderTransformer.use_sep",
            [True, False],
        ),
        top_k=1,
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


def total_run_estimate() -> int:
    # 중복 제거 전 기준 run 수
    total = 0
    carried = 1

    for stage in STAGES:
        total += carried * len(stage.variants)
        carried = min(stage.top_k, carried * len(stage.variants))

    return total
