"""
단계별로 상위 config를 고르는 규칙과, 사람이 확인해야 할 신호를 찾는 검사.

선택 규칙은 sweep/stages.py의 METRIC_PRIORITY를 따른다.
앞 지표의 차이가 tolerance 이내면 동률로 보고 다음 지표로 내려간다.

tolerance를 쓰기 때문에 "A와 B는 동률, B와 C도 동률, 그런데 A와 C는 아님"
같은 경우가 생길 수 있다. 이런 관계로는 정렬을 정의할 수 없으므로,
전체를 한 번에 정렬하지 않고 1등을 하나 뽑아 빼내는 것을 반복한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sweep.stages import (
    FINAL_TIEBREAK_KEY,
    METRIC_PRIORITY,
    SELECTION_RULES,
    MetricRule,
)


STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"


class NotEnoughSuccessfulRuns(RuntimeError):
    """성공한 run이 top_k보다 적어 자동 선택을 할 수 없는 경우."""


# 후보 5개 중 정답 1개이므로 아무것도 학습하지 못한 모델의 기댓값
RANDOM_TOP1 = 0.2
RANDOM_AUC = 0.5

# 과적합으로 볼 train-validation Top-1 격차
OVERFIT_GAP = 0.15


def _metric_value(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    # NaN은 비교에서 제외한다
    if value != value:
        return None

    return value


def _best_value(
    rows: List[Dict[str, Any]],
    rule: MetricRule,
) -> Optional[float]:
    values = [
        value
        for value in (_metric_value(row, rule.key) for row in rows)
        if value is not None
    ]

    if not values:
        return None

    return max(values) if rule.direction == "max" else min(values)


def _within_tolerance(
    value: Optional[float],
    best: float,
    rule: MetricRule,
) -> bool:
    if value is None:
        return False

    if rule.direction == "max":
        return value >= best - rule.tolerance

    return value <= best + rule.tolerance


def pick_winner(
    rows: List[Dict[str, Any]],
    rules: Optional[List[MetricRule]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    # 지표 우선순위를 따라 내려가며 후보를 좁힌다.
    # 어느 지표에서 승부가 났는지도 함께 돌려준다.
    #
    # rules를 넘기면 다른 우선순위를 쓸 수 있다.
    # seed 검증에서는 지표 평균과 표준편차로 비교해야 하므로 필요하다.
    if rules is None:
        rules = SELECTION_RULES

    pool = list(rows)
    trace: List[str] = []

    for rule in rules:
        if len(pool) <= 1:
            break

        best = _best_value(pool, rule)

        if best is None:
            trace.append(f"{rule.label}: 값 없음, 건너뜀")
            continue

        narrowed = [
            row
            for row in pool
            if _within_tolerance(
                _metric_value(row, rule.key),
                best,
                rule,
            )
        ]

        if not narrowed:
            trace.append(f"{rule.label}: 비교 불가, 건너뜀")
            continue

        if len(narrowed) < len(pool):
            trace.append(
                f"{rule.label}={best:.6f} 기준 "
                f"{len(pool)}개 -> {len(narrowed)}개"
            )
        else:
            trace.append(
                f"{rule.label}: {len(pool)}개가 "
                f"tolerance {rule.tolerance} 이내로 동률"
            )

        pool = narrowed

    if len(pool) > 1:
        # 마지막 결정은 config_hash 오름차순으로 한다.
        # 실행 순서에 의존하지 않으므로 같은 입력이면 항상 같은 결과가 나온다.
        with_hash = [
            candidate
            for candidate in pool
            if candidate.get(FINAL_TIEBREAK_KEY)
        ]

        if with_hash:
            winner = min(
                with_hash,
                key=lambda candidate: str(candidate[FINAL_TIEBREAK_KEY]),
            )

            trace.append(
                f"{len(pool)}개가 끝까지 동률이라 "
                f"{FINAL_TIEBREAK_KEY} 오름차순으로 결정 "
                f"({winner[FINAL_TIEBREAK_KEY]})"
            )

            return winner, trace

        trace.append(
            f"마지막까지 {len(pool)}개가 동률이고 "
            f"{FINAL_TIEBREAK_KEY}도 없어 먼저 실행된 run을 선택"
        )

    return pool[0], trace


def rank_rows(
    rows: List[Dict[str, Any]],
    rules: Optional[List[MetricRule]] = None,
) -> List[Tuple[Dict[str, Any], List[str]]]:
    # 1등을 뽑아 빼내는 것을 반복해 전체 순위를 만든다.
    remaining = list(rows)
    ordered: List[Tuple[Dict[str, Any], List[str]]] = []

    while remaining:
        winner, trace = pick_winner(remaining, rules=rules)
        ordered.append((winner, trace))
        remaining = [row for row in remaining if row is not winner]

    return ordered


def check_row_warnings(row: Dict[str, Any]) -> List[str]:
    # run 하나에 대한 경고. 자동 선택을 막지는 않고 요약에 남긴다.
    warnings: List[str] = []

    best_epoch = row.get("best_epoch")
    epochs_run = row.get("num_epochs_run")
    stopped_early = row.get("stopped_early")

    if isinstance(best_epoch, int) and best_epoch <= 2:
        warnings.append(
            f"best_epoch={best_epoch}: 학습이 거의 진행되지 않았다"
        )

    if (
        isinstance(best_epoch, int)
        and isinstance(epochs_run, int)
        and best_epoch == epochs_run
        and not stopped_early
        and epochs_run > 0
    ):
        warnings.append(
            f"마지막 epoch({epochs_run})가 best: "
            "아직 개선 중이므로 epoch 수가 부족하다"
        )

    val_top1 = _metric_value(row, "val_top1_accuracy")

    if val_top1 is not None and val_top1 <= RANDOM_TOP1:
        warnings.append(
            f"val_top1_accuracy={val_top1:.4f}: "
            f"무작위 기준({RANDOM_TOP1}) 이하"
        )

    val_auc = _metric_value(row, "val_auc")

    if val_auc is not None and val_auc <= RANDOM_AUC:
        warnings.append(
            f"val_auc={val_auc:.4f}: 무작위 기준({RANDOM_AUC}) 이하"
        )

    train_top1 = _metric_value(row, "train_top1_accuracy")

    if (
        train_top1 is not None
        and val_top1 is not None
        and train_top1 - val_top1 > OVERFIT_GAP
    ):
        warnings.append(
            f"train-val Top-1 격차 {train_top1 - val_top1:.4f}: 과적합 의심"
        )

    return warnings


def check_stage_warnings(
    rows: List[Dict[str, Any]],
    winner: Dict[str, Any],
) -> List[str]:
    # 단계 전체에 대한 경고.
    # 사람이 CSV를 직접 봐야 하는 상황을 알린다.
    warnings: List[str] = []

    if len(rows) < 2:
        return warnings

    primary = METRIC_PRIORITY[0]
    values = [
        value
        for value in (_metric_value(row, primary.key) for row in rows)
        if value is not None
    ]

    # 이 파라미터가 성능에 영향이 없는 경우
    if values and (max(values) - min(values)) <= primary.tolerance:
        warnings.append(
            f"{primary.label}의 1등과 꼴등 차이가 "
            f"{max(values) - min(values):.6f}로 "
            f"tolerance({primary.tolerance}) 이내다. "
            "이 파라미터는 영향이 없을 수 있으니 기본값 유지를 검토할 것."
        )

    # 1순위 지표의 승자와 다른 지표의 승자가 엇갈리는 경우
    for rule in METRIC_PRIORITY[1:4]:
        best = _best_value(rows, rule)

        if best is None:
            continue

        winner_value = _metric_value(winner, rule.key)

        if winner_value is None:
            continue

        if not _within_tolerance(winner_value, best, rule):
            warnings.append(
                f"선택된 config의 {rule.label}={winner_value:.6f}이(가) "
                f"이 단계 최고값 {best:.6f}보다 낮다. "
                "지표 간 판단이 엇갈리므로 확인이 필요하다."
            )

    return warnings


def is_successful(row: Dict[str, Any]) -> bool:
    return str(row.get("status", "")).upper() == STATUS_SUCCESS


def split_by_status(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # 실패한 run은 순위 선정에서 제외한다.
    # 성능이 나빠서가 아니라 측정 자체가 안 된 것이므로
    # 다른 run과 같은 기준으로 비교할 수 없다.
    successful = [row for row in rows if is_successful(row)]
    failed = [row for row in rows if not is_successful(row)]

    return successful, failed


def select_stage_top_k(
    rows: List[Dict[str, Any]],
    top_k: int,
) -> Tuple[
    List[Tuple[Dict[str, Any], List[str]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    # 단계 결과에서 상위 top_k개를 고른다.
    #
    # 돌려주는 값:
    #   ordered  성공한 run의 순위와 판단 근거
    #   failed   실패한 run
    #   selected 다음 단계로 넘길 상위 top_k개
    successful, failed = split_by_status(rows)

    if len(successful) < top_k:
        raise NotEnoughSuccessfulRuns(
            f"성공한 run이 {len(successful)}개뿐이라 "
            f"상위 {top_k}개를 고를 수 없습니다. "
            f"(실패 {len(failed)}개)\n"
            "실패한 run의 _FAILED.json과 train_log.txt를 확인한 뒤 "
            "--retry-failed로 다시 실행하세요."
        )

    ordered = rank_rows(successful)
    selected = [row for row, _ in ordered[:top_k]]

    return ordered, failed, selected


def filter_complete_seed_configs(
    rows: List[Dict[str, Any]],
    expected_seed_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # seed 검증에서는 모든 seed가 성공한 설정만 비교한다.
    #
    # seed 2개만 성공한 설정과 3개가 성공한 설정을 같이 비교하면,
    # 우연히 나쁜 seed가 실패한 설정이 평균에서 유리해진다.
    # 평균과 표준편차를 비교하려면 표본 수가 같아야 한다.
    complete = [
        row
        for row in rows
        if row.get("successful_seed_count") == expected_seed_count
    ]

    incomplete = [
        row
        for row in rows
        if row.get("successful_seed_count") != expected_seed_count
    ]

    return complete, incomplete
