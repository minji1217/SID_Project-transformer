"""
지표 우선순위에 따른 선택 규칙을 검사한다.

    cd Transformer
    python -m sweep.test_selection
"""

from __future__ import annotations

import sys

from sweep.selection import (
    NotEnoughSuccessfulRuns,
    check_stage_warnings,
    filter_complete_seed_configs,
    pick_winner,
    rank_rows,
    select_stage_top_k,
    split_by_status,
)
from sweep.testing import Checker


def row(name: str, **metrics) -> dict:
    # 지정하지 않은 지표는 모든 후보가 같은 값을 갖게 해서
    # 검사하려는 지표에서만 승부가 나도록 한다.
    base = {
        "status": "SUCCESS",
        "config": name,
        "val_top1_accuracy": 0.300,
        "val_mrr": 0.500,
        "val_ndcg@5": 0.600,
        "val_auc": 0.700,
        "val_preference_loss": 1.000,
        "val_positive_prob": 0.300,
        "val_score_gap": 1.000,
    }
    base.update(metrics)

    return base


def main() -> int:
    checker = Checker("선택 규칙 테스트")

    # ------------------------------------------------------------------
    checker.section("1. Top-1 차이가 tolerance(0.005)보다 크면 Top-1이 결정")

    rows = [
        row("A", val_top1_accuracy=0.310, val_mrr=0.400),
        row("B", val_top1_accuracy=0.300, val_mrr=0.900),
    ]
    winner, trace = pick_winner(rows)
    checker.equals("승자", winner["config"], "A")
    checker.check(
        "MRR이 훨씬 높아도 Top-1이 이김",
        winner["config"] == "A",
        f"trace={trace}",
    )

    # ------------------------------------------------------------------
    checker.section("2. Top-1 차이가 tolerance 이내면 MRR이 결정")

    rows = [
        row("A", val_top1_accuracy=0.300, val_mrr=0.500),
        row("B", val_top1_accuracy=0.298, val_mrr=0.550),
    ]
    winner, trace = pick_winner(rows)
    checker.equals("승자", winner["config"], "B")
    checker.check(
        "판단 근거에 MRR이 남음",
        any("MRR" in line for line in trace),
        f"trace={trace}",
    )

    # ------------------------------------------------------------------
    checker.section("3. MRR도 tolerance(0.003) 이내면 nDCG@5가 결정")

    rows = [
        row("A", val_top1_accuracy=0.300, val_mrr=0.5000, **{"val_ndcg@5": 0.600}),
        row("B", val_top1_accuracy=0.298, val_mrr=0.4985, **{"val_ndcg@5": 0.650}),
    ]
    winner, trace = pick_winner(rows)
    checker.equals("승자", winner["config"], "B")
    checker.check(
        "판단 근거에 nDCG@5가 남음",
        any("nDCG@5" in line for line in trace),
        f"trace={trace}",
    )

    # ------------------------------------------------------------------
    checker.section("4. Preference Loss는 작을수록 좋게 비교")

    rows = [
        row("A", val_preference_loss=1.000),
        row("B", val_preference_loss=0.900),
    ]
    winner, _ = pick_winner(rows)
    checker.equals("승자", winner["config"], "B")

    rows = [
        row("A", val_preference_loss=0.900),
        row("B", val_preference_loss=1.000),
    ]
    winner, _ = pick_winner(rows)
    checker.equals("반대 배치에서도 승자", winner["config"], "A")

    # ------------------------------------------------------------------
    checker.section("5. NaN과 누락 지표 처리")

    missing = row("MISSING")
    del missing["val_top1_accuracy"]

    rows = [
        row("GOOD", val_top1_accuracy=0.300),
        row("NAN", val_top1_accuracy=float("nan")),
        missing,
    ]
    ordered = rank_rows(rows)
    names = [entry[0]["config"] for entry in ordered]

    checker.equals("1등", names[0], "GOOD")
    checker.equals("전체 수", len(names), 3)
    checker.check(
        "NaN과 누락 row도 순위에서 사라지지 않음",
        set(names) == {"GOOD", "NAN", "MISSING"},
        f"names={names}",
    )

    # ------------------------------------------------------------------
    checker.section("6. 실패한 run은 순위 선정에서 제외")

    rows = [
        row("A", val_top1_accuracy=0.300),
        {"status": "FAILED", "config": "OOM", "error_type": "CUDA_OOM"},
        row("C", val_top1_accuracy=0.280),
    ]
    successful, failed = split_by_status(rows)
    checker.equals("성공 수", len(successful), 2)
    checker.equals("실패 수", len(failed), 1)

    ordered, failed_rows, selected = select_stage_top_k(rows, 2)
    checker.equals(
        "순위",
        [entry[0]["config"] for entry in ordered],
        ["A", "C"],
    )
    checker.equals(
        "선택",
        [entry["config"] for entry in selected],
        ["A", "C"],
    )
    checker.equals("실패 목록", [entry["config"] for entry in failed_rows], ["OOM"])

    # ------------------------------------------------------------------
    checker.section("7. 성공 run이 top_k보다 적으면 오류")

    raised = False
    message = ""

    try:
        select_stage_top_k(rows, 3)
    except NotEnoughSuccessfulRuns as error:
        raised = True
        message = str(error).splitlines()[0]

    checker.check("NotEnoughSuccessfulRuns 발생", raised, message)

    # top_k와 성공 수가 같으면 통과해야 한다
    ordered, _, selected = select_stage_top_k(rows, 2)
    checker.equals("top_k == 성공 수", len(selected), 2)

    # ------------------------------------------------------------------
    checker.section("8. 지표가 모두 같으면 모델 파라미터 수가 작은 쪽")

    rows = [
        row("BIG", total_parameters=20_000_000, config_hash="aaa"),
        row("SMALL", total_parameters=10_000_000, config_hash="zzz"),
    ]
    winner, trace = pick_winner(rows)
    checker.equals("승자", winner["config"], "SMALL")
    checker.check(
        "판단 근거에 파라미터 수가 남음",
        any("파라미터" in line for line in trace),
        f"trace={trace[-1:]}",
    )

    # ------------------------------------------------------------------
    checker.section("9. 파라미터 수도 같으면 epoch당 학습 시간이 짧은 쪽")

    rows = [
        row(
            "SLOW",
            total_parameters=10_000_000,
            mean_epoch_seconds=200.0,
            config_hash="aaa",
        ),
        row(
            "FAST",
            total_parameters=10_000_000,
            mean_epoch_seconds=100.0,
            config_hash="zzz",
        ),
    ]
    winner, trace = pick_winner(rows)
    checker.equals("승자", winner["config"], "FAST")

    # ------------------------------------------------------------------
    checker.section("10. 시간도 같으면 GPU 메모리를 덜 쓰는 쪽")

    rows = [
        row(
            "HEAVY",
            total_parameters=10_000_000,
            mean_epoch_seconds=100.0,
            peak_gpu_memory_mb=9000.0,
            config_hash="aaa",
        ),
        row(
            "LIGHT",
            total_parameters=10_000_000,
            mean_epoch_seconds=100.0,
            peak_gpu_memory_mb=4000.0,
            config_hash="zzz",
        ),
    ]
    winner, trace = pick_winner(rows)
    checker.equals("승자", winner["config"], "LIGHT")

    # ------------------------------------------------------------------
    checker.section("11. 전부 같으면 config_hash 오름차순")

    def identical(name: str, config_hash: str) -> dict:
        return row(
            name,
            total_parameters=10_000_000,
            mean_epoch_seconds=100.0,
            peak_gpu_memory_mb=4000.0,
            config_hash=config_hash,
        )

    rows = [
        identical("Z", "zzz111"),
        identical("A", "aaa999"),
        identical("M", "mmm555"),
    ]
    winner, trace = pick_winner(rows)
    checker.equals("승자", winner["config"], "A")
    checker.check(
        "판단 근거에 config_hash가 남음",
        any("config_hash" in line for line in trace),
        f"trace={trace[-1:]}",
    )

    # 입력 순서를 바꿔도 결과가 같아야 한다
    reversed_rows = list(reversed(rows))
    reversed_winner, _ = pick_winner(reversed_rows)
    checker.equals(
        "입력 순서를 바꿔도 같은 승자",
        reversed_winner["config"],
        "A",
    )
    checker.equals(
        "전체 순위도 동일",
        [entry[0]["config"] for entry in rank_rows(reversed_rows)],
        ["A", "M", "Z"],
    )

    # ------------------------------------------------------------------
    checker.section("12. 비용 기준이 성능을 뒤집지는 않는다")

    rows = [
        row(
            "BIG_BUT_BETTER",
            val_top1_accuracy=0.320,
            total_parameters=50_000_000,
            mean_epoch_seconds=900.0,
            peak_gpu_memory_mb=20000.0,
            config_hash="zzz",
        ),
        row(
            "SMALL_BUT_WORSE",
            val_top1_accuracy=0.300,
            total_parameters=1_000_000,
            mean_epoch_seconds=10.0,
            peak_gpu_memory_mb=500.0,
            config_hash="aaa",
        ),
    ]
    winner, _ = pick_winner(rows)
    checker.equals("승자", winner["config"], "BIG_BUT_BETTER")
    checker.check(
        "Top-1 차이가 tolerance 밖이면 비용 기준은 쓰이지 않음",
        winner["config"] == "BIG_BUT_BETTER",
    )

    # ------------------------------------------------------------------
    checker.section("13. seed가 모두 성공한 설정만 비교")

    aggregate_rows = [
        {"config": "A", "successful_seed_count": 3},
        {"config": "B", "successful_seed_count": 2},
        {"config": "C", "successful_seed_count": 3},
        {"config": "D", "successful_seed_count": 0},
    ]

    complete, incomplete = filter_complete_seed_configs(aggregate_rows, 3)

    checker.equals(
        "모두 성공한 설정",
        [entry["config"] for entry in complete],
        ["A", "C"],
    )
    checker.equals(
        "제외된 설정",
        [entry["config"] for entry in incomplete],
        ["B", "D"],
    )

    complete, incomplete = filter_complete_seed_configs(aggregate_rows, 2)
    checker.equals(
        "기대 seed 수가 2면 B만 통과",
        [entry["config"] for entry in complete],
        ["B"],
    )

    # ------------------------------------------------------------------
    checker.section("14. score gap이 미세하게 달라도 비용 기준이 작동한다")

    # 실제 데이터에서 score gap이 소수점까지 같을 일은 거의 없다.
    # score gap이 선택 규칙에 남아 있었다면 여기서 BIG이 이겨버리고
    # total_parameters 이하는 영원히 호출되지 않는다.
    rows = [
        row(
            "BIG",
            val_score_gap=1.9000,
            total_parameters=20_000_000,
            config_hash="aaa",
        ),
        row(
            "SMALL",
            val_score_gap=1.8500,
            total_parameters=10_000_000,
            config_hash="zzz",
        ),
    ]
    winner, trace = pick_winner(rows)

    checker.equals("승자", winner["config"], "SMALL")
    checker.check(
        "score gap이 더 높은 BIG이 이기지 않음",
        winner["config"] == "SMALL",
        f"BIG gap=1.9000 > SMALL gap=1.8500",
    )
    checker.check(
        "판단 근거가 파라미터 수에서 끝남",
        "파라미터" in trace[-1],
        f"마지막 근거={trace[-1]!r}",
    )
    checker.check(
        "score gap은 판단 근거에 등장하지 않음",
        not any("Score Gap" in line for line in trace),
        f"trace={trace}",
    )

    # 차이가 더 작아도, 더 커도 결과는 같아야 한다
    for gap in (1.8999, 1.5000, 0.0001):
        rows = [
            row("BIG", val_score_gap=1.9, total_parameters=20_000_000),
            row("SMALL", val_score_gap=gap, total_parameters=10_000_000),
        ]
        winner, _ = pick_winner(rows)
        checker.equals(
            f"SMALL의 gap이 {gap}일 때도 승자",
            winner["config"],
            "SMALL",
        )

    # ------------------------------------------------------------------
    checker.section("15. score gap은 경고로만 쓰인다")

    # 상대 차이 10% 미만이면 경고하지 않는다
    rows = [
        row("BIG", val_score_gap=1.90, total_parameters=20_000_000),
        row("SMALL", val_score_gap=1.85, total_parameters=10_000_000),
    ]
    winner, _ = pick_winner(rows)
    warnings = check_stage_warnings(rows, winner)

    checker.check(
        "2.6% 차이는 경고하지 않음",
        not any("Score Gap" in line for line in warnings),
        f"warnings={warnings}",
    )

    # 상대 차이가 10%를 넘으면 경고한다
    rows = [
        row("BIG", val_score_gap=2.00, total_parameters=20_000_000),
        row("SMALL", val_score_gap=1.50, total_parameters=10_000_000),
    ]
    winner, _ = pick_winner(rows)
    warnings = check_stage_warnings(rows, winner)

    checker.equals("승자는 여전히 작은 모델", winner["config"], "SMALL")
    checker.check(
        "25% 차이는 경고함",
        any("Score Gap" in line for line in warnings),
        f"warnings={[w for w in warnings if 'Score Gap' in w]}",
    )

    return checker.finish()


if __name__ == "__main__":
    sys.exit(main())
