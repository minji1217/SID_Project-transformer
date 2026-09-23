"""
지표 우선순위에 따른 선택 규칙을 검사한다.

    cd Transformer
    python -m sweep.test_selection
"""

from __future__ import annotations

import sys

from sweep.selection import (
    NotEnoughSuccessfulRuns,
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

    return checker.finish()


if __name__ == "__main__":
    sys.exit(main())
