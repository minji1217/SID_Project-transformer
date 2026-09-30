"""P2 direct score와 V1 weighted generation score가 서로 다른 정보를 갖는가.

재학습하지 않는다. 추론하지 않는다. 이미 저장된 두 score를 읽어
합쳐 보기만 한다.

    D = P2 direct bilinear score       u^T W v
    G = V1 weighted generation score   L1 + 0.5*L2 + 0.1*L3

impression 안의 후보 5개 각각에 대해 z-normalize한 뒤

    H(lambda) = lambda * zD + (1 - lambda) * zG

lambda를 0에서 1까지 훑는다.

    lambda = 0  ->  G만
    lambda = 1  ->  D만

z-normalize는 impression 안에서의 affine 변환이므로 순위를 바꾸지
않는다. 따라서 lambda=0은 기존 G 결과를, lambda=1은 기존 D 결과를
그대로 재현해야 한다. 두 끝점이 곧 재현 검증이다.

이 sweep은 Validation에서 고른다. G 자체도 Validation에서 42개 weight
조합을 훑어 얻은 값이다. 따라서 결과는 최종 성능 추정치가 아니라
"direct score가 보완적인 정보를 갖는가"를 보는 진단이다.

    cd Transformer
    python -m analysis.run_score_fusion_diagnostic \
        --direct sweep_out/ebnerd_v2/priority2_direct/regularization_sweep/do0.00_wd0.0000/seed42/validation_predictions.parquet \
        --priority0-dir sweep_out/ebnerd_v2/priority0_shuffled/seed42 \
        --out sweep_out/ebnerd_v2/priority2_direct/score_fusion
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluate.metrics import evaluate_ranking
from sweep.run_stage import write_csv


BASE_DIR = Path(__file__).resolve().parent.parent

NUM_CANDIDATES = 5
GROUP_COLUMN = "sample_index"

# G = L1 + 0.5*L2 + 0.1*L3
G_WEIGHTS = (1.0, 0.5, 0.1)

LAMBDAS = [round(0.1 * i, 1) for i in range(11)]

EPS = 1e-8

# 두 끝점이 재현해야 하는 값. GPU 비결정성과 무관하게 같은 파일을
# 읽어 계산하므로 아주 작은 오차만 허용한다.
REPRODUCE_TOLERANCE = 1e-6

RANDOM_TOP1 = 0.20
V1_EQUAL_TOP1 = 0.27059
G_REFERENCE_TOP1 = 0.29312
D_REFERENCE_TOP1 = 0.26689

CAVEAT_EN = (
    "This hybrid sweep is Validation-tuned and is used only to diagnose\n"
    "whether the direct score contains complementary ranking information.\n"
    "It is not a final unbiased performance estimate."
)


def spearman(a: pd.Series, b: pd.Series) -> float:
    """rank를 매긴 뒤 Pearson을 낸다.

    pandas의 method="spearman"은 scipy를 부르는데 이 환경에는 없다.
    동점은 평균 순위로 처리한다 (pandas rank 기본값). 정의상 같다.
    """
    return float(a.rank().corr(b.rank(), method="pearson"))


def section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def resolve_path(path: str) -> Path:
    path_obj = Path(path).expanduser()
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


# ---------------------------------------------------------------- 정렬 검증


def align(
    direct: pd.DataFrame, generation: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """두 score가 같은 후보를 가리키는지 확인한 뒤 합친다."""
    section("정렬 검증")

    for name, df in (("direct", direct), ("generation", generation)):
        counts = df.groupby(GROUP_COLUMN).size()

        if not (counts == NUM_CANDIDATES).all():
            raise ValueError(
                f"{name}에 후보가 {NUM_CANDIDATES}개가 아닌 impression이 있습니다."
            )

    if len(direct) != len(generation):
        raise ValueError(
            f"row 수가 다릅니다: direct {len(direct):,} vs "
            f"generation {len(generation):,}"
        )

    direct = direct.sort_values(
        [GROUP_COLUMN, "candidate_index"], kind="stable"
    ).reset_index(drop=True)
    generation = generation.sort_values(
        [GROUP_COLUMN, "candidate_index"], kind="stable"
    ).reset_index(drop=True)

    print(f"  row {len(direct):,}개, impression "
          f"{len(direct) // NUM_CANDIDATES:,}개")
    print()

    checks: List[Dict[str, Any]] = []
    all_ok = True

    def check(label: str, a: pd.Series, b: pd.Series) -> None:
        nonlocal all_ok

        left = a.astype(str).to_numpy()
        right = b.astype(str).to_numpy()
        mismatched = int((left != right).sum())
        ok = mismatched == 0

        all_ok = all_ok and ok
        checks.append({
            "field": label, "mismatched": mismatched, "passed": ok,
        })

        print(f"  {'OK  ' if ok else 'FAIL'}  {label:<28} "
              f"불일치 {mismatched:,} / {len(left):,}")

    check("impression_id", direct["impression_id"], generation["impression_id"])
    check("candidate position", direct["candidate_index"],
          generation["candidate_index"])
    check("label", direct["label"].astype(int),
          generation["label"].astype(int))

    for level in ("c1", "c2", "c3"):
        if level in direct.columns and level in generation.columns:
            check(f"candidate {level}", direct[level], generation[level])

    # 기사 ID는 양쪽에 다 있을 때만 대조할 수 있다.
    article_column = "candidate_article_id"
    article_status = "없음"

    if article_column in direct.columns and article_column in generation.columns:
        check("candidate_article_id", direct[article_column],
              generation[article_column])
        article_status = "양쪽 대조"
    elif article_column in generation.columns:
        article_status = "generation 쪽에만 있음 (대조 불가)"
        print(f"  ----  candidate_article_id           "
              f"direct 쪽에 없어 대조하지 못했습니다")
    elif article_column in direct.columns:
        article_status = "direct 쪽에만 있음 (대조 불가)"
        print(f"  ----  candidate_article_id           "
              f"generation 쪽에 없어 대조하지 못했습니다")
    else:
        print(f"  ----  candidate_article_id           "
              f"양쪽 모두 없어 대조하지 못했습니다")

    print()

    if article_status != "양쪽 대조":
        print("  기사 ID를 양쪽에서 대조하지 못했습니다. 대신 impression,")
        print("  후보 위치, label, (c1,c2,c3)로 확인했습니다. 진단에서")
        print("  impression 안의 후보 5개는 c1c2c3가 항상 서로 달랐으므로")
        print("  (impression, c1, c2, c3)가 후보를 유일하게 가리킵니다.")
        print()

    if not all_ok:
        raise ValueError(
            "두 score의 후보 정렬이 어긋납니다. 위 불일치를 먼저 확인하세요."
        )

    print("  => 같은 후보를 가리킵니다.")

    merged = pd.DataFrame({
        "impression_id": direct["impression_id"],
        GROUP_COLUMN: direct[GROUP_COLUMN],
        "candidate_index": direct["candidate_index"],
        "label": direct["label"].astype(int),
        "c1": direct["c1"], "c2": direct["c2"], "c3": direct["c3"],
        "D": direct["direct_score"].astype(np.float64),
        "L1": generation["L1"].astype(np.float64),
        "L2": generation["L2"].astype(np.float64),
        "L3": generation["L3"].astype(np.float64),
    })

    if article_column in generation.columns:
        merged[article_column] = generation[article_column]
    elif article_column in direct.columns:
        merged[article_column] = direct[article_column]

    merged["G"] = (
        G_WEIGHTS[0] * merged["L1"]
        + G_WEIGHTS[1] * merged["L2"]
        + G_WEIGHTS[2] * merged["L3"]
    )

    return merged, {
        "checks": checks,
        "all_passed": all_ok,
        "article_id_status": article_status,
    }


# ---------------------------------------------------------------- z-normalize


def z_normalize(
    values: np.ndarray, num_impressions: int
) -> Tuple[np.ndarray, int]:
    """impression 안의 후보 5개끼리 z-normalize한다.

    std가 0에 가까우면 후보 5개 점수가 사실상 같다는 뜻이다. eps를
    더해 0으로 나누는 것을 막고, 그런 impression 수를 따로 센다.
    """
    grid = values.reshape(num_impressions, NUM_CANDIDATES)

    mean = grid.mean(axis=1, keepdims=True)
    std = grid.std(axis=1, ddof=0, keepdims=True)

    degenerate = int((std.reshape(-1) <= EPS).sum())

    return ((grid - mean) / (std + EPS)).reshape(-1), degenerate


# ---------------------------------------------------------------- 평가


def evaluate_scores(
    frame: pd.DataFrame, scores: np.ndarray
) -> Dict[str, float]:
    work = frame[[GROUP_COLUMN, "label"]].copy()
    work["candidate_score"] = scores
    metrics = evaluate_ranking(df=work, group_column=GROUP_COLUMN)

    return {
        "top1_accuracy": metrics["top1_accuracy"],
        "mrr": metrics["mrr"],
        "ndcg5": metrics["ndcg@5"],
        "auc": metrics["auc"],
        "num_impressions": metrics["num_impressions"],
    }


def correct_mask(
    scores: np.ndarray, labels: np.ndarray, num_impressions: int
) -> np.ndarray:
    """impression별로 1등이 정답인지."""
    grid = scores.reshape(num_impressions, NUM_CANDIDATES)
    label_grid = labels.reshape(num_impressions, NUM_CANDIDATES)

    return grid.argmax(axis=1) == label_grid.argmax(axis=1)


# ---------------------------------------------------------------- 보고


def print_lambda_table(rows: List[Dict[str, Any]]) -> None:
    section("lambda sweep")

    header = (
        f"  {'lambda':>7} {'Top-1':>10} {'MRR':>9} {'nDCG@5':>9} "
        f"{'AUC':>9}   설명"
    )
    print(header)
    print("  " + "-" * (len(header) + 6))

    best = max(rows, key=lambda r: r["top1_accuracy"])

    for row in rows:
        note = ""

        if row["lambda"] == 0.0:
            note = "G만 (generation)"
        elif row["lambda"] == 1.0:
            note = "D만 (direct)"

        if row is best:
            note = (note + "  <- 최고").strip()

        print(
            f"  {row['lambda']:>7.1f} {row['top1_accuracy'] * 100:>9.3f}% "
            f"{row['mrr']:>9.4f} {row['ndcg5']:>9.4f} {row['auc']:>9.4f}   "
            f"{note}"
        )


def print_agreement(agreement: Dict[str, Any], total: int) -> None:
    section("G와 D가 서로 다른 것을 맞히는가")

    print(f"  {'':<26} {'impression':>12} {'비율':>9}")
    print("  " + "-" * 50)

    for label, key in (
        ("G 정답 / D 정답", "both_correct"),
        ("G 정답 / D 오답", "g_only"),
        ("G 오답 / D 정답", "d_only"),
        ("G 오답 / D 오답", "both_wrong"),
    ):
        count = agreement[key]
        print(f"  {label:<26} {count:>12,} {100 * count / total:>8.2f}%")

    print()
    print(f"  G 단독 Top-1 : {agreement['g_top1'] * 100:.3f}%")
    print(f"  D 단독 Top-1 : {agreement['d_top1'] * 100:.3f}%")
    print(f"  둘 중 하나라도 맞힌 비율 (상한) : "
          f"{agreement['union'] * 100:.3f}%")
    print(f"  둘 다 맞힌 비율 (교집합)        : "
          f"{agreement['both_correct'] / total * 100:.3f}%")

    print()

    d_only = agreement["d_only"] / total

    print(f"  D만 맞힌 impression : {agreement['d_only']:,} "
          f"({d_only * 100:.2f}%)")

    if d_only < 0.02:
        print("  => 2% 미만입니다. D가 G와 거의 같은 것을 맞히고 있습니다.")
        print("     direct score는 generation score의 약한 복제에 가깝습니다.")
    elif d_only < 0.05:
        print("  => 적지만 없지는 않습니다. hybrid가 얼마나 살리는지 보세요.")
    else:
        print("  => 무시할 수 없는 양입니다. D가 G가 놓치는 것을 잡고 있습니다.")
        print("     Train에서 fusion weight를 학습할 값이 있습니다.")

    print()
    print(f"  상한 {agreement['union'] * 100:.3f}%는 두 score를 완벽하게")
    print(f"  골라 썼을 때의 Top-1입니다. hybrid 최고가 여기서 얼마나")
    print(f"  떨어져 있는지가 fusion의 여지를 보여줍니다.")


def print_correlation(correlation: Dict[str, float]) -> None:
    section("D와 G의 상관")

    print(f"  {'':<34} {'Pearson':>10} {'Spearman':>10}")
    print("  " + "-" * 56)
    print(f"  {'후보 전체 (raw score)':<34} "
          f"{correlation['pearson_raw']:>10.4f} "
          f"{correlation['spearman_raw']:>10.4f}")
    print(f"  {'후보 전체 (impression 내 z)':<34} "
          f"{correlation['pearson_z']:>10.4f} "
          f"{correlation['spearman_z']:>10.4f}")
    print(f"  {'impression별 평균':<34} "
          f"{correlation['pearson_per_impression']:>10.4f} "
          f"{correlation['spearman_per_impression']:>10.4f}")

    print()
    print("  impression 안에서 재는 값이 ranking에 직접 관련됩니다.")
    print("  raw score는 impression마다 scale이 달라 상관이 부풀 수 있습니다.")

    value = correlation["spearman_per_impression"]

    print()

    if value > 0.8:
        print(f"  impression별 Spearman {value:.4f} — 두 score가 거의 같은")
        print("  순서를 매깁니다. 합쳐도 얻을 것이 적습니다.")
    elif value > 0.5:
        print(f"  impression별 Spearman {value:.4f} — 상당히 겹치지만")
        print("  다른 부분도 있습니다.")
    else:
        print(f"  impression별 Spearman {value:.4f} — 두 score가 꽤 다른")
        print("  순서를 매깁니다. 합칠 여지가 있습니다.")


def print_comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("비교")

    best = max(rows, key=lambda r: r["top1_accuracy"])

    table = [
        ("Random", RANDOM_TOP1, ""),
        ("V1 equal  L1+L2+L3", V1_EQUAL_TOP1, "생성확률 동일 가중"),
        ("P2 Direct", D_REFERENCE_TOP1, "u^T W v"),
        ("V1 weighted G  L1+0.5L2+0.1L3", G_REFERENCE_TOP1,
         "Validation에서 탐색한 weight"),
        (f"Hybrid best  lambda={best['lambda']:g}",
         best["top1_accuracy"], "이번 진단"),
    ]

    for name, value, note in table:
        print(f"  {name:<36} {value * 100:>8.3f}%   {note}")

    g_row = next(r for r in rows if r["lambda"] == 0.0)
    d_row = next(r for r in rows if r["lambda"] == 1.0)

    print()
    print(f"  best lambda : {best['lambda']:g}")
    print(f"  G 대비      : "
          f"{(best['top1_accuracy'] - g_row['top1_accuracy']) * 100:+.3f}%p")
    print(f"  D 대비      : "
          f"{(best['top1_accuracy'] - d_row['top1_accuracy']) * 100:+.3f}%p")

    print()
    print("  " + CAVEAT_EN.replace("\n", "\n  "))

    return best


def write_readme(
    path: Path, rows: List[Dict[str, Any]], agreement: Dict[str, Any],
    correlation: Dict[str, float], alignment: Dict[str, Any],
    reproduction: Dict[str, Any], best: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    lines: List[str] = []
    add = lines.append

    add("# score fusion diagnostic")
    add("")
    add("재학습하지 않았다. 추론하지 않았다. 이미 저장된 두 score를 읽어")
    add("합쳐 보기만 했다.")
    add("")
    add("    D = P2 direct bilinear score       u^T W v")
    add(f"    G = V1 weighted generation score   "
        f"L1 + {G_WEIGHTS[1]:g}*L2 + {G_WEIGHTS[2]:g}*L3")
    add("")
    add("impression 안의 후보 5개끼리 z-normalize한 뒤")
    add("")
    add("    H(lambda) = lambda * zD + (1 - lambda) * zG")
    add("")
    add("## 주의")
    add("")
    add("```")
    add(CAVEAT_EN)
    add("```")
    add("")
    add(f"`G = L1 + {G_WEIGHTS[1]:g}*L2 + {G_WEIGHTS[2]:g}*L3` 자체가 이미")
    add("Validation에서 42개 weight 조합을 탐색해 얻은 값이다. 이번 lambda")
    add("sweep도 Validation에서 하므로, hybrid 최고 성능은 최종 성능")
    add("추정치가 아니라 development diagnostic이다.")
    add("")

    add("## 실행 정보")
    add("")
    add(f"- direct score  : `{metadata['direct_path']}`")
    add(f"- generation    : `{metadata['generation_path']}`")
    add(f"- impression    : {metadata['num_impressions']:,}")
    add(f"- lambda        : {', '.join(f'{v:g}' for v in metadata['lambdas'])}")
    add(f"- git commit    : {metadata.get('git_commit')}")
    add("")

    add("## 정렬 검증")
    add("")
    add("| 항목 | 불일치 | 판정 |")
    add("|---|---|---|")

    for check in alignment["checks"]:
        add(f"| {check['field']} | {check['mismatched']:,} | "
            f"{'OK' if check['passed'] else 'FAIL'} |")

    add("")
    add(f"candidate_article_id: {alignment['article_id_status']}")
    add("")

    if alignment["article_id_status"] != "양쪽 대조":
        add("기사 ID를 양쪽에서 대조하지는 못했다. 대신 impression, 후보 위치,")
        add("label, `(c1,c2,c3)`로 확인했다. 진단에서 impression 안의 후보")
        add("5개는 `c1c2c3`가 항상 서로 달랐으므로 `(impression, c1, c2, c3)`가")
        add("후보를 유일하게 가리킨다.")
        add("")

    add("## 끝점 재현")
    add("")
    add("z-normalize는 impression 안의 affine 변환이라 순위를 바꾸지 않는다.")
    add("따라서 두 끝점은 기존 결과를 그대로 재현해야 한다.")
    add("")
    add("| 끝점 | 기준값 | 이번 값 | 차이 | 판정 |")
    add("|---|---|---|---|---|")

    for item in reproduction["checks"]:
        add(f"| {item['name']} | {item['reference'] * 100:.3f}% | "
            f"{item['actual'] * 100:.3f}% | "
            f"{item['deviation'] * 100:.4f}%p | "
            f"{'OK' if item['passed'] else '허용치 밖'} |")

    add("")

    add("## lambda sweep")
    add("")
    add("| lambda | Top-1 | MRR | nDCG@5 | AUC | |")
    add("|---|---|---|---|---|---|")

    for row in rows:
        note = ""

        if row["lambda"] == 0.0:
            note = "G만"
        elif row["lambda"] == 1.0:
            note = "D만"

        if row["lambda"] == best["lambda"]:
            note = (note + " **최고**").strip()

        add(
            f"| {row['lambda']:g} | {row['top1_accuracy'] * 100:.3f}% | "
            f"{row['mrr']:.4f} | {row['ndcg5']:.4f} | {row['auc']:.4f} | "
            f"{note} |"
        )

    add("")

    add("## G와 D의 정답 분포")
    add("")
    add("| 경우 | impression | 비율 |")
    add("|---|---|---|")

    total = agreement["total"]

    for label, key in (
        ("G 정답 / D 정답", "both_correct"),
        ("G 정답 / D 오답", "g_only"),
        ("**G 오답 / D 정답**", "d_only"),
        ("G 오답 / D 오답", "both_wrong"),
    ):
        add(f"| {label} | {agreement[key]:,} | "
            f"{100 * agreement[key] / total:.2f}% |")

    add("")
    add(f"- G 단독 Top-1 : {agreement['g_top1'] * 100:.3f}%")
    add(f"- D 단독 Top-1 : {agreement['d_top1'] * 100:.3f}%")
    add(f"- 둘 중 하나라도 맞힌 비율 (상한) : "
        f"{agreement['union'] * 100:.3f}%")
    add("")
    add("상한은 두 score를 완벽하게 골라 썼을 때의 Top-1이다. hybrid")
    add("최고가 여기서 얼마나 떨어져 있는지가 fusion의 여지를 보여준다.")
    add("")

    add("## D와 G의 상관")
    add("")
    add("| 기준 | Pearson | Spearman |")
    add("|---|---|---|")
    add(f"| 후보 전체 (raw score) | {correlation['pearson_raw']:.4f} | "
        f"{correlation['spearman_raw']:.4f} |")
    add(f"| 후보 전체 (impression 내 z) | {correlation['pearson_z']:.4f} | "
        f"{correlation['spearman_z']:.4f} |")
    add(f"| impression별 평균 | "
        f"{correlation['pearson_per_impression']:.4f} | "
        f"{correlation['spearman_per_impression']:.4f} |")
    add("")
    add("impression 안에서 재는 값이 ranking에 직접 관련된다. raw score는")
    add("impression마다 scale이 달라 상관이 부풀 수 있다.")
    add("")

    add("## 비교")
    add("")
    add("| 구분 | Top-1 | 비고 |")
    add("|---|---|---|")
    add(f"| Random | {RANDOM_TOP1 * 100:.2f}% | |")
    add(f"| P2 Direct | {D_REFERENCE_TOP1 * 100:.3f}% | `u^T W v` |")
    add(f"| V1 equal L1+L2+L3 | {V1_EQUAL_TOP1 * 100:.3f}% | 생성확률 동일 가중 |")
    add(f"| V1 weighted G | {G_REFERENCE_TOP1 * 100:.3f}% | "
        f"Validation에서 탐색한 weight |")
    add(f"| **Hybrid best** (lambda={best['lambda']:g}) | "
        f"{best['top1_accuracy'] * 100:.3f}% | 이번 진단 |")
    add("")

    g_row = next(r for r in rows if r["lambda"] == 0.0)
    d_row = next(r for r in rows if r["lambda"] == 1.0)

    add(f"- best lambda : {best['lambda']:g}")
    add(f"- G 대비 : "
        f"{(best['top1_accuracy'] - g_row['top1_accuracy']) * 100:+.3f}%p")
    add(f"- D 대비 : "
        f"{(best['top1_accuracy'] - d_row['top1_accuracy']) * 100:+.3f}%p")
    add("")

    add("## 하지 않은 것")
    add("")
    add("- P2 재학습, 추론")
    add("- dropout / weight decay 재탐색, LR 변경")
    add("- hidden weight, beta4 변경")
    add("- normalization layer 추가, MLP, candidate c4")
    add("- level-wise direct scorer (D1/D2/D3)")
    add("- Test 평가")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P2 direct score와 V1 weighted generation score의 보완성 진단"
    )
    parser.add_argument(
        "--direct", required=True,
        help="P2 validation_predictions.parquet",
    )
    parser.add_argument(
        "--priority0-dir",
        default="sweep_out/ebnerd_v2/priority0_shuffled/seed42",
    )
    parser.add_argument(
        "--generation", default=None,
        help="생략하면 priority0-dir의 candidate_scores.parquet",
    )
    parser.add_argument(
        "--out",
        default="sweep_out/ebnerd_v2/priority2_direct/score_fusion",
    )

    args = parser.parse_args()

    started_at = time.time()

    direct_path = resolve_path(args.direct)
    out_dir = resolve_path(args.out)

    generation_path = (
        resolve_path(args.generation) if args.generation
        else resolve_path(args.priority0_dir) / "candidate_scores.parquet"
    )

    section("score fusion diagnostic")

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"출력 폴더가 비어 있지 않습니다. 덮어쓰지 않습니다: {out_dir}")
        return 1

    for path, label in (
        (direct_path, "direct score parquet"),
        (generation_path, "generation score parquet"),
    ):
        if not path.exists():
            print(f"{label}가 없습니다: {path}")
            return 1

    print(f"  direct     : {direct_path}")
    print(f"  generation : {generation_path}")
    print(f"  출력       : {out_dir}")
    print()
    print("  재학습하지 않습니다. 추론하지 않습니다. Test를 쓰지 않습니다.")

    direct = pd.read_parquet(direct_path)
    generation = pd.read_parquet(generation_path)

    for column, frame, name in (
        ("direct_score", direct, "direct"),
        ("L1", generation, "generation"),
    ):
        if column not in frame.columns:
            print(f"{name} parquet에 {column}이 없습니다.")
            return 1

    merged, alignment = align(direct, generation)

    num_impressions = len(merged) // NUM_CANDIDATES

    # ---- z-normalize
    section("z-normalize")

    d_values = merged["D"].to_numpy()
    g_values = merged["G"].to_numpy()

    zd, degenerate_d = z_normalize(d_values, num_impressions)
    zg, degenerate_g = z_normalize(g_values, num_impressions)

    merged["zD"] = zd
    merged["zG"] = zg

    print(f"  impression 안의 후보 5개끼리 정규화했습니다. eps = {EPS:g}")
    print()
    print(f"  {'':<22} {'평균':>10} {'표준편차':>10} {'최소':>10} {'최대':>10}")
    print("  " + "-" * 66)

    for label, values in (
        ("D (raw)", d_values), ("G (raw)", g_values),
        ("zD", zd), ("zG", zg),
    ):
        print(f"  {label:<22} {values.mean():>10.4f} {values.std():>10.4f} "
              f"{values.min():>10.4f} {values.max():>10.4f}")

    print()
    print(f"  후보 5개 점수가 사실상 같은 impression")
    print(f"    D : {degenerate_d:,} / {num_impressions:,}")
    print(f"    G : {degenerate_g:,} / {num_impressions:,}")

    if degenerate_d or degenerate_g:
        print("    이런 impression은 z가 전부 0에 가까워 무작위와 같습니다.")

    # ---- lambda sweep
    section(f"{len(LAMBDAS)}개 lambda 평가")

    ordered = merged[[GROUP_COLUMN, "label"]]
    labels = merged["label"].to_numpy()

    rows: List[Dict[str, Any]] = []

    for value in LAMBDAS:
        scores = value * zd + (1.0 - value) * zg
        metrics = evaluate_scores(ordered, scores)

        rows.append({"lambda": value, **metrics})

        print(f"  lambda {value:>4.1f}  Top-1 {metrics['top1_accuracy'] * 100:6.3f}%  "
              f"MRR {metrics['mrr']:.4f}  AUC {metrics['auc']:.4f}")

    # ---- 끝점 재현
    section("끝점 재현")

    g_row = next(r for r in rows if r["lambda"] == 0.0)
    d_row = next(r for r in rows if r["lambda"] == 1.0)

    raw_g = evaluate_scores(ordered, g_values)
    raw_d = evaluate_scores(ordered, d_values)

    repro_checks = []

    for name, zrow, raw, reference in (
        ("lambda=0 (G)", g_row, raw_g, G_REFERENCE_TOP1),
        ("lambda=1 (D)", d_row, raw_d, D_REFERENCE_TOP1),
    ):
        # z 정규화가 순위를 바꾸지 않았는지 먼저 본다.
        internal = abs(zrow["top1_accuracy"] - raw["top1_accuracy"])
        external = abs(zrow["top1_accuracy"] - reference)

        repro_checks.append({
            "name": name,
            "reference": reference,
            "actual": zrow["top1_accuracy"],
            "raw": raw["top1_accuracy"],
            "deviation": external,
            "z_vs_raw": internal,
            "passed": external <= 5e-4,
        })

        print(f"  {name}")
        print(f"    z 적용 후      {zrow['top1_accuracy'] * 100:.4f}%")
        print(f"    raw score      {raw['top1_accuracy'] * 100:.4f}%")
        print(f"    z vs raw 차이  {internal * 100:.6f}%p  "
              f"{'(순위 보존)' if internal <= REPRODUCE_TOLERANCE else '(순위가 바뀜)'}")
        print(f"    기존 기준값    {reference * 100:.4f}%")
        print(f"    기준값 차이    {external * 100:.4f}%p")
        print()

    reproduction = {"checks": repro_checks}

    # ---- 정답 분포
    d_correct = correct_mask(d_values, labels, num_impressions)
    g_correct = correct_mask(g_values, labels, num_impressions)

    agreement = {
        "total": num_impressions,
        "both_correct": int((g_correct & d_correct).sum()),
        "g_only": int((g_correct & ~d_correct).sum()),
        "d_only": int((~g_correct & d_correct).sum()),
        "both_wrong": int((~g_correct & ~d_correct).sum()),
        "g_top1": float(g_correct.mean()),
        "d_top1": float(d_correct.mean()),
        "union": float((g_correct | d_correct).mean()),
    }

    print_agreement(agreement, num_impressions)

    # ---- 상관
    d_grid = d_values.reshape(num_impressions, NUM_CANDIDATES)
    g_grid = g_values.reshape(num_impressions, NUM_CANDIDATES)

    def per_impression(method: str) -> float:
        values = []

        for i in range(num_impressions):
            a = pd.Series(d_grid[i])
            b = pd.Series(g_grid[i])

            if a.std(ddof=0) <= EPS or b.std(ddof=0) <= EPS:
                continue

            values.append(
                spearman(a, b) if method == "spearman"
                else a.corr(b, method="pearson")
            )

        return float(np.nanmean(values)) if values else float("nan")

    d_series = pd.Series(d_values)
    g_series = pd.Series(g_values)
    zd_series = pd.Series(zd)
    zg_series = pd.Series(zg)

    correlation = {
        "pearson_raw": float(d_series.corr(g_series, method="pearson")),
        "spearman_raw": spearman(d_series, g_series),
        "pearson_z": float(zd_series.corr(zg_series, method="pearson")),
        "spearman_z": spearman(zd_series, zg_series),
        "pearson_per_impression": per_impression("pearson"),
        "spearman_per_impression": per_impression("spearman"),
    }

    print_correlation(correlation)

    print_lambda_table(rows)
    best = print_comparison(rows)

    # ---- 저장
    section("저장")

    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "lambda_sweep.csv", rows)

    keep = [
        "impression_id", GROUP_COLUMN, "candidate_index", "label",
        "c1", "c2", "c3", "D", "G", "zD", "zG", "L1", "L2", "L3",
    ]

    if "candidate_article_id" in merged.columns:
        keep.insert(3, "candidate_article_id")

    merged[keep].to_parquet(
        out_dir / "fused_scores.parquet", index=False
    )

    metadata = {
        "experiment": "score_fusion_diagnostic",
        "direct_path": str(direct_path),
        "generation_path": str(generation_path),
        "g_weights": list(G_WEIGHTS),
        "g_formula": (
            f"L1 + {G_WEIGHTS[1]:g}*L2 + {G_WEIGHTS[2]:g}*L3"
        ),
        "hybrid_formula": "lambda * zD + (1 - lambda) * zG",
        "normalization": "per-impression z over 5 candidates",
        "eps": EPS,
        "lambdas": LAMBDAS,
        "num_impressions": num_impressions,
        "num_rows": int(len(merged)),
        "degenerate_impressions": {"D": degenerate_d, "G": degenerate_g},
        "alignment": alignment,
        "reproduction": reproduction,
        "lambda_results": rows,
        "agreement": agreement,
        "correlation": correlation,
        "best": best,
        "caveat": CAVEAT_EN,
        "did_not_do": [
            "P2 재학습, 추론",
            "dropout / weight decay 재탐색, LR 변경",
            "hidden weight, beta4 변경",
            "normalization layer, MLP, candidate c4",
            "level-wise direct scorer (D1/D2/D3)",
            "Test 평가",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 1),
        "git_commit": git_commit_hash(),
        "python": sys.version.split()[0],
    }

    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    write_readme(
        out_dir / "README.md", rows, agreement, correlation,
        alignment, reproduction, best, metadata,
    )

    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {out_dir / name}")

    print()
    print(f"  총 소요 {time.time() - started_at:.0f}초")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
