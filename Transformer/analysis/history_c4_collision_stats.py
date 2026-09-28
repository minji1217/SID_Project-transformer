"""history에서 c4를 빼면 item identity가 얼마나 무너지는지 센다.

학습하지 않는다. 모델을 만들지 않는다. parquet만 읽는다.

c4는 codebook이 아니라 (c1,c2,c3)가 겹치는 기사를 구분하는 일련번호다.
history에서 c4를 빼면 같은 삼중항을 가진 서로 다른 기사가 모델 입장에서
완전히 같은 토큰열이 된다. 이 통계는 그 정도를 센다.

세는 것

  1. 서로 다른 article_id인데 같은 c123을 갖는 history article 위치 수
  2. 한 history 안에 서로 다른 article_id가 같은 c123으로 같이 나오는
     impression 수  <- c4 제거로 실제 정보가 사라지는 경우
  3. 전체 history article 중 c123 collision group에 속하는 비율
  4. distinct c1234 수 vs distinct c123 수

    cd Transformer
    python -m analysis.history_c4_collision_stats \
        --out sweep_out/ebnerd_v2/priority2_direct/c4_collision_stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sweep.run_stage import write_csv


BASE_DIR = Path(__file__).resolve().parent.parent

HISTORY_COLUMNS = [
    "history_article_ids",
    "history_c1", "history_c2", "history_c3", "history_c4",
]

# Dataset이 history를 잘라 쓰는 길이. 모델이 실제로 보는 범위에 맞춘다.
DEFAULT_MAX_HISTORY = 50


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def resolve_path(path: str) -> Path:
    path_obj = Path(path).expanduser()
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else float("nan")


def analyse(path: Path, name: str, max_history: Optional[int]) -> Dict[str, Any]:
    section(f"{name}  —  {path.name}")

    df = pd.read_parquet(path, columns=HISTORY_COLUMNS)

    print(f"  impression {len(df):,}개")

    if max_history:
        print(f"  history는 뒤에서 {max_history}개만 봅니다 "
              f"(NewsSequenceDataset와 동일)")

    # 1) 전체 history 위치를 편다
    article_chunks: List[np.ndarray] = []
    triple_chunks: List[np.ndarray] = []
    quad_chunks: List[np.ndarray] = []
    impression_chunks: List[np.ndarray] = []

    lengths: List[int] = []

    for index, row in enumerate(df.itertuples(index=False)):
        ids = np.asarray(row.history_article_ids)
        c1 = np.asarray(row.history_c1)
        c2 = np.asarray(row.history_c2)
        c3 = np.asarray(row.history_c3)
        c4 = np.asarray(row.history_c4)

        if max_history:
            ids, c1, c2, c3, c4 = (
                a[-max_history:] for a in (ids, c1, c2, c3, c4)
            )

        length = len(c1)

        if length == 0:
            lengths.append(0)
            continue

        lengths.append(length)

        article_chunks.append(ids.astype(str))
        triple_chunks.append(np.char.add(np.char.add(
            np.char.add(c1.astype(str), "_"),
            np.char.add(c2.astype(str), "_")), c3.astype(str)))
        quad_chunks.append(np.char.add(np.char.add(
            triple_chunks[-1], "_"), c4.astype(str)))
        impression_chunks.append(np.full(length, index))

    articles = np.concatenate(article_chunks)
    triples = np.concatenate(triple_chunks)
    quads = np.concatenate(quad_chunks)
    impressions = np.concatenate(impression_chunks)

    total_positions = len(articles)

    print(f"  history article 위치 {total_positions:,}개 "
          f"(impression당 평균 {np.mean(lengths):.1f}개)")

    # 4) distinct 수
    distinct_quad = len(np.unique(quads))
    distinct_triple = len(np.unique(triples))
    distinct_article = len(np.unique(articles))

    section_4 = {
        "distinct_article_id": distinct_article,
        "distinct_c1234": distinct_quad,
        "distinct_c123": distinct_triple,
        "collapsed_identities": distinct_quad - distinct_triple,
        "collapse_ratio": pct(distinct_quad - distinct_triple, distinct_quad),
    }

    # 2) triple -> distinct article 수
    triple_to_articles: Dict[str, set] = defaultdict(set)

    for triple, article in zip(triples, articles):
        triple_to_articles[triple].add(article)

    colliding_triples = {
        triple for triple, group in triple_to_articles.items()
        if len(group) > 1
    }

    # 1) collision triple에 속한 위치 수
    is_colliding = np.fromiter(
        (triple in colliding_triples for triple in triples),
        dtype=bool, count=total_positions,
    )

    colliding_positions = int(is_colliding.sum())

    # 3) collision group에 속한 distinct article 수
    colliding_articles = set()
    for triple in colliding_triples:
        colliding_articles |= triple_to_articles[triple]

    # 2') 한 impression 안에서 실제로 부딪히는 경우
    within: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for impression, triple, article in zip(impressions, triples, articles):
        within[int(impression)][triple].add(article)

    impressions_with_clash = sum(
        1 for groups in within.values()
        if any(len(group) > 1 for group in groups.values())
    )

    clashing_pairs = sum(
        len(group) - 1
        for groups in within.values()
        for group in groups.values()
        if len(group) > 1
    )

    result = {
        "split": name,
        "path": str(path),
        "max_history": max_history,
        "impressions": int(len(df)),
        "history_positions": total_positions,
        "mean_history_length": float(np.mean(lengths)),

        "colliding_positions": colliding_positions,
        "colliding_positions_pct": pct(colliding_positions, total_positions),

        "impressions_with_within_history_clash": impressions_with_clash,
        "impressions_with_within_history_clash_pct":
            pct(impressions_with_clash, len(df)),
        "within_history_clashing_articles": clashing_pairs,

        "distinct_articles": distinct_article,
        "colliding_articles": len(colliding_articles),
        "colliding_articles_pct": pct(len(colliding_articles), distinct_article),

        "colliding_triples": len(colliding_triples),
        "distinct_triples": distinct_triple,
        **section_4,
    }

    # 출력
    print()
    print("  1) 같은 c123을 여러 article_id가 공유하는 history 위치")
    print(f"       {colliding_positions:,} / {total_positions:,}  "
          f"({result['colliding_positions_pct']:.2f}%)")

    print()
    print("  2) 한 history 안에서 서로 다른 article이 같은 c123으로 겹치는 impression")
    print(f"       {impressions_with_clash:,} / {len(df):,}  "
          f"({result['impressions_with_within_history_clash_pct']:.2f}%)")
    print(f"       겹쳐서 사라지는 article 수: {clashing_pairs:,}")
    print("       <- c4를 빼면 이 경우에만 실제로 정보가 없어집니다.")

    print()
    print("  3) collision group에 속한 article (전체 distinct 대비)")
    print(f"       {len(colliding_articles):,} / {distinct_article:,}  "
          f"({result['colliding_articles_pct']:.2f}%)")

    print()
    print("  4) c4 제거 전후 distinct 수")
    print(f"       distinct article_id : {distinct_article:>10,}")
    print(f"       distinct c1234      : {distinct_quad:>10,}")
    print(f"       distinct c123       : {distinct_triple:>10,}")
    print(f"       줄어든 identity     : {section_4['collapsed_identities']:>10,}  "
          f"({section_4['collapse_ratio']:.2f}%)")

    return result


def interpret(results: List[Dict[str, Any]]) -> None:
    section("해석")

    worst = max(
        results, key=lambda r: r["impressions_with_within_history_clash_pct"]
    )

    clash = worst["impressions_with_within_history_clash_pct"]

    print(f"  c4를 빼서 실제로 정보가 사라지는 impression 비율: "
          f"최대 {clash:.2f}% ({worst['split']})")
    print()

    if clash < 1.0:
        print("  => 1% 미만입니다. c4 제거로 잃는 정보가 사실상 없습니다.")
        print("     c123 실험이 좋아지면 그건 '정보를 잃었는데도 좋아진 것'이")
        print("     아니라 'c4가 애초에 방해였다'는 뜻으로 읽어야 합니다.")
    elif clash < 5.0:
        print("  => 몇 % 수준입니다. 잃는 정보가 있지만 작습니다.")
        print("     c123이 이 폭보다 크게 좋아지면 c4가 방해였다고 볼 수 있습니다.")
    else:
        print("  => 무시할 수 없는 비율입니다. c123이 나빠지면 그게 c4 제거의")
        print("     손해인지 다른 이유인지 구분해서 봐야 합니다.")

    print()
    print("  이 통계는 데이터만 본 것입니다. 학습 구조를 바꾸지 않았습니다.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="history에서 c4를 뺄 때의 identity collapse를 센다."
    )
    parser.add_argument(
        "--train-path",
        default="datasets/ebnerd/train_sequences_1pos4neg.parquet",
    )
    parser.add_argument(
        "--validation-path",
        default="datasets/ebnerd/validation_sequences_1pos4neg_half.parquet",
    )
    parser.add_argument(
        "--out",
        default="sweep_out/ebnerd_v2/priority2_direct/c4_collision_stats",
    )
    parser.add_argument("--max-history", type=int, default=DEFAULT_MAX_HISTORY)

    args = parser.parse_args()

    out_dir = resolve_path(args.out)

    section("history c4 collision 통계")
    print("  학습하지 않습니다. 모델을 만들지 않습니다. parquet만 읽습니다.")

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"\n출력 폴더가 비어 있지 않습니다. 덮어쓰지 않습니다: {out_dir}")
        return 1

    results: List[Dict[str, Any]] = []

    for label, raw in (
        ("train", args.train_path),
        ("validation", args.validation_path),
    ):
        path = resolve_path(raw)

        if not path.exists():
            print(f"\n{label} parquet이 없습니다: {path}")
            return 1

        results.append(analyse(path, label, args.max_history))

    interpret(results)

    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "c4_collision_stats.csv", results)

    (out_dir / "c4_collision_stats.json").write_text(
        json.dumps({
            "experiment": "history_c4_collision_stats",
            "max_history": args.max_history,
            "results": results,
            "note": (
                "c4는 codebook이 아니라 (c1,c2,c3) 충돌 구분용 일련번호다. "
                "학습 구조를 바꾸지 않고 데이터만 센 통계다."
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    section("저장")
    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {out_dir / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
