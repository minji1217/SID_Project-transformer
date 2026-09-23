"""테스트 스크립트가 함께 쓰는 아주 작은 검사 도구."""

from __future__ import annotations

from typing import Any, List


class Checker:
    def __init__(self, title: str) -> None:
        self.title = title
        self.failures: List[str] = []
        print("=" * 70)
        print(title)
        print("=" * 70)

    def section(self, name: str) -> None:
        print()
        print(name)

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  [OK]   {name}" + (f" — {detail}" if detail else ""))
        else:
            self.failures.append(f"{name}: {detail}")
            print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

    def equals(self, name: str, actual: Any, expected: Any) -> None:
        self.check(
            name,
            actual == expected,
            f"actual={actual!r} expected={expected!r}",
        )

    def finish(self) -> int:
        print()
        print("=" * 70)

        if self.failures:
            print(f"FAILED — {len(self.failures)}개")
            for failure in self.failures:
                print(f"  - {failure}")
            return 1

        print("PASSED")
        return 0
