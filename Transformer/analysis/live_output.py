"""파이프로 묶여도 출력이 한 줄씩 바로 보이게 한다.

파이썬은 stdout이 터미널이면 줄 단위로, 파이프면 블록(보통 4~8KB)
단위로 버퍼링한다. 그래서 `python -m ... | tee log`로 돌리면 스크립트가
멀쩡히 도는 중에도 화면에 아무것도 안 나와서, 멈춘 것처럼 보인다.

`python -u`를 붙이면 되지만 매번 기억해야 하므로, 스크립트가 알아서
줄 단위 버퍼링으로 바꾸게 한다. 계산에는 영향이 없다.
"""

from __future__ import annotations

import sys


def enable_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if reconfigure is None:
            continue

        try:
            reconfigure(line_buffering=True)
        except (ValueError, OSError):
            # 이미 닫혔거나 다시 설정할 수 없는 stream이면 그냥 둔다.
            pass
