"""
실패한 run의 원인 분류.

OOM은 성능이 나쁜 설정이 아니라 현재 장비에서 실행할 수 없는 조합이다.
데이터 오류는 설정과 무관하게 모든 run이 실패한다.
둘을 구분해야 무엇을 고쳐야 하는지 알 수 있다.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


ERROR_CUDA_OOM = "CUDA_OOM"
ERROR_DATA = "DATA_ERROR"
ERROR_RUNTIME = "RUNTIME_ERROR"
ERROR_UNKNOWN = "UNKNOWN"


# 앞에 있는 규칙이 먼저 적용된다
_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        ERROR_CUDA_OOM,
        (
            "cuda out of memory",
            "outofmemoryerror",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "not enough memory",
        ),
    ),
    (
        ERROR_DATA,
        (
            "missing required columns",
            "train file not found",
            "validation file not found",
            "test file not found",
            "filenotfounderror",
            "candidate lengths do not match",
            "history sid lengths do not match",
            "expected exactly",
            "duplicate (c1,c2,c3) candidate",
            "no impressions were processed",
            "candidate_labels must contain only",
            "arrowinvalid",
        ),
    ),
    (
        ERROR_RUNTIME,
        (
            "runtimeerror",
            "valueerror",
            "typeerror",
            "assertionerror",
            "keyerror",
            "indexerror",
        ),
    ),
]

# 로그에서 원인 줄로 볼 만한 표시
_MESSAGE_MARKERS = (
    "Error",
    "error:",
    "Exception",
    "Traceback",
)


def classify_error(log_text: str) -> str:
    lowered = log_text.lower()

    for error_type, needles in _PATTERNS:
        for needle in needles:
            if needle in lowered:
                return error_type

    return ERROR_UNKNOWN


def extract_message(log_text: str, max_length: int = 500) -> str:
    # 로그 끝에서 오류로 보이는 줄을 찾아 요약한다.
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]

    if not lines:
        return "로그가 비어 있습니다."

    for line in reversed(lines):
        if any(marker in line for marker in _MESSAGE_MARKERS):
            return line[:max_length]

    return lines[-1][:max_length]


def build_failure_record(
    log_text: str,
    return_code: int,
    log_path: str,
) -> Dict[str, object]:
    return {
        "status": "FAILED",
        "error_type": classify_error(log_text),
        "return_code": return_code,
        "message": extract_message(log_text),
        "log_path": log_path,
    }
