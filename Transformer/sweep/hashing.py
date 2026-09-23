"""
run의 신원(hash)을 정하는 규칙.

같은 hash면 같은 학습이므로 이전 결과를 재사용한다.
따라서 학습 결과를 바꾸는 값은 모두 hash에 들어가야 한다.

hash에 포함하는 것:
  - 기준 gin config 파일 내용의 SHA-256
    (같은 override라도 기준 config가 바뀌면 다른 학습이다)
  - 학습에 적용되는 모든 gin binding
    (seed, num_epochs, early_stopping_patience 포함)

hash에서 빼는 것:
  - 결과 저장 위치와 저장 방식
    (train.save_dir / save_every_epoch / save_optimizer_state)
    학습된 모델 자체는 달라지지 않는다.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


# 학습 결과에 영향을 주지 않는 binding
HASH_EXCLUDED_BINDINGS = frozenset(
    {
        "train.save_dir",
        "train.save_every_epoch",
        "train.save_optimizer_state",
    }
)


# "Class.param = value" 형태의 gin binding 한 줄
_BINDING_PATTERN = re.compile(
    r"^\s*([A-Za-z_][\w]*\.[A-Za-z_][\w]*)\s*=\s*(.+?)\s*$"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_gin_bindings(text: str) -> Dict[str, Any]:
    # "Class.param = value" 형태의 줄에서 값을 읽는다.
    # gin config 파일과 gin.config_str() 출력 양쪽에 쓴다.
    #
    # 값을 읽지 못하는 줄은 건너뛴다. 기준 config 내용 자체는
    # SHA-256으로 hash에 들어가므로 정확성에는 문제가 없고,
    # 중복 제거가 조금 덜 될 뿐이다.
    bindings: Dict[str, Any] = {}

    for line in text.splitlines():
        line = line.split("#", 1)[0]

        matched = _BINDING_PATTERN.match(line)

        if matched is None:
            continue

        try:
            bindings[matched.group(1)] = ast.literal_eval(matched.group(2))
        except (ValueError, SyntaxError):
            continue

    return bindings


def load_base_bindings(path: Path) -> Dict[str, Any]:
    # 기준 config에 이미 들어 있는 값을 읽는다.
    #
    # 예를 들어 2단계에서 어떤 값이 기준 config의 값과 같다면
    # 앞 단계에서 이미 학습한 config와 동일하므로 다시 돌릴 필요가 없다.
    return parse_gin_bindings(path.read_text(encoding="utf-8"))


def strip_defaults(
    bindings: Dict[str, Any],
    base_bindings: Dict[str, Any],
) -> Dict[str, Any]:
    # 기준 config와 값이 같은 binding은 뺀다.
    # 학습에는 원래 binding을 그대로 넘기고,
    # 이 결과는 run의 신원을 정할 때만 쓴다.
    return {
        key: value
        for key, value in bindings.items()
        if key not in base_bindings or base_bindings[key] != value
    }


def hash_payload(
    base_config_path: Path,
    bindings: Dict[str, Any],
    base_bindings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # hash를 만드는 재료. 디버깅을 위해 따로 꺼내 볼 수 있게 한다.
    if base_bindings is None:
        base_bindings = load_base_bindings(base_config_path)

    significant = {
        key: value
        for key, value in strip_defaults(bindings, base_bindings).items()
        if key not in HASH_EXCLUDED_BINDINGS
    }

    return {
        "base_config_sha256": file_sha256(base_config_path),
        "bindings": {key: significant[key] for key in sorted(significant)},
    }


def compute_config_hash(
    base_config_path: Path,
    bindings: Dict[str, Any],
    base_bindings: Optional[Dict[str, Any]] = None,
    length: int = 12,
) -> str:
    payload = json.dumps(
        hash_payload(
            base_config_path=base_config_path,
            bindings=bindings,
            base_bindings=base_bindings,
        ),
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def params_hash(bindings: Dict[str, Any], length: int = 12) -> str:
    # seed와 무관하게 "이 파라미터 조합"만 식별한다.
    # seed 검증에서 설정끼리 동점일 때의 최종 판단 기준으로 쓴다.
    payload = json.dumps(
        {key: bindings[key] for key in sorted(bindings)},
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def source_fingerprint(root: Path) -> Dict[str, Optional[str]]:
    # 어떤 코드로 돌린 결과인지 기록해 둔다. hash에는 넣지 않는다.
    # 코드를 고칠 때마다 모든 run을 다시 돌려야 한다면 탐색이 불가능해진다.
    commit: Optional[str] = None
    dirty: Optional[bool] = None

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())

    except (subprocess.CalledProcessError, OSError):
        commit = None

    # git을 쓸 수 없는 환경에서도 코드 버전을 구분할 수 있게
    # Transformer 폴더의 .py 내용으로 지문을 만든다.
    digest = hashlib.sha256()

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue

        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": digest.hexdigest()[:16],
    }
