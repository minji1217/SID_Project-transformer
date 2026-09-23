# 단계별 파라미터 탐색

전체 조합(96,768가지) 대신, 영향이 큰 파라미터부터 순서대로 정해 나가며
**62 run**으로 최적 설정을 찾는다.

판단 기준은 [DECISION_RULE.md](DECISION_RULE.md)에 있다.

---

## 사용법

```bash
cd Transformer

# 1) 무엇이 실행될지 먼저 확인
python -m sweep.run_stage \
  --config configs/transformer_mind.gin \
  --stage 1 --out sweep_out/mind --dry-run

# 2) 실행
python -m sweep.run_stage \
  --config configs/transformer_mind.gin \
  --stage 1 --out sweep_out/mind

# 3) 결과 확인
#    sweep_out/mind/stage_01_learning_rate/summary.csv

# 4) 자동 선택을 그대로 확정
python -m sweep.run_stage \
  --config configs/transformer_mind.gin \
  --stage 1 --out sweep_out/mind --accept-auto

# 5) 다음 단계
python -m sweep.run_stage \
  --config configs/transformer_mind.gin \
  --stage 2 --out sweep_out/mind
```

9단계까지 반복한다.
EBNeRD로 하려면 `--config configs/transformer_ebnerd.gin --out sweep_out/ebnerd`.

---

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--config` | (필수) | 기준 gin config |
| `--stage` | (필수) | 단계 번호 1~9 |
| `--out` | `sweep_out/default` | 결과 폴더 |
| `--num-epochs` | 12 | 탐색용 최대 epoch (최종 학습은 30) |
| `--patience` | 3 | 탐색용 early stopping patience |
| `--seed` | 42 | 난수 seed |
| `--accept-auto` | 꺼짐 | 자동 선택을 확정하고 바로 다음 단계로 |
| `--keep-optimizer-state` | 꺼짐 | checkpoint에 optimizer 포함 (용량 약 3배) |
| `--dry-run` | 꺼짐 | 실행 계획만 출력 |

탐색 단계에서 `--num-epochs 12 --patience 3`을 기본값으로 둔 이유는,
기본 설정(30 epoch / patience 5)으로는 나쁜 config도 최소 6 epoch를 돌아
탐색 시간이 두 배 이상 늘어나기 때문이다.

---

## 결과 폴더 구조

```
sweep_out/mind/
├── runs/
│   └── <config_hash>/
│       ├── _COMPLETE.json      완료 표시 (있으면 다시 학습하지 않음)
│       ├── bindings.json       이 run에 적용된 설정
│       ├── run_summary.json    best epoch 지표
│       ├── epoch_history.csv   epoch별 train/val 지표
│       ├── checkpoint_best.pt
│       ├── checkpoint_final.pt
│       └── train_log.txt
├── stage_01_learning_rate/
│   ├── summary.csv             ← 이 파일을 보고 판단한다
│   ├── selected_auto.json      자동 선택 결과
│   └── selected.json           확정본 (다음 단계가 읽는다)
├── stage_02_max_history_length/
└── ...
```

### 중복 제거

같은 설정은 다시 학습하지 않는다.
설정이 기준 config의 값과 같으면 같은 run으로 본다.

예를 들어 2단계의 `max_history_length=20`은 기준 config의 값과 같으므로,
1단계에서 이미 학습한 결과를 그대로 쓴다.
단계마다 1~2개씩 절약되어 전체로는 10회 안팎이 줄어든다.

### 중단되어도 이어서 실행된다

`_COMPLETE.json`이 있는 run은 건너뛴다.
Spot 인스턴스가 중단되어도 같은 명령을 다시 실행하면 남은 것부터 진행한다.

완료 표시가 없는 폴더를 만나면 **덮어쓰지 않고 멈춘다.**
실패한 run의 흔적일 수 있으므로 `train_log.txt`를 확인한 뒤 직접 지운다.

---

## 자동 선택을 바꾸고 싶을 때

`selected_auto.json`의 `configs`를 원하는 설정으로 고쳐
`selected.json`으로 저장하면 된다.

```json
{
  "stage": 1,
  "stage_name": "learning_rate",
  "top_k": 2,
  "configs": [
    { "train.learning_rate": 0.0002 },
    { "train.learning_rate": 0.0001 }
  ]
}
```

다음 단계는 `selected.json`만 읽는다.

---

## 탐색이 끝난 뒤

1. 최종 config로 **30 epoch, patience 5**로 다시 학습한다.
2. seed를 3개 이상 바꿔 돌려 분산을 함께 보고한다.
3. `predict_sid.py` + `evaluate/metrics.py`로 테스트 성능을 측정한다.
4. 최종 config에서 **1단계를 한 번 더** 돌려 learning_rate가 바뀌지 않는지 확인한다.
