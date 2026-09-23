# 단계별 파라미터 탐색 (EB-NeRD 기준)

전체 조합은 96,768가지라 전수 탐색이 불가능하다.
서로 얽힌 파라미터를 같은 단계에서 함께 탐색하고 상위 2개만 넘기는 방식으로
**84 run**으로 줄인다.

판단 기준은 [DECISION_RULE.md](DECISION_RULE.md)에 있다.

**Test 데이터는 파라미터 선택에 쓰지 않는다.**
모든 단계의 판단은 Validation 지표로만 하고,
최종 설정을 확정한 뒤에만 Test를 평가한다.

---

## 데이터 경로 — 먼저 심볼릭 링크를 건다

gin config는 `datasets/<이름>/...` 상대경로를 쓴다.
공용 서버에서는 데이터를 각자 폴더에 복사하지 않고 링크로 연결한다.

```bash
ln -s ~/shared/datasets ~/<본인이름>/<repo>/Transformer/datasets
```

**이 링크 하나가 없으면 아무것도 돌아가지 않는다.**
`split_validation.py`와 `predict_sid.py`도 같은 상대경로를 쓰기 때문이다.

```
~/shared/
├── raw/          공통 전처리 산출물
└── datasets/     Transformer 입력
    ├── mind/     train / validation_half / test
    └── ebnerd/   train / validation_half / test
                        ↑
   Transformer/datasets ─┘  (심볼릭 링크)
```

링크 상태와 실제로 보고 있는 경로는 `sweep.preflight`가 출력한다.
링크가 없으면 만들 명령까지 같이 알려준다.

절대경로나 `~/shared/datasets/...` 표기를 gin에 직접 적어도 동작한다.
다만 위 두 스크립트를 위해 링크는 어차피 필요하다.

`Transformer/datasets`는 `.gitignore`에 들어 있어 커밋되지 않는다.

---

## 설치

PyTorch를 먼저 설치한다. GPU와 CUDA 버전에 맞는 빌드를 써야 한다.

```bash
# 공식 안내: https://pytorch.org/get-started/locally/
# 예 (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

그 다음 나머지를 설치한다.

```bash
cd Transformer
pip install -r requirements.txt
```

`requirements.txt`에 들어 있는 것:
`transformers`, `gin-config`, `numpy`, `pandas`, `pyarrow`,
그리고 `split_validation.py`에서만 쓰는 `polars`.

---

## 전체 순서

```bash
cd Transformer

# 0. 환경 검사
python -m sweep.preflight \
  --config configs/transformer_ebnerd.gin

# 1. Ranking metric 구현 검증
python -m evaluate.test_ranking

# 2. Stage 1 계획 확인
python -m sweep.run_stage \
  --config configs/transformer_ebnerd.gin \
  --stage 1 \
  --out sweep_out/ebnerd \
  --dry-run

# 3. Stage 1 실행
python -m sweep.run_stage \
  --config configs/transformer_ebnerd.gin \
  --stage 1 \
  --out sweep_out/ebnerd

# 4. 결과 확인 후 확정
#    sweep_out/ebnerd/stage_01_max_history_use_sep/summary.csv
python -m sweep.run_stage \
  --config configs/transformer_ebnerd.gin \
  --stage 1 \
  --out sweep_out/ebnerd \
  --accept-auto

# 5. Stage 2~6 반복 (--stage 2 ... --stage 6)

# 6. Final Top-3 seed robustness
python -m sweep.run_seed_robustness \
  --config configs/transformer_ebnerd.gin \
  --sweep-out sweep_out/ebnerd \
  --seeds 42 123 2026 \
  --num-epochs 30 \
  --patience 5

# 7. 최종 설정을 확정한 뒤에만 Test 평가
python -m sweep.run_final_test \
  --config configs/transformer_ebnerd.gin \
  --sweep-out sweep_out/ebnerd \
  --test-path datasets/ebnerd/test_sequences_1pos4neg.parquet
```

MIND로 하려면 `--config configs/transformer_mind.gin --out sweep_out/mind`.

---

## Stage 구성

| Stage | 함께 탐색 | 조합 | 입력 branch | run | 유지 |
|---:|---|---:|---:|---:|---|
| 1 | `max_history_length` × `use_sep` | 8 | 1 | 8 | Top-2 |
| 2 | `d_model` × `num_heads` | 7 | 2 | 14 | Top-2 |
| 3 | `num_layers` | 3 | 2 | 6 | Top-2 |
| 4 | `d_ff` | 4 | 2 | 8 | Top-2 |
| 5 | `learning_rate` × `batch_size` | 12 | 2 | 24 | Top-2 |
| 6 | `dropout` × `weight_decay` | 12 | 2 | 24 | **Final Top-3** |
| | | | | **84** | |

Stage 2는 `d_model % num_heads != 0`이면 모델이 에러를 내므로
9쌍 중 유효한 7쌍만 생성한다. `(256, 6)`과 `(512, 6)`은 만들지 않는다.

---

## 주요 옵션 (`run_stage`)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--config` | (필수) | 기준 gin config |
| `--stage` | (필수) | 단계 번호 1~6 |
| `--out` | `sweep_out/default` | 결과 폴더 |
| `--num-epochs` | 12 | 탐색용 최대 epoch |
| `--patience` | 3 | 탐색용 early stopping patience |
| `--seed` | 42 | 난수 seed |
| `--accept-auto` | 꺼짐 | 자동 선택을 확정하고 다음 단계로 |
| `--keep-optimizer-state` | 꺼짐 | checkpoint에 optimizer 포함 (용량 약 3배) |
| `--fail-fast` | 꺼짐 | 첫 실패에서 중단 |
| `--retry-failed` | 꺼짐 | 실패한 run을 다시 실행 |
| `--dry-run` | 꺼짐 | 실행 계획만 출력 |

탐색 단계에서 `--num-epochs 12 --patience 3`을 기본값으로 둔 이유는,
기준 설정(30 epoch / patience 5)으로는 나쁜 config도 최소 6 epoch를 돌아
탐색 시간이 두 배 이상 늘어나기 때문이다.

최종 학습은 반드시 `30 epoch / patience 5`로 다시 한다.
`run_seed_robustness`가 그 역할을 한다.

---

## 결과 폴더 구조

```
sweep_out/ebnerd/
├── runs/
│   └── <config_hash>/
│       ├── _COMPLETE.json      완료 표시 (있으면 다시 학습하지 않음)
│       ├── _FAILED.json        실패 표시 (원인 분류 포함)
│       ├── bindings.json       이 run에 적용된 설정
│       ├── summary_extra.json  sweep이 넘긴 실행 맥락
│       ├── run_summary.json    best epoch 지표와 실행 정보
│       ├── epoch_history.csv   epoch별 train/val 지표
│       ├── checkpoint_best.pt
│       ├── checkpoint_final.pt
│       └── train_log.txt
├── stage_01_max_history_use_sep/
│   ├── summary.csv             ← 이 파일을 보고 판단한다
│   ├── selected_auto.json      자동 선택 결과
│   └── selected.json           확정본 (다음 단계가 읽는다)
├── ...
├── stage_06_dropout_weight_decay/
│   ├── summary.csv
│   ├── selected_auto.json      Final Top-3
│   └── selected.json
├── seed_robustness/
│   ├── runs/
│   ├── per_seed_results.csv
│   ├── config_aggregate.csv
│   ├── selected_final_auto.json
│   └── selected_final.json
└── final_test/
    ├── per_seed_test_results.csv
    └── test_summary.json
```

---

## 중복 제거와 재개

### run hash

같은 설정은 다시 학습하지 않는다. hash에 들어가는 것:

- 기준 gin config **파일 내용의 SHA-256**
- 학습에 적용되는 모든 gin binding (seed, num_epochs, patience 포함)

들어가지 않는 것 (학습된 모델이 달라지지 않으므로):

- `train.save_dir`, `train.save_every_epoch`, `train.save_optimizer_state`

기준 config를 고치면 같은 override라도 다른 hash가 나오므로,
예전 결과를 잘못 재사용하지 않는다.

기준 config의 값과 같은 설정은 생략한 것과 같게 취급한다.
예를 들어 Stage 1의 `max_history_length=20`이 기준 config의 값과 같다면
그 조합은 기준 설정과 동일한 학습이다.

Git commit SHA와 소스 지문은 `_COMPLETE.json`과 `run_summary.json`에
**기록만** 한다. hash에는 넣지 않는다.
코드를 고칠 때마다 84 run을 전부 다시 돌려야 한다면 탐색이 불가능해진다.

### 중단되어도 이어서 실행된다

`_COMPLETE.json`이 있는 run은 건너뛴다.
Spot 인스턴스가 중단되어도 같은 명령을 다시 실행하면 남은 것부터 진행한다.

---

## 실패 처리

run 하나가 실패해도 **나머지는 계속 진행한다.**

- 실패한 run 폴더에 `_FAILED.json`을 남긴다
- `summary.csv`에 `status=FAILED`로 남기되 **순위 선정에서는 제외한다**
- 성공한 run이 `top_k`보다 적으면 자동 선택을 중단하고 오류를 출력한다

`error_type` 분류:

| 값 | 의미 |
|---|---|
| `CUDA_OOM` | 현재 장비에서 실행할 수 없는 조합. 성능이 나쁜 설정이 아니다 |
| `DATA_ERROR` | 파일 경로, 컬럼, 후보 구조 문제. 보통 모든 run이 같이 실패한다 |
| `RUNTIME_ERROR` | 그 외 학습 중 오류 |
| `UNKNOWN` | 분류하지 못함 (OOM killer 등) |

다시 돌리려면:

```bash
python -m sweep.run_stage ... --retry-failed
```

기존 실패 폴더는 **지우지 않고** `<hash>.failed.<timestamp>` 이름으로 옮긴 뒤
새로 학습한다. 같은 오류가 반복되는지 나중에 비교할 수 있다.

첫 실패에서 바로 멈추려면 `--fail-fast`를 쓴다.

---

## Preflight

84 run을 돌리다가 30번째에서 데이터 문제로 실패하는 일을 막는다.

```bash
python -m sweep.preflight --config configs/transformer_ebnerd.gin
python -m sweep.preflight --config configs/transformer_ebnerd.gin --full-data-check
```

검사 항목: 패키지 import, CUDA와 AMP dtype, train/validation parquet 존재,
필수 컬럼, 후보 5개, positive 1개, SID가 vocab 범위 안인지,
`d_model % num_heads`, 출력 폴더 쓰기 권한, 디스크 여유,
합성 batch로 model forward와 loss.

기본은 앞쪽 2,000행만 본다. `--full-data-check`는 전체를 확인한다(느림).
통과하면 `PREFLIGHT PASSED`를 출력하고 0으로 종료한다.

---

## Seed robustness 안전장치

`run_seed_robustness`는 다음을 지킨다.

- **모든 seed가 성공한 설정만 비교한다.** 표본 수가 다르면 평균과 표준편차를
  나란히 놓을 수 없다. seed가 모자란 설정은 CSV에 남기되 순위에서 제외한다.
- **하나라도 실패하면 `--accept-auto`로 확정되지 않는다.**
  `selected_final.json`을 만들지 않고 0이 아닌 코드로 종료한다.
  `--retry-failed`로 다시 실행하거나 사람이 직접 복사해야 한다.

```bash
python -m sweep.run_seed_robustness ... --retry-failed --accept-auto
```

---

## 성능이 같을 때의 선택 기준

Validation 지표 **6개**가 모두 tolerance 이내로 같으면 비용으로 고른다.

```
Top-1 → MRR → nDCG@5 → AUC → Preference Loss → Positive Probability
   → total_parameters ↑ → mean_epoch_seconds ↑ → peak_gpu_memory_mb ↑ → config_hash ↑
```

`Positive−Negative Score Gap`은 **선택에 쓰지 않는다.** config마다 스케일이 달라
비교 기준을 정할 수 없고, 선택에 남겨두면 실제 데이터에서 값이 항상 달라
비용 기준이 한 번도 호출되지 않기 때문이다.
대신 최고값보다 상대적으로 10% 이상 낮으면 경고로 알린다.
자세한 근거는 [DECISION_RULE.md](DECISION_RULE.md) 참고.

`peak_gpu_memory_mb`는 run마다 `torch.cuda.max_memory_allocated`로 측정해
`run_summary.json`과 `summary.csv`에 저장한다.
STEP 5에서 batch 256이 들어갈 여유가 있는지 판단할 때도 이 값을 본다.

마지막 `config_hash` 기준 덕분에 선택 결과는 실행 순서에 의존하지 않는다.

---

## 자동 선택을 바꾸고 싶을 때

`selected_auto.json`의 `configs`를 고쳐 `selected.json`으로 저장한다.

```json
{
  "stage": 1,
  "stage_name": "max_history_use_sep",
  "top_k": 2,
  "configs": [
    {
      "NewsSequenceDataset.max_history_length": 30,
      "NewsEncoderDecoderTransformer.use_sep": true
    },
    {
      "NewsSequenceDataset.max_history_length": 20,
      "NewsEncoderDecoderTransformer.use_sep": true
    }
  ]
}
```

다음 단계는 `selected.json`만 읽는다.

---

## 테스트

```bash
cd Transformer
python -m compileall .

python -m evaluate.test_ranking    # ranking metric이 기준 구현과 일치하는지
python -m sweep.test_stages        # Stage 구성과 84 run
python -m sweep.test_selection     # 지표 우선순위와 실패 run 처리
python -m sweep.test_hashing       # hash가 무엇에 반응하는지
```
