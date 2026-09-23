# 파라미터 탐색 판단 규칙

단계별 탐색에서 "이 단계의 상위 config"를 무엇으로 정하는지를 적어 둔다.

사람이 매번 다르게 판단하면 결과를 재현할 수 없고, 논문에
"어떤 기준으로 하이퍼파라미터를 정했는가"를 쓸 수 없다.
그래서 규칙을 먼저 고정하고 그대로 적용한다.

구현은 `sweep/selection.py`, 값은 `sweep/stages.py`의 `METRIC_PRIORITY`에 있다.
검증은 `python -m sweep.test_selection`.

---

## 0. Test 사용 원칙

**어떤 단계의 선택에도 Test 지표를 쓰지 않는다.**

- Stage 1~6의 판단: Validation 지표만
- Final Top-3 중 최종 설정 선택: Validation seed 평균만
- Test: 최종 설정을 확정한 뒤 **한 번만** 평가하고 보고한다

`sweep/run_final_test.py`는 `seed_robustness/selected_final.json`이
없으면 실행을 거부한다. 그 파일이 곧 "Validation으로 확정했다"는 표시다.

Test 결과를 보고 파라미터를 되돌려 바꾸면 그 순간 Test는
더 이상 일반화 성능의 추정치가 아니다.

---

## 1. 선택 규칙

Validation 지표를 아래 순서로 본다.
앞 지표의 최고값과 tolerance 이내인 후보만 다음 지표로 내려간다.

| 순위 | 지표 | 방향 | tolerance |
|---|---|---|---|
| 1 | Top-1 Accuracy | 클수록 좋음 | 0.005 |
| 2 | MRR | 클수록 좋음 | 0.003 |
| 3 | nDCG@5 | 클수록 좋음 | 0.003 |
| 4 | AUC | 클수록 좋음 | 0.002 |
| 5 | Preference Loss | 작을수록 좋음 | 0.005 |
| 6 | Positive Probability | 클수록 좋음 | 0.002 |

### Score Gap은 선택에 쓰지 않는다 (진단 전용)

`Positive − Negative Score Gap`은 기록하고 **경고에만** 쓴다.

**이유 3가지:**

1. **config마다 스케일이 다르다.**
   score gap은 log 확률 3개 합의 차이다. `d_model`이나 `num_layers`가 바뀌면
   모델의 전체 확신도 수준이 통째로 달라지므로, 서로 다른 구조 사이에서
   "0.01 차이는 같다"고 말할 절대 기준을 정할 수 없다.

2. **여기까지 내려왔다면 이미 노이즈다.**
   앞의 6개 지표가 모두 tolerance 이내라는 것은 순위 품질이 사실상 같다는 뜻이다.
   그 상태의 score gap 차이는 성능 차이가 아니라 확신도의 우연한 변동에 가깝다.

3. **선택에 남겨두면 비용 기준이 죽는다.**
   실제 데이터에서 score gap이 소수점까지 같을 일은 거의 없다.
   tolerance 0으로 이 규칙이 남아 있으면 아래 비용 기준은 영원히 호출되지 않는다.

대신 선택된 config의 score gap이 그 단계 최고값보다
**상대적으로 10% 이상 낮으면 경고**를 남긴다.
절대값이 아니라 비율로 보는 이유는 위 1번과 같다.

> 절대 tolerance(예: 0.01)를 주는 방식도 가능하다.
> 그렇게 하려면 `sweep/stages.py`의 `DIAGNOSTIC_METRICS`에 있는 항목을
> tolerance를 채워 `METRIC_PRIORITY` 끝으로 옮기면 된다.
> 다만 실제 score gap의 크기를 측정하기 전에는 그 값을 정할 근거가 없다.

### 성능이 같을 때 — 비용 기준

지표 7개까지 봐도 동률이면, 성능이 아닌 **비용**으로 고른다.
같은 성능이면 작고 빠르고 메모리를 덜 쓰는 설정이 낫다.

| 순위 | 기준 | 방향 | 이유 |
|---|---|---|---|
| 7 | `total_parameters` | 작을수록 | 과적합 위험이 낮고 추론이 빠르다 |
| 8 | `mean_epoch_seconds` | 작을수록 | 남은 단계의 탐색 비용이 줄어든다 |
| 9 | `peak_gpu_memory_mb` | 작을수록 | 뒤 단계에서 batch를 키울 여유가 생긴다 |
| 10 | `config_hash` | 오름차순 | 실행 순서와 무관하게 항상 같은 결과 |

tolerance는 0이다. 여기까지 왔다는 것은 성능 판단이 이미 끝났다는 뜻이므로
비용은 조금이라도 낮은 쪽을 고른다.

**비용 기준이 성능을 뒤집지는 않는다.** 앞의 7개 지표에서 이미 갈렸다면
모델이 아무리 크고 느려도 성능이 좋은 쪽이 이긴다.

`peak_gpu_memory_mb`는 각 run에서 `torch.cuda.max_memory_allocated`로 측정해
`run_summary.json`과 `summary.csv`에 저장한다.
CPU 학습이면 값이 비어 있고, 그 경우 이 기준은 건너뛴다.

마지막 `config_hash` 기준 덕분에 **선택 결과는 실행 순서에 의존하지 않는다.**
같은 입력이면 몇 번을 돌려도 같은 설정이 뽑힌다.

### 선정 우선순위에 넣지 않는 지표

| 지표 | 이유 |
|---|---|
| `total_loss` | `lambda_preference = 1`이라 Preference Loss와 값이 같다 |
| `negative_prob` | softmax 합이 1이라 Positive Probability에서 계산된다 |
| `nDCG@10` | 후보가 5개라 rank가 항상 10 이하이므로 nDCG@5와 항상 같다 |
| `score_gap` | config 간 스케일이 달라 비교 불가. 경고에만 사용 (위 참고) |

기록은 모두 남긴다. 선정에만 쓰지 않는다.

### tolerance가 필요한 이유

tolerance가 0이면 사실상 1순위 지표만 쓰인다.
Top-1 Accuracy는 `맞힌 수 / 전체`라 값이 촘촘해서, 서로 다른 config가
정확히 같은 값을 갖는 일이 거의 없기 때문이다.
그러면 2~7순위는 영원히 호출되지 않는다.

### tolerance 기본값의 근거

Validation 5만 건, Top-1이 0.3 근처일 때 순수 표본오차는

```
sqrt(0.3 × 0.7 / 50000) ≈ 0.002
```

seed에 따른 변동은 이보다 크므로 그 2배인 **0.005**를 기본값으로 둔다.
나머지 지표는 값의 범위에 맞춰 비슷한 비율로 정했다.

> **이 값들은 잠정값이다.** 아래 2번으로 실제 표준편차를 재면 그 값으로 바꾼다.

---

## 2. tolerance를 실제 측정값으로 바꾸는 법

Stage 6까지 끝내면 `run_seed_robustness`가 Final Top-3를
seed 3개로 각각 학습한다. 그 결과인 `config_aggregate.csv`의
`std_val_top1`, `std_val_mrr` 등이 곧 실제 표준편차 σ다.

`sweep/stages.py`의 `METRIC_PRIORITY`에서 각 tolerance를 그 σ로 바꾼다.

탐색을 시작하기 전에 σ를 알고 싶다면, 기준 config를 seed만 바꿔
세 번 학습한 뒤 같은 계산을 하면 된다.

**σ 없이는 "A가 B보다 낫다"를 말할 수 없다.**
Top-1 0.3120 대 0.3095가 의미 있는 차이인지 아닌지는
σ를 알아야만 판단할 수 있다.

---

## 3. best epoch의 정의

각 run의 지표는 **Validation Top-1 Accuracy가 가장 높았던 epoch**의 값이다.
`checkpoint_best.pt`와 early stopping도 같은 기준을 쓴다.
(`train_transformer.py`)

즉 2순위 이하 지표를 비교할 때도, 그 지표가 가장 좋았던 epoch가 아니라
**Top-1 기준 best epoch에서 측정된 값**을 쓴다.
실제로 저장되는 체크포인트가 하나이므로 이렇게 맞추는 것이 일관적이다.

---

## 4. 경고 조건

아래에 해당하면 요약 CSV의 `warnings` 칸에 남는다.
자동 선택을 막지는 않지만, **사람이 확인해야 하는 신호**다.

### run 단위

| 조건 | 의미 |
|---|---|
| `best_epoch <= 2` | 학습이 거의 진행되지 않음. config 자체를 의심 |
| 마지막 epoch가 best이고 early stopping이 안 걸림 | 아직 개선 중. epoch 예산 부족 |
| `val_top1_accuracy <= 0.2` | 후보 5개 중 1개이므로 무작위 이하 |
| `val_auc <= 0.5` | 무작위 이하 |
| `train_top1 − val_top1 > 0.15` | 과적합 의심 |

### 단계 단위

| 조건 | 의미 |
|---|---|
| 1등과 꼴등의 Top-1 차이가 tolerance 이내 | **이 파라미터는 영향이 없다.** 기본값 유지를 검토 |
| 선택된 config의 MRR/nDCG@5/AUC가 그 단계 최고값보다 낮음 | 지표 간 판단이 엇갈림. 사람 확인 필요 |

---

## 5. 실패한 run의 처리

실패한 run은 **순위 선정에서 제외한다.**
성능이 나빠서가 아니라 측정 자체가 되지 않은 것이므로
다른 run과 같은 기준으로 비교할 수 없다.

특히 `CUDA_OOM`은 "이 설정이 나쁘다"가 아니라
"현재 장비에서 이 설정을 실행할 수 없다"는 뜻이다.
더 큰 GPU에서는 가장 좋은 설정일 수도 있다.

성공한 run이 `top_k`보다 적으면 자동 선택을 중단한다.
남은 것 중에서 억지로 고르면 탐색이 아니라 우연이 된다.

---

## 6. 단계 구성과 그 한계

| 단계 | 함께 탐색 | 조합 | top-k |
|---|---|---|---|
| 1 | `max_history_length` × `use_sep` | 8 | 2 |
| 2 | `d_model` × `num_heads` | 7 | 2 |
| 3 | `num_layers` | 3 | 2 |
| 4 | `d_ff` | 4 | 2 |
| 5 | `learning_rate` × `batch_size` | 12 | 2 |
| 6 | `dropout` × `weight_decay` | 12 | **3 (Final)** |

전체 조합은 96,768가지라 전수 탐색이 불가능하다.
위 방식은 **84 run**으로 줄인다.

### 함께 탐색하는 이유

| 묶음 | 이유 |
|---|---|
| `max_history_length` × `use_sep` | 둘 다 encoder 입력 시퀀스 길이를 결정한다 |
| `d_model` × `num_heads` | `d_model % num_heads != 0`이면 모델이 에러를 낸다 |
| `learning_rate` × `batch_size` | 학습 dynamics가 연결되어 따로 정하면 잘못된 조합에 빠진다 |
| `dropout` × `weight_decay` | 둘 다 정규화 강도라 한쪽만 보면 총량을 알 수 없다 |

### 알려진 한계

1. **순서 의존성** — 먼저 고정한 값이 뒤 단계에 영향을 준다.
   입력 구조와 모델 용량을 앞에 두고, 학습률과 정규화를 뒤에 두었다.
2. **단계를 넘는 상호작용** — 예를 들어 `d_model`과 `learning_rate`의
   상호작용은 다루지 못한다. top-2를 들고 넘어가는 것으로만 완화한다.
3. **지역 최적** — 단계마다 상위 1개가 아니라 2개를 들고 넘어가
   한 번의 잘못된 선택으로 전체가 틀어지지 않게 한다.

### 권장 마무리

6단계가 끝나면 최종 config로 **5단계(learning_rate × batch_size)를 한 번 더**
돌려 값이 바뀌지 않는지 확인한다.
바뀐다면 순서 의존성이 실제로 작용한 것이므로 그 사실을 논문에 적는다.

---

## 7. Final Top-3와 seed 검증

Stage 6은 다음 단계로 넘길 top-2가 아니라 **Final Top-3**를 남긴다.

탐색은 12 epoch / patience 3의 짧은 예산으로 돌기 때문에
그 결과를 최종 성능으로 쓸 수 없다.
Final Top-3 × seed 3개를 모두 **30 epoch / patience 5**로 새로 학습한다.
**seed 42도 예외 없이 다시 학습한다.**

최종 설정은 다음 순서로 고른다.

1. 평균 Validation Top-1 Accuracy
2. 평균 Validation MRR
3. 평균 Validation nDCG@5
4. 평균 Validation AUC
5. 평균 Validation Preference Loss
6. 평균 Positive Probability
7. **Top-1 표준편차가 작은 설정** (seed에 덜 흔들리는 쪽)
8. `total_parameters` → `mean_epoch_seconds` → `peak_gpu_memory_mb`
9. `config_hash` 오름차순

평균 Score Gap은 `config_aggregate.csv`에 기록하되 선택에는 쓰지 않는다.

seed는 파라미터가 아니다.
성능이 잘 나온 seed를 고르지 않는다.

### seed가 모두 성공한 설정만 비교한다

**seed 3개가 모두 성공한 설정끼리만 비교한다.**

seed 2개만 성공한 설정과 3개가 성공한 설정을 나란히 놓으면,
우연히 나쁜 seed가 실패한 설정이 평균에서 유리해진다.
평균과 표준편차를 비교하려면 표본 수가 같아야 한다.

seed가 모자란 설정은 `config_aggregate.csv`에 남기되 순위에서 제외하고,
제외된 이유를 `decision` 칸에 적는다.

### 하나라도 실패하면 자동 확정하지 않는다

실패한 run이 하나라도 있으면 `--accept-auto`를 써도
`selected_final.json`을 만들지 않고 0이 아닌 코드로 종료한다.

비교되지 않은 설정이 더 나았을 가능성을 모른 채 Test로 넘어가면 안 되기 때문이다.

`--retry-failed`로 실패한 run을 다시 실행하거나,
그래도 진행하겠다면 `selected_final_auto.json`을
`selected_final.json`으로 직접 복사해야 한다.
사람이 명시적으로 결정하게 만드는 장치다.

---

## 8. 논문에 쓸 문장 (초안)

> 전체 하이퍼파라미터 조합은 96,768가지로 전수 탐색이 불가능하여,
> 상호작용이 강한 파라미터를 같은 단계에서 함께 탐색하는
> 6단계 순차 탐색(coordinate descent)을 사용하였다.
> 각 단계에서 상위 2개 설정을 다음 단계로 전달하여 총 84회 학습하였다.
> 설정 선택은 Validation Top-1 Accuracy를 주 기준으로 하고,
> 차이가 seed 간 표준편차 이내인 경우 MRR, nDCG@5, AUC,
> Preference Loss 순으로 비교하였다.
> 각 설정의 성능은 Validation Top-1 Accuracy가 최대인 epoch에서 측정하였다.

> 마지막 단계에서 선정한 상위 3개 설정을 3개의 seed(42, 123, 2026)로
> 각각 전체 예산(30 epoch, patience 5)으로 재학습하고,
> Validation 지표의 seed 평균으로 최종 설정을 확정하였다.
> Test 데이터는 최종 설정 확정 이후 평가에만 사용하였다.

> 후보가 1 positive + 4 negative로 고정되어 있어 nDCG@10은 nDCG@5와
> 항상 동일한 값을 가지므로 별도로 보고하지 않는다.


---

## 9. 데이터 경로

gin config는 `Transformer/` 기준 상대경로를 쓴다.

```
train.train_path      = "datasets/ebnerd/train_sequences_1pos4neg.parquet"
train.validation_path = "datasets/ebnerd/validation_sequences_1pos4neg_half.parquet"
```

공용 서버에서는 데이터를 각자 폴더에 복사하지 않고 링크로 연결한다.

```bash
ln -s ~/shared/datasets ~/<본인이름>/<repo>/Transformer/datasets
```

`split_validation.py`와 `predict_sid.py`도 같은 상대경로를 쓰므로
이 링크 하나로 모든 스크립트가 같은 데이터를 본다.
절대경로나 `~` 표기를 gin에 직접 적어도 동작한다.

서버 배치:

```
~/shared/
├── raw/                  공통 전처리 산출물
│   ├── mind/
│   └── ebnerd/
└── datasets/             Transformer 입력 (gin config가 보는 위치)
    ├── mind/
    │   ├── train_sequences_1pos4neg.parquet
    │   ├── validation_sequences_1pos4neg.parquet
    │   └── validation_sequences_1pos4neg_half.parquet
    └── ebnerd/
        ├── train_sequences_1pos4neg.parquet
        ├── validation_sequences_1pos4neg.parquet
        ├── validation_sequences_1pos4neg_half.parquet
        └── test_sequences_1pos4neg.parquet
```

`validation_..._half.parquet`와 `test_...parquet`는 `split_validation.py`가
원본 validation을 impression 시간순 50:50으로 나누어 만든다.
EB-NeRD 공식 데이터는 test 정답을 공개하지 않으므로 이렇게 만든다.

실제로 어느 경로를 보고 있는지는 `sweep.preflight`가 출력한다.
