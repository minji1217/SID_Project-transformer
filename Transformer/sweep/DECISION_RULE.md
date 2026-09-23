# 파라미터 탐색 판단 규칙

단계별 탐색에서 "이 단계의 상위 config"를 무엇으로 정하는지를 적어 둔다.

사람이 매번 다르게 판단하면 결과를 재현할 수 없고, 논문에
"어떤 기준으로 하이퍼파라미터를 정했는가"를 쓸 수 없다.
그래서 규칙을 먼저 고정하고 그대로 적용한다.

구현은 `sweep/selection.py`, 값은 `sweep/stages.py`의 `METRIC_PRIORITY`에 있다.

---

## 1. 선택 규칙

Validation 지표를 아래 순서로 본다.
앞 지표의 차이가 tolerance 이내면 **동률**로 보고 다음 지표로 내려간다.

| 순위 | 지표 | 방향 | tolerance |
|---|---|---|---|
| 1 | Top-1 Accuracy | 클수록 좋음 | 0.005 |
| 2 | MRR | 클수록 좋음 | 0.003 |
| 3 | nDCG@5 | 클수록 좋음 | 0.003 |
| 4 | AUC | 클수록 좋음 | 0.002 |
| 5 | Preference Loss | 작을수록 좋음 | 0.005 |
| 6 | Positive Probability | 클수록 좋음 | 0.002 |
| 7 | Positive − Negative Score Gap | 클수록 좋음 | 0 |

마지막까지 동률이면 먼저 실행된 run을 선택한다.

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

기준 config를 seed만 바꿔 3번 돌린다.

```bash
cd Transformer
for SEED in 42 43 44; do
  python -m sweep.run_stage \
    --config configs/transformer_mind.gin \
    --stage 1 --out sweep_out/noise_seed_$SEED \
    --seed $SEED --accept-auto
done
```

같은 config에서 나온 세 값의 표준편차 σ를 지표마다 구하고,
`sweep/stages.py`의 `METRIC_PRIORITY`에서 각 tolerance를 σ로 바꾼다.

**이 측정 없이는 "A가 B보다 낫다"를 말할 수 없다.**
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

## 5. 단계 순서와 그 한계

| 단계 | 파라미터 | top-k |
|---|---|---|
| 1 | learning_rate | 2 |
| 2 | max_history_length | 2 |
| 3 | d_model + num_heads (함께) | 2 |
| 4 | num_layers | 2 |
| 5 | d_ff | 2 |
| 6 | dropout | 2 |
| 7 | weight_decay | 2 |
| 8 | batch_size | 1 |
| 9 | use_sep | 1 |

전체 조합은 96,768가지라 전수 탐색이 불가능하다.
위 방식은 **62 run**으로 줄인다.

### 알려진 한계

1. **순서 의존성** — 먼저 고정한 값이 뒤 단계에 영향을 준다.
   영향이 크다고 알려진 learning_rate와 max_history_length를 앞에 두어 완화한다.
2. **상호작용을 일부 놓침** — `d_model`과 `num_heads`처럼 강하게 얽힌 것은
   한 단계에서 함께 탐색한다. `learning_rate`와 `batch_size`도 얽혀 있으나
   batch_size를 마지막 쪽에 두는 것으로만 다룬다.
3. **지역 최적** — 단계마다 상위 1개가 아니라 2개를 들고 넘어가
   한 번의 잘못된 선택으로 전체가 틀어지지 않게 한다.

### 권장 마무리

9단계가 끝나면 최종 config로 **1단계(learning_rate)를 한 번 더** 돌려
값이 바뀌지 않는지 확인한다 (4 run 추가).
바뀐다면 순서 의존성이 실제로 작용한 것이므로 그 사실을 논문에 적는다.

---

## 6. 논문에 쓸 문장 (초안)

> 전체 하이퍼파라미터 조합은 96,768가지로 전수 탐색이 불가능하여,
> 영향이 큰 파라미터부터 순차적으로 탐색하는 단계적 탐색(coordinate descent)을
> 사용하였다. 각 단계에서 상위 2개 설정을 다음 단계로 전달하여 총 62회 학습하였다.
> 설정 선택은 Validation Top-1 Accuracy를 주 기준으로 하고,
> 차이가 seed 간 표준편차 이내인 경우 MRR, nDCG@5, AUC 순으로 비교하였다.
> 각 설정의 성능은 Validation Top-1 Accuracy가 최대인 epoch에서 측정하였다.

> 후보가 1 positive + 4 negative로 고정되어 있어 nDCG@10은 nDCG@5와
> 항상 동일한 값을 가지므로 보고하지 않는다.
