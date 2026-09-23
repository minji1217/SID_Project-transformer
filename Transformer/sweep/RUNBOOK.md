# 파라미터 탐색 실행 안내서

이 문서는 **처음 실행하는 사람**과 **그 사람을 돕는 AI 어시스턴트**를 위한 것이다.
명령을 순서대로 따라가면 되고, 각 단계에서 무엇을 판단해야 하는지도 함께 적었다.

- 왜 이렇게 설계했는지: [DECISION_RULE.md](DECISION_RULE.md)
- 옵션과 폴더 구조 상세: [README.md](README.md)

---

## 0. 한 줄 요약

> 하이퍼파라미터 10개를 6단계로 나눠 탐색한다.
> 각 단계에서 상위 2개 설정만 다음 단계로 넘긴다.
> 총 84번 학습하고, 마지막에 상위 3개를 seed 3개로 재학습해 최종 1개를 고른다.

전체 조합은 96,768가지라 전수 탐색이 불가능해서 이렇게 줄였다.

---

## 1. 무엇을 하는 실험인가

Transformer가 사용자의 기사 열람 이력(history)을 보고,
후보 기사 5개(정답 1 + 오답 4) 중 어느 것을 클릭할지 맞히는 모델이다.
이 모델의 하이퍼파라미터를 고른다.

```
사용자 history: 기사별 Semantic ID (c1,c2,c3,c4)
후보 기사 5개 : (c1,c2,c3)
모델 출력     : 후보 5개의 점수
정답          : 점수가 가장 높은 후보가 실제 클릭한 기사인가
```

### 단계 구성

| STEP | 함께 탐색하는 파라미터 | 조합 | 입력 branch | run 수 | 다음으로 |
|---:|---|---:|---:|---:|---|
| 1 | `max_history_length` × `use_sep` | 8 | 1 | 8 | 상위 2개 |
| 2 | `d_model` × `num_heads` | 7 | 2 | 14 | 상위 2개 |
| 3 | `num_layers` | 3 | 2 | 6 | 상위 2개 |
| 4 | `d_ff` | 4 | 2 | 8 | 상위 2개 |
| 5 | `learning_rate` × `batch_size` | 12 | 2 | 24 | 상위 2개 |
| 6 | `dropout` × `weight_decay` | 12 | 2 | 24 | **Final Top-3** |
| | | | | **84** | |

서로 영향을 주는 파라미터는 같은 단계에서 함께 본다.
따로 정하면 잘못된 조합에 빠지기 때문이다.

### 한 단계씩 실행하는 이유

STEP 2에서 무엇을 학습할지는 STEP 1이 끝나야 정해진다.
그래서 84개를 한 번에 돌릴 수 없고, 단계마다 멈춰서 결과를 확인한 뒤 진행한다.

---

## 2. 서버 환경

| | |
|---|---|
| 인스턴스 | 팀 공용 (`agent-team-server`) |
| GPU | NVIDIA L4 24GB, bf16 지원 |
| 접속 | EC2 Instance Connect (브라우저) 또는 SSH |
| 데이터 | `~/shared/datasets/{mind,ebnerd}/` |
| 작업 폴더 | `~/<본인이름>/` |

### 데이터 현황

| | train | validation | test |
|---|---:|---:|---:|
| mind | 214,962 | 52,920 | 51,777 |
| ebnerd | 232,874 | 122,549 | 122,668 |

`validation`과 `test`는 원본 validation을 impression 시간순 50:50으로 나눈 것이다.
(`split_validation.py`) EB-NeRD 공식 데이터가 test 정답을 공개하지 않기 때문이다.

---

## 3. 준비 (처음 한 번만)

아래 예시는 MIND 기준이고, 폴더 이름 `yeomin`은 본인 것으로 바꾼다.

```bash
# 1) 본인 폴더에 clone
cd ~/yeomin
git clone -b claude/hopeful-mendel-fhc2n4 \
  https://github.com/minji1217/SID_Project-transformer.git

# 2) 데이터 링크 (팀 규칙: 복사하지 말고 링크)
ln -s ~/shared/datasets ~/yeomin/SID_Project-transformer/Transformer/datasets

# 3) 패키지
cd ~/yeomin/SID_Project-transformer/Transformer
pip install gin-config transformers pyarrow polars

# 4) 검사
python -m sweep.preflight --config configs/transformer_mind.gin
```

**`PREFLIGHT PASSED`가 나와야 다음으로 간다.**

`preflight`가 검사하는 것: 패키지, CUDA, AMP dtype, 데이터 링크, parquet 경로,
필수 컬럼, 후보 5개, positive 1개, SID가 vocab 범위 안인지,
`d_model % num_heads`, 출력 폴더 쓰기 권한, 디스크 용량, 합성 batch forward/loss.

### ⚠️ torch를 다시 설치하지 말 것

서버의 PyTorch는 CUDA에 맞춰 빌드된 DLAMI 제공 버전이다.
`pip install -r requirements.txt`를 그대로 쓰면 torch를 덮어쓸 수 있으니,
위처럼 **부족한 패키지만 개별 설치**한다.

---

## 4. 실행

### tmux 안에서 실행할 것

84 run은 하루 이상 걸린다. 터미널이 끊기면 학습이 죽는다.

```bash
tmux new -s sweep-mind
cd ~/yeomin/SID_Project-transformer/Transformer
```

| | |
|---|---|
| 나가기 (학습은 계속) | `Ctrl+B` 누르고 `D` |
| 다시 붙기 | `tmux attach -t sweep-mind` |
| 새 창 (모니터링용) | `Ctrl+B` 누르고 `C` |
| 창 전환 | `Ctrl+B` 누르고 `0` / `1` |

### STEP 1~6 반복

`--stage` 숫자만 1 → 6으로 바꿔가며 **같은 3단계를 6번** 반복한다.

```bash
STAGE=1   # 1 → 2 → 3 → 4 → 5 → 6

# (1) 실행
python -m sweep.run_stage \
  --config configs/transformer_mind.gin \
  --stage $STAGE --out sweep_out/mind

# (2) 결과 확인
cat sweep_out/mind/stage_0${STAGE}_*/summary.csv

# (3) 확정 (상위 2개를 다음 단계로 전달)
python -m sweep.run_stage \
  --config configs/transformer_mind.gin \
  --stage $STAGE --out sweep_out/mind --accept-auto
```

(3)은 이미 학습한 것을 다시 돌리지 않는다. 확정만 한다.

**EB-NeRD로 하려면** `transformer_mind.gin` → `transformer_ebnerd.gin`,
`sweep_out/mind` → `sweep_out/ebnerd` 두 곳만 바꾼다.

### 단계 폴더 이름

```
stage_01_max_history_use_sep
stage_02_d_model_num_heads
stage_03_num_layers
stage_04_d_ff
stage_05_learning_rate_batch_size
stage_06_dropout_weight_decay
```

### 탐색 예산

탐색 단계는 기본 **최대 12 epoch / patience 3**으로 빠르게 비교한다.
이 짧은 학습은 최종 모델이 아니라 후보를 거르는 용도다.
최종 학습은 6단계 뒤 seed 검증에서 30 epoch으로 다시 한다.

---

## 5. STEP 6 이후

### Seed 검증 (9 run)

```bash
python -m sweep.run_seed_robustness \
  --config configs/transformer_mind.gin \
  --sweep-out sweep_out/mind \
  --seeds 42 123 2026 \
  --num-epochs 30 --patience 5
```

Final Top-3 × seed 3개를 **전부 30 epoch으로 새로 학습**한다 (총 9 run).
12 epoch 탐색 결과는 조건이 달라 재사용하지 않는다. **seed 42도 다시 학습한다.**

결과 확인 후 확정:

```bash
cat sweep_out/mind/seed_robustness/config_aggregate.csv

python -m sweep.run_seed_robustness \
  --config configs/transformer_mind.gin \
  --sweep-out sweep_out/mind \
  --seeds 42 123 2026 --num-epochs 30 --patience 5 --accept-auto
```

### 최종 Test (마지막에 딱 한 번)

```bash
python -m sweep.run_final_test \
  --config configs/transformer_mind.gin \
  --sweep-out sweep_out/mind \
  --test-path datasets/mind/test_sequences_1pos4neg.parquet
```

`selected_final.json`이 없으면 실행 자체를 거부한다.
Validation으로 최종 설정을 확정한 뒤에만 Test를 본다는 원칙을 코드가 강제한다.

---

## 6. 어떤 기준으로 고르는가

각 단계에서 상위 config를 고르는 규칙이다. 코드가 자동으로 적용하고,
판단 근거를 `summary.csv`의 `decision` 칸에 남긴다.

### 성능 지표 (앞에서 갈리면 거기서 끝)

| 순위 | 지표 | 방향 | tolerance |
|---:|---|---|---:|
| 1 | Validation Top-1 Accuracy | 클수록 | 0.005 |
| 2 | Validation MRR | 클수록 | 0.003 |
| 3 | Validation nDCG@5 | 클수록 | 0.003 |
| 4 | Validation AUC | 클수록 | 0.002 |
| 5 | Validation Preference Loss | 작을수록 | 0.005 |
| 6 | Positive Probability | 클수록 | 0.002 |

앞 지표의 최고값과 tolerance 이내면 **동률**로 보고 다음 지표로 내려간다.

### 성능이 같으면 비용으로

| 순위 | 기준 | 방향 |
|---:|---|---|
| 7 | `total_parameters` | 작을수록 |
| 8 | `mean_epoch_seconds` | 작을수록 |
| 9 | `peak_gpu_memory_mb` | 작을수록 |
| 10 | `config_hash` | 오름차순 |

마지막 기준 덕분에 선택 결과가 실행 순서에 의존하지 않는다.

### 선정에 쓰지 않는 값

| 지표 | 이유 |
|---|---|
| `total_loss` | `lambda_preference=1`이라 Preference Loss와 값이 같다 |
| `negative_prob` | Positive Probability에서 계산된다 |
| `nDCG@10` | 후보가 5개라 nDCG@5와 항상 같다 |
| `score_gap` | config마다 스케일이 달라 비교 불가. 경고에만 사용 |

### 경고가 뜨면 사람이 봐야 한다

`summary.csv`의 `warnings` 칸에 아래가 남는다.

| 경고 | 의미 |
|---|---|
| `best_epoch <= 2` | 학습이 거의 진행되지 않음. config 의심 |
| 마지막 epoch가 best | 아직 개선 중. epoch 예산 부족 |
| `val_top1 <= 0.2` | 후보 5개 중 1개이므로 무작위 이하 |
| `val_auc <= 0.5` | 무작위 이하 |
| train-val 격차 > 0.15 | 과적합 의심 |
| 1등과 꼴등 차이가 tolerance 이내 | **이 파라미터는 영향이 없다.** 기본값 유지 검토 |
| 선택된 config의 MRR/nDCG/AUC가 최고값보다 낮음 | 지표 간 판단이 엇갈림. 확인 필요 |

---

## 7. 결과 파일

```
sweep_out/mind/
├── runs/<config_hash>/
│   ├── _COMPLETE.json      완료 표시 (있으면 다시 학습하지 않음)
│   ├── _FAILED.json        실패 표시 (원인 분류 포함)
│   ├── bindings.json       이 run에 적용된 설정
│   ├── run_summary.json    best epoch 지표와 실행 정보
│   ├── epoch_history.csv   epoch별 train/val 지표
│   ├── checkpoint_best.pt
│   ├── checkpoint_final.pt
│   └── train_log.txt
├── stage_01_max_history_use_sep/
│   ├── summary.csv         ← 사람이 보는 파일
│   ├── selected_auto.json  자동 선택 결과
│   └── selected.json       확정본 (다음 단계가 읽는다)
├── ...
├── seed_robustness/
│   ├── per_seed_results.csv
│   ├── config_aggregate.csv
│   └── selected_final.json
└── final_test/
    ├── per_seed_test_results.csv
    └── test_summary.json
```

**확인할 파일은 `summary.csv` 하나면 된다.** 나머지는 필요할 때 본다.

---

## 8. 문제가 생기면

### 중간에 끊겼다 / 서버를 껐다

끝난 run은 보존된다. 같은 명령에 `--retry-failed`만 붙여 다시 실행한다.

```bash
python -m sweep.run_stage --config configs/transformer_mind.gin \
  --stage 1 --out sweep_out/mind --retry-failed
```

- 완료된 run → 건너뜀
- 끊긴 run → 다시 학습
- 남은 run → 이어서 진행

### 일부 run이 실패했다

**나머지는 계속 진행된다.** 실패한 run은 `summary.csv`에 `status=FAILED`로 남고
순위 선정에서는 제외된다.

| `error_type` | 의미 | 대응 |
|---|---|---|
| `CUDA_OOM` | 현재 GPU에서 실행 불가능한 조합 | 성능이 나쁜 게 아니다. 그대로 두면 된다 |
| `DATA_ERROR` | 경로·컬럼·후보 구조 문제 | 보통 모든 run이 같이 실패한다. preflight 재실행 |
| `RUNTIME_ERROR` | 그 외 학습 중 오류 | `train_log.txt` 확인 |
| `UNKNOWN` | 분류 실패 (OOM killer 등) | `train_log.txt` 확인 |

성공한 run이 필요한 개수보다 적으면 자동 선택을 중단하고 오류를 낸다.

### GPU 사용률이 낮다 (30% 미만)

데이터 로딩이 병목이다. `configs/transformer_mind.gin`의
`train.num_workers`를 올린다 (기본 4).

### 디스크가 부족하다

```bash
du -sh sweep_out/mind/runs/* | sort -h | tail
find sweep_out/mind/runs -name "checkpoint_final.pt" -delete   # best는 남긴다
```

---

## 9. 하면 안 되는 것

| | 이유 |
|---|---|
| **Test 결과를 보고 파라미터를 바꾸기** | Test leakage. 그 순간 Test는 일반화 성능의 추정치가 아니게 된다 |
| **torch를 pip으로 덮어쓰기** | DLAMI의 CUDA 맞춤 빌드가 깨진다 |
| **GPU를 다른 사람과 동시에 쓰기** | 둘 다 느려지고, `mean_epoch_seconds` 기준이 오염된다. 시작 전 `nvidia-smi`로 확인 |
| **데이터를 각자 폴더에 복사** | 용량 3배 낭비. 심볼릭 링크를 쓴다 |
| **성능 좋은 seed 고르기** | seed는 파라미터가 아니다 |
| **`selected.json`을 지우고 다시 돌리기** | 앞 단계 판단이 사라진다. 바꾸려면 파일 내용을 편집한다 |

---

## 10. AI 어시스턴트에게 도움을 받을 때

각 STEP이 끝나면 `summary.csv` 내용을 그대로 전달하면 된다.

```bash
cat sweep_out/mind/stage_01_max_history_use_sep/summary.csv
```

### 물어보면 좋은 것

- 상위 2개 선택이 타당한가 (경고가 떴다면 특히)
- 1등과 2등 차이가 의미 있는 차이인가, 노이즈인가
- 경고 메시지가 무엇을 뜻하는가
- 남은 단계가 얼마나 걸릴지

### 어시스턴트가 알아야 할 맥락

- **선택은 Validation 지표로만 한다. Test는 마지막 한 번.**
- 자동 선택 결과를 바꾸려면 `selected_auto.json`을 편집해
  `selected.json`으로 저장한다. 다음 단계는 `selected.json`만 읽는다.
- tolerance 기본값은 잠정치다. seed 검증의 `config_aggregate.csv`에서
  나온 실제 표준편차로 `sweep/stages.py`의 `METRIC_PRIORITY`를 갱신하면
  "의미 있는 차이"의 기준이 정확해진다.
- 각 run의 `decision` 칸에 어느 지표에서 승부가 났는지 기록돼 있다.

### 자가 점검

코드가 의도대로 동작하는지 언제든 확인할 수 있다.

```bash
python -m evaluate.test_ranking    # ranking metric이 기준 구현과 일치하는지
python -m sweep.test_stages        # 6단계 구성과 84 run
python -m sweep.test_selection     # 선택 규칙, 동점 처리, 실패 run 제외
python -m sweep.test_hashing       # 중복 판별 hash
```

모두 `PASSED`가 나와야 한다.
