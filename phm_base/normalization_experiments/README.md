> 전체 실행 설정: 경로·장치 설정 후 전체 실행하면 smoke → 본학습 → 선택 고정 → Test → ZIP 저장 순서로 진행합니다. 준비만 확인하려면 RUN_SMOKE/RUN_TRAINING/RUN_TEST를 False로 바꾸세요. 현재 별도 프로세스에서 진행 중인 N01은 중복 실행하지 마세요. 아래 단계별 실행 안내는 수동 제어 시 사용할 수 있습니다.

# 정규화·상대신호 비교 실험

## 자동 순차 실행과 보고서

`run_campaign.py`는 별도 프로세스로 실행 중인 N01의 완료를 기다린 후 N03 → N02 → N04를 순차 학습합니다. 기존 `N01_runner_status.json`이 있어야 하며, N01은 `run_n01.py`로 시작한 실행을 사용합니다. 같은 CSV 해시·분할 해시·seed·학습 설정을 확인합니다. 네 실험의 가중치·임계값을 모두 고정한 뒤 Test를 평가하고 `summarize_campaign.py`로 보고서를 만듭니다. W&B와 패키지 자동 설치는 사용하지 않습니다.

```bash
.venv/bin/python -u phm_base/normalization_experiments/run_campaign.py > phm_base/results/normalization/campaign.log 2>&1
```

중복 실행하지 마세요. `.campaign.lock`은 실행 중 잠금이며 실패 시 해제됩니다. 재실행 시 완료된 학습·Test는 재사용하고 중단된 학습은 기존 호환성 검사를 거쳐 재개합니다. 실패 상태는 `campaign_status.json`, 세부 오류는 `campaign.log`에서 확인합니다. 각 실험의 실행 경로는 `campaign_runs.json`에 기록합니다. 경로·데이터 내용은 로컬 결과에만 저장합니다.

완료 후 결과:

- `phm_base/results/normalization/report/normalization_report.md`
- `phm_base/results/normalization/report/metrics_comparison.csv`
- `phm_base/results/normalization/report/training_summary.csv`
- `phm_base/results/normalization/report/metric_comparison.png`
- `phm_base/results/normalization/report/validation_curves.png`
- `phm_base/results/normalization/normalization_report.zip`

보고서 ZIP은 요약 표와 그래프를 포함합니다. 원본 센서 데이터·개별 윈도우 점수·가중치를 포함하지 않으며 전체 학습 결과는 각 run 폴더에 남습니다. Test 결과를 본 뒤 자동으로 모델을 변경하거나 추가 튜닝하지 않습니다.

브랜치: `experiment/normalization-relative-signals`

원본 CSV는 수정하지 않습니다. 현재 기본 실험을 기준으로 정규화 또는 상대신호 변환을 바꿉니다. 노트북은 독립 실행 가능하며, **점검·본학습·Test·ZIP 저장은 기본 ON, W&B는 OFF**입니다. 준비 셀은 CSV를 읽고 전처리하지만 모델을 학습하지 않습니다. 원본 노트북의 일반 안내보다 이 문서와 실제 SPEC 설정을 우선하세요.

## 비교 구성

| 파일 | 모델 | 파형 입력 | 보조 입력 | 비교 목적 |
|---|---|---|---|---|
| N01_normal_reference_zscore.ipynb | baseline LSTM | 정상 Train 기준 z-score | 없음 | 현재 방식 기준선 |
| N02_normal_reference_minmax.ipynb | 동일 | 정상 Train 기준 min-max, clipping 없음 | 없음 | N01과 스케일링만 비교 |
| N03_centered_reference_zscore.ipynb | 동일 | 구간 평균 제거 → 정상 Train 상대신호 기준 z-score | 없음 | 기준선 의존 감소 |
| N04_window_zscore.ipynb | 동일 | 각 구간의 평균·표준편차로 z-score | 없음 | 진폭을 제거한 파형 비교 |
| N05_sensor_reference_features.ipynb | sensor LSTM | N01 방식 | FFT·P2P·AC RMS | 보조 특징 모델의 대조군 |
| N06_sensor_centered_features.ipynb | 동일 sensor LSTM | N03 방식 | N05와 동일 | 보조 특징을 유지하며 기준선 제거 |

**N01↔N02, N01↔N03, N03↔N04, N05↔N06를 우선 비교하세요.** N03↔N06에는 모델 구조와 입력 특징 변화가 함께 있으므로 성능 차이를 정규화 효과 하나로 해석할 수 없습니다. N05/N06은 원본 04의 일반 RMS를 AC RMS로 바꾼 별도 대조 쌍입니다.

N03은 `r = x - window_mean`, `z = (r - normal_train_relative_mean) / normal_train_relative_std`입니다. 기준선은 제거하지만 정상 기준 분모가 고정되어 있어 진폭 차이는 남습니다. 스케일러는 겹치는 정상 Train 윈도우의 표본을 사용하므로 윈도우 중심의 관측이 반복 가중됩니다. N01/N02는 기존처럼 정상 Train 원본 행을 사용합니다. N03 비교에는 이 가중 방식의 차이도 포함됩니다.

N04는 `z = (x - window_mean) / max(window_std, 1e-8)`입니다. 추가 전역 scaler는 적용하지 않습니다. 완성된 구간 자체의 통계는 추론 시에도 계산할 수 있지만, 진폭 증가가 지워질 수 있습니다. 20행은 반드시 완전한 기계 주기라는 뜻이 아니므로 구간 통계에는 위상 영향도 있습니다.

AC RMS는 `sqrt(mean((x - window_mean)^2))`입니다. FFT·P2P·AC RMS는 파형 정규화 **전** 신호에서 계산하고 정상 Train 특징으로만 표준화합니다. N06에서는 크기 정보를 보존하되 직접적인 구간 평균은 입력에 추가하지 않습니다. 부하·운전 조건 등의 간접 정보는 여전히 남을 수 있습니다.

## 유지하는 조건

- 진동 2채널·전류 1채널, 20행 윈도우, seed 42, 기존 모델·optimizer·학습 설정.
- 기존 중복 제거, 0.15초보다 큰 공백에서 구간 분리, 구간 경계 기반 분할, 분할 안에서 윈도우 생성.
- 정상만 Train에 사용하고 이상은 Valid/Test에 사용. 날짜·행 번호·상태 라벨은 모델 입력에서 제외.
- 구간 평균·표준편차는 해당 윈도우 안에서만 계산. 파일 전체나 날짜별 통계로 변환하지 않음.
- 기존 Valid 모델 선택·임계값·Test 평가 규칙 유지. Test로 전처리 또는 모델을 고르지 않음.
- 새 저장 캠페인 `normalization_v1`, 실험별 SPEC·코드 해시·전처리 정보 저장. 기존 실행 재개는 사용하지 않음.

기본 설정은 최대 800 epoch입니다. 비교용으로 줄일 경우 **모든 실험에 동일한 TRAIN_CONFIG**를 적용하고 새 커널로 시작하세요. 단일 seed 결과만으로 우열을 확정하지 말고 후속 반복에서는 모든 실험에 동일한 seed 목록을 적용하세요.

## 사용자가 실행하는 순서

1. N01을 열고 첫 환경 설정에서 다음 값을 지정합니다. 같은 설정을 각 노트북에 적용합니다.

   ```python
   DATA_MODE = 'folder'
   DATA_ROOT = '/Users/pds2023/Desktop/KAMP/3. 소성가공 예지보전 AI 데이터셋'
   OUTPUT_ROOT = '/Users/pds2023/IdeaProjects/phm-fine-blanking-press/phm_base/results/normalization'
   DEVICE_REQUEST = 'cpu'  # CPU 실행. NVIDIA CUDA 환경에서는 'cuda' 또는 'auto'
   WANDB_ENABLED = False
   RUN_SMOKE = True
   RUN_TRAINING = True
   RUN_TEST = True
   RESUME_RUN = None
   ```

   필요한 환경은 numpy, pandas, scikit-learn, matplotlib, joblib, PyTorch입니다. 기본값은 패키지를 자동 설치하지 않습니다. Colab에서는 DATA_MODE='upload'로 CSV를 업로드하고 장치 설정을 맞추세요. 데이터 경로는 공개 노트북에 미리 기록하지 않았습니다.

2. 「데이터·정규화 준비」까지 실행하고 Train/Valid/Test 윈도우 수를 비교합니다. 같은 데이터라면 분할 수와 DATA['split_hash']는 모든 실험에서 같아야 합니다. 준비 셀의 scaler 표에서 N04의 raw scaler가 없는 것은 의도한 동작입니다.
3. 실행 내용을 이해한 뒤 RUN_SMOKE=True로 변경하고 2 epoch 점검 셀을 실행할 수 있습니다. smoke는 Test를 사용하지 않으며 본학습에 이어 쓰지 않습니다.
4. 본학습은 RUN_TRAINING=True로 변경한 뒤 학습 셀을 실행합니다. N01~N04를 먼저 비교하고 N05/N06을 별도로 비교하세요. 노트북마다 새 커널을 사용하세요.
5. Valid를 기준으로 선택을 고정한 후 RUN_TEST=True로 변경하고 Test 셀을 실행합니다. 학습 중 성능 비교는 Valid만 사용하세요. 준비만 할 때는 RUN_TEST=False를 유지해야 합니다.
6. 노트북의 기존 결과 표와 저장 결과를 확인합니다. 같은 판정 정책·모델 선택 기준의 F1, precision, recall, AP(PR 요약 지표), ROC-AUC 및 혼동행렬을 비교하고 정상 오탐률 `FP / (FP + TN)`도 확인하세요. AP와 사다리꼴 PR-AUC는 같은 계산이 아니므로 이름을 혼용하지 마세요.

N02의 정상 학습 범위를 벗어난 값은 0 미만·1 초과가 될 수 있습니다. 자르지 않는 것이 의도입니다. 방법마다 손실의 단위·가중이 바뀌므로 서로 다른 전처리의 복원 loss 숫자만으로 성능 순위를 정하지 마세요.

## 결과 해석의 범위

정상은 7월 12일, 이상은 7월 17일에만 존재합니다. 이 실험은 **특정 기준선·진폭 변화에 대한 모델 의존성을 비교**하는 실험이며 날짜 편향 제거 또는 새로운 날짜에서의 일반화 입증은 아닙니다. 이를 입증하려면 여러 날짜에 정상·이상이 함께 존재하는 추가 데이터와 날짜/운전 세션별 평가가 필요합니다.

N03의 성능이 떨어져도 기준선이 불필요한 편향이었다고 확정할 수 없고, 성능이 유지되어도 날짜 편향이 모두 사라졌다고 확정할 수 없습니다. N04가 놓치는 이상이 늘어나면 진폭 정보 제거가 영향을 주었을 가능성을 검토하세요.

## 코드 생성과 검증

`python3 phm_base/normalization_experiments/generate_notebooks.py`는 원본 01/04에서 여섯 노트북을 생성하고 각 코드 셀의 Python 문법을 검사합니다. **학습이나 전처리를 실행하지 않습니다.** 생성된 노트북을 수동 수정한 뒤 다시 생성하면 수정 사항이 덮어써지므로 변환 로직 변경은 생성기에 먼저 반영하세요.

작성 시에는 문법·실험 설정·기본 실행 ON·분할 함수 동일성·출력 제거를 정적으로 검증했습니다. 현재 기본 python3에 수치·PyTorch 라이브러리가 없어 실제 전처리와 학습 동작 검증은 아직 수행하지 않았습니다. 사용자의 준비 실행과 smoke에서 확인해야 합니다.
