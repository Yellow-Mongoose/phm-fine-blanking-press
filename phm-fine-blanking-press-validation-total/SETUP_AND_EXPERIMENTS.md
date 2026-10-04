# 설정 및 실험 변경점 요약

## 1. 목적과 증거의 범위

이 패키지는 동일한 KAMP 프레스 시계열을 대상으로 한 세 가지 후속 가설 검증을 기록한다.
모든 수치는 창(window) 단위 결과이며, 단일 seed·탐색적 프로토콜의 기술 통계다. 따라서
통계적 유의성, 다른 설비·기간에 대한 일반화, 특정 물리 고장 원인의 확정은 주장하지 않는다.

| 가설 | 질문 | 최종 채택 결과 |
|---|---|---|
| H1 | 절댓값 대신 signed 입력을 유지하면 LSTM-AE가 달라지는가? | `results/h1_signed_input` |
| H2 | signed AE에 진동 P2P를 더하면 보완되는가? | `results/h2_p2p_fusion` |
| H3 | 전류 PSD·주기성이 독립 보완 신호가 되는가? | `results/h3_psd_periodicity` |

## 2. 실행 환경

정확한 패키지 pin은 `environment.yml`에 있다.

| 항목 | 설정 |
|---|---|
| Python | 3.9.7 |
| TensorFlow baseline 호환 | TensorFlow 2.7.0, NumPy 1.19.5, pandas 1.3.5, scikit-learn 1.0.2 |
| PyTorch 실험 런타임 | torch 2.7.1+cu118, CUDA 11.8 wheel |
| 시각화 | matplotlib 3.5.1, seaborn 0.11.2 |
| H1 학습 장치 | CUDA 사용 기록 (`gpu_ablation_20261004`) |
| H2/H3 | 저장된 signed checkpoint에 대한 inference·평가; 재학습하지 않음 |

CUDA/드라이버가 다른 환경에서는 해당 환경에 맞는 PyTorch wheel을 설치해야 한다. CPU 실행은
가능할 수 있으나 본 결과의 실행 시간이나 부동소수점 세부 결과를 동일하게 보장하지 않는다.

## 3. 공통 데이터·평가 프로토콜

| 항목 | 고정 설정 |
|---|---|
| 시퀀스 길이 | W20 |
| label offset | 100 |
| 정상 Train | 15,000 행 |
| Validation 구성 | 정상 880 / 이상 300 창 |
| seed | H1 학습 seed 42 |
| AE 입력/점수 | 3채널, 마지막 시점의 reconstruction MSE |
| H1 학습 | 최대 800 epoch, batch 128, learning rate 0.001 |
| 시계열 주의점 | H1/H2는 원 프로토콜의 time-gap-agnostic window 및 validation/test 입력 경계 overlap을 보존 |

이 구조는 leakage-resistant 재설계가 아니라 **기존 프로토콜에서의 입력/특징 ablation**이다.
후속 결론은 이 경계를 벗어나지 않는다.

## 4. H1 — absolute 입력과 signed 입력

### 바뀐 것

두 변형은 동일한 창 인덱스, 라벨, 모델 구조(63,171 trainable parameters), 초기 가중치,
shuffle 순서, seed, 학습 제어를 사용한다. 유일한 의도적 차이는 모델별 normal-train
MinMaxScaler 이전에 `abs()`를 적용하는지 여부다.

| 항목 | absolute | signed |
|---|---|---|
| 입력 | 센서값 절댓값 | 원 신호의 부호 유지 |
| scaler | 각 변형의 정상 Train에 별도 적합 | 각 변형의 정상 Train에 별도 적합 |
| 임계값 | Validation의 원 `precision == recall` 규칙 | 동일 |
| Test | 공통 4,180 창 | 공통 4,180 창 |

### 결과와 해석

signed의 F1은 0.7595, absolute의 F1은 0.6933이었다. signed는 FP를 114에서 65로 줄였고,
FN은 24에서 30으로 늘었다. 그러므로 결론은 “오탐을 크게 줄여 F1을 개선했다”이며, recall이
모든 상황에서 개선됐다는 주장이 아니다. 서로 다른 scaler를 사용하므로 MSE 절대값은 직접
비교하지 않는다.

## 5. H2 — raw signed vibration P2P와 fusion

### H1에서 바뀐 것

H1의 **signed checkpoint와 scaler를 고정**하고 재학습하지 않는다. raw signed
`AI0_Vibration`, `AI1_Vibration`의 W20 P2P를 추가한다.

| 항목 | 설정 |
|---|---|
| P2P | `max(window) - min(window)`; abs 또는 MinMax scaling 전 raw signed 신호에서 계산 |
| 기준 | 센서별 normal-train P2P Q95, IQR |
| 센서 결합 | 두 센서 점수의 maximum |
| fusion | `normalized_AE + α × normalized_P2P` |
| α 후보 | 0, 0.5, 1.0, 2.0 |
| α 선택 | Validation F1 최대; 동률이면 작은 α |
| 임계값 | 각 방법의 정상 Validation empirical FPR ≤ 1%, strict `score > threshold` |
| Test | 4,180 창; 선택·적합에는 사용하지 않음 |

### 결과와 해석

선택된 α=2.0의 fusion C와 P2P-only B는 모두 F1 0.8197, FP 0, FN 55였다. AE-only A 대비
F1은 +0.1484, FN은 −27이다. 다만 normal-train의 one-sided P2P 점수가 대부분 0이어서 IQR도
0이었고, 사전 정의한 `1e-6` IQR floor가 비영 P2P 초과를 크게 만들었다. 따라서 이는 일반적인
두 점수의 상보 fusion 증거가 아니라 **P2P가 ranking을 지배한 결과**로 해석한다.

## 6. H3 — 전류 PSD·autocorrelation

### H2에서 바뀐 것

H1의 signed checkpoint·scaler를 고정하고, 전류 신호에서 spectral entropy 기반 PSD 점수와
autocorrelation 점수(선택 lag 1)를 산출한다. W20 내부 19개 시간 간격이 모두 150 ms 이하인
연속성 적격 창만 A/B/C/D의 공통 대상으로 사용한다.

| 항목 | 설정 |
|---|---|
| gap 기준 | 0.15 초 |
| PSD 기준 | normal-train Q95/IQR |
| ACF 기준 | normal-train Q05/IQR |
| fusion 정규화 | `(score - median) / IQR` |
| 임계값 | 정상 Validation의 empirical FPR ≤ 1% |
| Test 모집단 | 2,094 창 — H1/H2의 4,180 창과 다름 |

### 결과와 해석

AE는 F1 0.9121, PSD 단독은 recall 1.0·F1 0.6953을 기록했다. PSD는 FN 0이지만 FP 78이었다.
PSD/ACF의 normal-train one-sided score IQR이 0 또는 너무 작아 C(AE+PSD), D(AE+ACF)는
계산하지 않았다. 사후 epsilon이나 다른 정규화로 결과를 만들지 않았으므로, **fusion 가설은
실패가 아니라 미판정**이다.

## 7. 숫자를 비교하는 규칙

- H1 내부에서는 absolute와 signed만 직접 비교한다.
- H2 내부에서는 A/B/C만 직접 비교한다. H1과 H2는 임계값 정책이 다르므로 F1을 순위화하지 않는다.
- H3는 연속성 적격 모집단(2,094 창)이므로 H1/H2와 F1을 직접 비교하지 않는다.
- Test 성능을 확인한 뒤 바꾼 설정은 같은 Test의 독립 최종 성능으로 보고하지 않는다.

## 8. 재실행 순서

원본 CSV를 `data/`에 두고 새 결과 폴더를 사용한다. 제공된 `results/` 스냅샷에는 모델 가중치를
의도적으로 넣지 않았으므로, H2/H3를 실행하기 전에 새 H1 결과가 필요하다.

```powershell
conda env create -f environment.yml
conda activate kamp_repo

# 1) H1: 새 signed/absolute 모델 및 결과 생성
python src/run_ablation.py --run --data-dir .\data --output-dir .\work\h1

# 2) H2: H1 results 아래의 signed model/scaler 사용
python src/run_p2p_fusion.py --run --data-dir .\data --first-results .\work\h1 --output-dir .\work\h2

# 3) H3: H1 signed checkpoint·scaler를 명시적으로 전달
python src/run_psd_periodicity.py --run --data-dir .\data --signed-checkpoint .\work\h1\signed\checkpoint_best.pt --signed-scaler .\work\h1\signed\scaler.joblib --output-dir .\work\h3
```

각 실행 전에는 `--run` 없이 실행해 preflight 결과와 입력 경로를 먼저 확인한다. H1에서 결과
폴더의 실제 구조가 다르면 H2의 `--first-results` 및 H3의 checkpoint/scaler 경로를 그 구조에 맞춘다.

## 9. 공개용 산출물 정책

`results/`에는 보고서에 필요한 PNG, 설정, preflight, 작은 비교표만 보존한다. 다음은 제외한다.

- 원본 데이터 CSV와 공급된 압축 파일
- `.pt`, `.joblib`, `.npz` 모델·scaler·재구성 산출물
- 대형 `features/`, `scores/`, split manifest CSV 및 run log

필요하면 데이터 라이선스·배포 권한을 확인한 뒤 GitHub Release 또는 Git LFS로 별도 제공한다.
