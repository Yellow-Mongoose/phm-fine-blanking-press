# KAMP 소성가공 예지보전 가이드 Baseline 재현 요약

## 1. 무엇을 재현했는가

KAMP 「소성가공 예지보전 AI 데이터셋」 분석실습 가이드북의 LSTM-Autoencoder baseline을 로컬 Windows CPU 환경에서 `.py` 기반으로 재구성했다.

가이드의 핵심 흐름은 다음과 같다.

- 입력 센서: `AI0_Vibration`(상부 진동), `AI1_Vibration`(하부 진동), `AI2_Current`(전류)
- 정상 데이터만 사용하여 LSTM-Autoencoder 학습
- `sequence=20` 시계열 window 구성
- 정상 데이터 앞 15,000행을 학습 구간으로 사용
- MinMaxScaler는 학습 정상 데이터에만 fit
- reconstruction MSE를 anomaly score로 사용
- validation set에서 Precision과 Recall이 정확히 같은 첫 지점을 threshold로 선택
- threshold 초과 시 이상으로 판정
- Confusion Matrix, Accuracy, F1-score로 평가

## 2. 재현 환경

- Python 3.9.7
- TensorFlow 2.7.0
- CPU 실행
- NumPy 1.19.5
- pandas 1.3.5
- scikit-learn 1.0.2
- matplotlib 3.5.1
- seaborn 0.11.2

가이드의 NumPy 1.20.0과는 차이가 있으나, 현재 Windows + TensorFlow 2.7.0 환경 호환성을 위해 1.19.5로 고정했다.

## 3. 재현 결과

### 확인 완료

- CSV load 성공
  - 정상: `(20000, 5)`
  - 이상: `(600, 5)`
- sequence / split 구성 성공
  - Train: `(14880, 20, 3)`
  - Validation: `(1180, 20, 3)`
  - Test: `(4180, 20, 3)`
- LSTM-Autoencoder 생성 성공
- 1 epoch 기준 `fit → threshold → evaluate` 전체 경로 실행 성공

### 1 epoch smoke test

- Threshold: `0.05300698`
- Confusion Matrix: `[[3807, 193], [81, 99]]`
- Accuracy: `93.44%`
- F1-score: `41.95%`
- Fit time: 약 `12.24초`

### Full training

가이드 설정 그대로 최대 800 epoch 학습을 두 차례 수행했으나, 최종 지표는 신뢰성 있게 저장하지 못했다.

- 1차 실행: 학습은 800 epoch까지 진행됐지만 실행 wrapper가 마지막 stdout을 보존하지 못함
- 2차 실행: 결과 JSON 저장 wrapper를 사용했지만 최종 JSON이 생성되지 않음
- 따라서 full-run threshold / confusion matrix / Accuracy / F1은 추정하지 않고 미확보로 기록

가이드 공개 기준값은 다음과 같다.

- Threshold ≈ `0.0035`
- Confusion Matrix = `[[3922, 78], [26, 154]]`
- Accuracy ≈ `97.51%`
- F1-score ≈ `74.76%`

## 4. 재현 과정에서 확인한 핵심 한계

### A. Threshold 규칙이 취약함

가이드는 validation set에서 `Precision == Recall`인 첫 지점을 threshold로 선택한다.

난수 초기화에 따라 Precision과 Recall이 정확히 같은 지점이 존재하지 않을 수 있으므로, 학습 완료 후 threshold 계산 단계에서 실패할 가능성이 있다. 이번 재현에서는 가이드 규칙을 임의로 수정하지 않았다.

### B. `100-step offset`이 곧 10초 선행예측을 의미하지는 않음

가이드는 20개 시점(2초)을 사용하여 100시점(10초) 이후 이상을 탐지한다고 설명한다.

하지만 실제 LSTM-Autoencoder는 입력한 20-step window 자체를 재구성하며, 10초 뒤 센서값을 예측하는 구조가 아니다. 또한 제공된 정상 파일의 `Equipment_state`는 0, 이상 파일은 1로 고정되어 있으므로 label을 100 step 뒤로 이동시키는 것만으로 실제 10초 lead-time을 입증하기 어렵다.

따라서 이 baseline은 현재 상태의 정상/이상 패턴을 구분하는 anomaly detection baseline으로 이해하는 것이 안전하다.

### C. 대회 요구사항과 baseline 사이의 차이

대회는 단순 이상탐지뿐 아니라 다음을 요구한다.

- 이상 조기탐지
- 정상 운전변화에 대한 오경보 감소
- False Negative / False Positive 발생조건 분석

따라서 가이드 baseline은 출발점으로 사용하되, 이후 Data Audit과 validation 설계에서 실제 조기탐지 가능 여부와 오류조건을 별도로 검증해야 한다.

## 5. 팀 차원의 결론

이번 재현의 목적은 가이드 숫자를 정확히 복제하는 것이 아니라 **공식 baseline의 처리 흐름과 한계를 확인하는 것**으로 본다.

현재까지 확보한 것:

1. 실행 가능한 baseline 코드와 환경
2. 데이터 split / scaling / sequence / LSTM-AE / threshold / 평가 흐름 확인
3. end-to-end smoke test 성공
4. full-run 재현성 문제와 threshold 규칙의 취약성 확인
5. `10초 offset ≠ 10초 선행예측 입증`이라는 핵심 검토 포인트 확인

다음 단계는 baseline 튜닝이 아니라 **실제 대회 데이터의 구조·시간 연속성·정상/이상 분포·label 의미·조기탐지 가능 여부를 확인하는 Data Audit**이다.
