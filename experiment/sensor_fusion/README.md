# Dual-Encoder Sensor-Fusion LSTM-AE 실험

이 폴더는 기존 single-encoder LSTM-Autoencoder와 dual-encoder sensor-fusion 변형을 통제된 조건에서 비교하기 위한 실험 파일을 담고 있습니다.

실험 질문은 하나로 제한합니다. 기존과 동일한 데이터 전처리, 학습, threshold 선정, test 평가 조건에서 진동과 전류를 modality별 encoder로 분리한 뒤 latent vector를 concatenate하면 결과에 차이가 생기는가?

## 폴더 구성

- `EXPERIMENT_CONTRACT.md` — 고정된 실험 규칙, 경로, 비교 제약사항
- `experiment_config.py` — 기준 notebook에서 옮긴 고정 설정 및 실행 위치에 독립적인 프로젝트/data 경로 탐색기
- `dual_encoder_sensor_fusion_reproduction.ipynb` — 독립 실험 notebook. `../pytorch_reproduction.ipynb`는 수정하지 않습니다.
- `smoke_test_sensor_fusion.py` — 추후 추가할 간단한 실행 점검 스크립트

원본 데이터는 이 폴더로 복사하지 않습니다. Notebook은 자신의 위치를 기준으로 상위 프로젝트의 `data/`를 찾아 사용합니다. Clone 후에는 아래와 같은 구조를 기대합니다.

```text
ManufactorAIContest/
├── data/
│   ├── press_data_normal.csv
│   └── outlier_data.csv
└── experiment/
    └── sensor_fusion/
        └── dual_encoder_sensor_fusion_reproduction.ipynb
```

학습 checkpoint, plot, prediction 등 실행 산출물은 `experiment/sensor_fusion/results/` 아래에 생성됩니다. 대용량 산출물은 Git으로 관리하지 않으며, 필요한 경우 작은 결과 요약만 별도로 추가합니다.

## 현재 상태

- 1단계 완료: 기준 실험 규칙과 경로 원칙을 `EXPERIMENT_CONTRACT.md`에 고정했습니다.
- 2단계 완료: 기준과 동일한 데이터 로딩, scaling, window, split, manifest 생성 절차를 새 notebook에 구현했습니다.
- 3단계 완료: 진동/전류 dual-encoder와 단순 latent concat decoder를 구현했습니다. 학습 및 평가는 아직 수행하지 않았습니다.
- 4단계 코드 준비: 기준과 동일한 학습·정상 validation·loss 시각화 코드가 notebook에 준비되어 있으며, 아직 실행하지 않았습니다.
- 5단계 코드 준비: validation-only threshold 선정과 test 평가·시각화 코드가 notebook에 준비되어 있으며, 아직 실행하지 않았습니다.
- 6단계 코드 준비: 동일 조건으로 실행한 single-encoder `metrics.json`을 지정하면 dual-encoder와의 비교표·confusion matrix를 생성합니다. 아직 실행하지 않았습니다.
