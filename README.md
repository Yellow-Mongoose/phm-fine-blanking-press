# PHM Fine-Blanking Press — KAMP Baseline Reproduction

KAMP 「소성가공 예지보전 AI 데이터셋」 분석실습 가이드북의 LSTM-Autoencoder baseline을 `.py` 형태로 재구성한 저장소입니다.

> 목적: 성능 개선이 아니라 **공식 가이드 baseline의 재현 및 구조 검토**

## Repository structure

```text
.
├── environment.yml
├── kamp_config.py
├── kamp_pipeline.py
├── run_baseline.py
├── smoke_test.py
├── record_reproduction_run.py
├── REPRODUCTION_NOTES.md
├── REPRODUCTION_RESULT.md
├── TEAM_SHARE_BASELINE_KR.md
└── data/
    ├── press_data_normal.csv
    └── outlier_data.csv
```

원본 CSV는 저장소에 직접 포함하지 않는 것을 권장합니다. 각 팀원은 KAMP에서 동일 데이터셋을 내려받아 `data/` 아래에 배치하세요.

## Environment

Tested environment:

- Python 3.9.7
- TensorFlow 2.7.0
- NumPy 1.19.5
- pandas 1.3.5
- scikit-learn 1.0.2
- matplotlib 3.5.1
- seaborn 0.11.2
- CPU execution

환경 생성:

```bash
conda env create -f environment.yml
```

Windows PowerShell에서 `conda activate kamp_repo`가 동작하지 않는 경우 다음 방식으로 실행할 수 있습니다.

```powershell
E:\miniconda\Scripts\conda.exe run -n kamp_repo python smoke_test.py
```

## Data placement

다음 두 파일을 `data/` 디렉터리에 배치합니다.

```text
data/press_data_normal.csv
data/outlier_data.csv
```

## 1. Smoke test

학습 없이 데이터 load, sequence/split, 모델 생성을 확인합니다.

```bash
conda run -n kamp_repo python smoke_test.py
```

예상 shape:

```text
normal: (20000, 5)
outlier: (600, 5)

X_train: (14880, 20, 3)
X_valid: (1180, 20, 3)
X_test: (4180, 20, 3)
```

## 2. Baseline training

가이드 설정 그대로 최대 800 epoch 학습을 실행합니다.

```bash
conda run -n kamp_repo python run_baseline.py --train
```

주의:

- random seed는 가이드에 없으므로 설정하지 않았습니다.
- 가이드의 threshold 규칙인 `첫 precision == recall 지점`을 그대로 사용합니다.
- 실행마다 결과가 달라질 수 있습니다.
- 어떤 run에서는 정확히 같은 Precision/Recall 지점이 없어 threshold 계산이 실패할 수 있습니다.

## 3. Result-recording run

학습 history와 최종 지표를 JSON으로 저장하려는 one-off 실행:

```bash
conda run -n kamp_repo python record_reproduction_run.py
```

현재 구현은 threshold/evaluation까지 성공한 뒤 마지막에 JSON을 기록합니다. 따라서 threshold 계산 단계에서 예외가 발생하면 JSON이 생성되지 않을 수 있습니다. 이 동작은 가이드 baseline의 취약성을 보존하기 위해 임의 수정하지 않았습니다.

## Reproduction status

확인 완료:

- CSV load
- preprocessing
- chronological split
- MinMax scaling
- sequence construction
- LSTM-Autoencoder construction
- 1-epoch end-to-end `fit → threshold → evaluate`

1-epoch smoke result:

```text
Threshold: 0.05300698
Confusion Matrix: [[3807, 193], [81, 99]]
Accuracy: 93.44%
F1-score: 41.95%
Fit time: 12.24 s
```

Full 800-epoch training은 두 차례 수행했지만 최종 metric persistence에 실패하여 full-run 수치는 주장하지 않습니다. 자세한 내용은 `REPRODUCTION_RESULT.md`를 참고하세요.

## Important interpretation note

가이드의 `sequence=20`, `offset=100` 구성은 문서상 “2초 입력으로 10초 이후 이상 탐지”로 설명됩니다.

그러나 현재 LSTM-Autoencoder는 입력 window 자체를 재구성하며, 10초 뒤 센서값을 직접 예측하지 않습니다. 또한 정상/이상 파일의 `Equipment_state`가 각각 고정 label로 구성되어 있어, 100-step label offset만으로 실제 10초 lead-time prediction을 입증하기 어렵습니다.

따라서 본 코드는 **가이드 baseline reproduction**으로 사용하고, 대회용 validation 및 조기탐지 정의는 별도로 설계해야 합니다.

## KAMP source

연구/공식 활용 시 KAMP 출처를 표기해야 합니다.

국문 표기 예시:

> 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 소성가공 예지보전 AI 데이터셋, 스마트제조혁신추진단(㈜인터엑스), 2022.12.23., www.kamp-ai.kr
