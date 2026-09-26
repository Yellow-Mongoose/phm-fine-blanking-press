# phm-fine-blanking-press

KAMP 소성가공 예지보전 데이터셋을 이용한 LSTM-Autoencoder 실습 준비 프로젝트.

실행 순서와 검증 기준: [실습 계획](docs/guide-run-plan.md).

원본 데이터, 가상환경, 학습 산출물은 `.gitignore`로 제외한다. 코드와 설정,
실행 환경 명세 및 요약 평가 결과를 Git으로 관리한다.

## 환경 설정

설치 오류의 원인은 pip 버전이 아니라 Python 버전이다. `python313`은 Python
3.13이며 TensorFlow 2.7.0의 Windows 배포 파일과 호환되지 않는다.
현재 Python을 사용하는 실행 환경은 다음과 같이 구성한다(PowerShell).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import sys, tensorflow as tf; print(sys.executable); print(tf.__version__)"
```

가상환경 활성화 없이 그 환경의 실행 파일을 직접 지정하므로 다른 pip가 선택되는 일을 방지한다.
이 환경은 **Python 3.13 + TensorFlow 2.20.0**이며 가이드의 알고리즘을 현대 환경에서 실행한다.
가이드와 같은 버전이 필요하면 Python 3.9를 별도 설치한 뒤 그 실행 파일로
`-m venv .venv`를 실행하고 `requirements-legacy.txt`를 설치한다.
기존 `.venv`와 혼합하지 말고 별도 환경 이름을 사용한다. legacy 명세는 호환 환경 후보이며 설치 검증은 아직 하지 않았다.

공식 배포 파일: [TensorFlow 2.7.0](https://pypi.org/project/tensorflow/2.7.0/),
[TensorFlow 2.20.0](https://pypi.org/project/tensorflow/2.20.0/).

## 2~6단계 실행

한글 주석이 포함된 [press_pipeline.py](press_pipeline.py)가 전체 흐름을 구현한다.
기본 데이터 경로는 사용자 홈의 `datasets/kamp/fine-blanking-press/raw`이다.
다른 위치는 `--data-dir '경로'` 또는 `DATA_DIR` 환경변수로 지정한다.
원본 CSV는 수정하지 않는다.

GPU 학습용으로 독립 실행 가능한 [Google Colab 노트북](notebooks/소성가공_예지보전_Colab.ipynb)도 제공한다.
노트북은 파이프라인 코드를 포함하므로 노트북 하나만 Colab에 올리면 된다.
CSV 원본은 Google Drive의 `kamp/fine-blanking-press/raw/` 폴더에 따로 넣는다.
Colab 버전은 `press_pipeline.py`를 수정한 뒤 `python tools_make_colab_notebook.py`로 다시 생성한다.

```powershell
# 2. 데이터 품질 검사와 원본/절댓값 시계열 그림 생성
.\.venv\Scripts\python.exe press_pipeline.py inspect --run inspect

# 3. 정규화, 시계열 분할 및 shape 확인(학습하지 않음)
.\.venv\Scripts\python.exe press_pipeline.py prepare --run prepare

# 4. 3 epochs로 전체 학습·평가·저장·재로딩 흐름 확인
.\.venv\Scripts\python.exe press_pipeline.py train --epochs 3 --run smoke

# 5~6. 가이드 분할로 본 학습, 임곗값 선정, 최종 평가 및 자동 재로딩 검증
.\.venv\Scripts\python.exe press_pipeline.py train --epochs 800 --run guide-full

# 저장된 모델로 원본부터 다시 처리하여 점수·판정 일치 여부만 재검증
.\.venv\Scripts\python.exe press_pipeline.py verify --run guide-full

# 추가 실험: 원본 구간을 먼저 분리한 엄격한 검증 + 검증 F1 최대 임곗값
.\.venv\Scripts\python.exe press_pipeline.py train --split strict --threshold-method f1 --epochs 800 --run strict-full

# 합성 데이터로 윈도 인덱스·정규화 누출·임곗값 경계를 검사
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`inspect`와 `prepare`는 TensorFlow 없이도 나머지 패키지가 설치되어 있으면 실행 가능하다.
각 실행 결과는 Git 제외 폴더 `artifacts/<run>/`에 생성한다.
학습이 저장된 이름은 재사용하지 않는다. 반복 실험에서는 `--run smoke-2`처럼 새 이름을 지정한다.

| 파일 | 내용 |
| --- | --- |
| quality.json, *_signals.png | 행 수·중복·시간 간격·통계·데이터 해시와 신호 그림 |
| shapes.json | 분할 방식과 배열 크기 |
| model.h5, scaler.joblib | 모델 가중치·구조와 학습 데이터로 적합한 정규화 객체 |
| config.json | 임곗값, 센서 순서, 전처리, seed, 버전, 학습 횟수와 시간 |
| history.json, evaluation.png | 학습 곡선, 평가 점수와 혼동행렬 |
| metrics.json | 정확도·정밀도·재현율·F1·정상 오탐률 |
| test_scores.npy, reload_check.json | 재로딩 비교용 점수와 검증 결과 |

guide 분할은 `(14880,20,3)` 학습, `(1180,20,3)` 검증, `(4180,20,3)` 평가다.
strict 분할은 정상 원본 15000/1000/4000행, 이상 원본 300/300행으로 먼저 나눠
검증 1060개, 평가 4060개가 된다. 서로 다른 분할 결과는 구분해서 비교한다.
guide 임곗값은 정밀도=재현율 지점을 우선하고 없으면 검증 F1 최대 지점을 선택한다.
PR 곡선과 일관되게 `점수 >= 임곗값`을 사용하므로 책의 `>`와 경계 판정이 다르다.
짧은 실행은 동작 확인용이며 성능 기준으로 사용하지 않는다.
현재 데이터만으로 고장 10초 전 예측 성능이 입증되지는 않는다.
