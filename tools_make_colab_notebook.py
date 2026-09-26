"""Build the standalone Google Colab notebook from the maintained pipeline."""
import json
from pathlib import Path

source = Path("press_pipeline.py").read_text(encoding="utf-8")
cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)})

md("""# 소성가공 예지보전 LSTM-Autoencoder (Google Colab)

가이드북 2~6단계 노트북입니다. Colab 메뉴에서 **런타임 → 런타임 유형 변경 → 하드웨어 가속기 → GPU**를 고르고 셀을 위에서부터 실행하세요.

CSV 원본은 GitHub에 올리지 마세요. Google Drive의 `내 드라이브/kamp/fine-blanking-press/raw/`에 `press_data_normal.csv`, `outlier_data.csv` 두 파일을 두고, 결과는 Drive의 `artifacts/`에 저장합니다. TensorFlow 버전을 강제로 교체하지 않고 Colab 기본 버전을 사용해 실제 실행 버전을 기록합니다.
""")

code("""# 1. 런타임 / TensorFlow 확인. 기본 이미지에 없을 때만 TF 2.20 CPU/GPU wheel을 설치한다.
import os, sys, subprocess
print("Python:", sys.version)
try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.20.0"])
    import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "scikit-learn", "matplotlib", "joblib"])
""")

code("""# 2. Google Drive 연결
from google.colab import drive
from pathlib import Path
drive.mount("/content/drive")
DRIVE_PROJECT = Path("/content/drive/MyDrive/kamp/fine-blanking-press")
DATA_DIR = DRIVE_PROJECT / "raw"
ARTIFACT_DIR = DRIVE_PROJECT / "artifacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print("CSV 존재 여부:", {p.name: p.exists() for p in [DATA_DIR / "press_data_normal.csv", DATA_DIR / "outlier_data.csv"]})
""")

md("""위 CSV 존재 여부가 모두 `True`인지 확인하세요. `False`면 로컬의 두 CSV를 Google Drive의 `kamp/fine-blanking-press/raw` 폴더에 업로드한 다음 이 셀을 다시 실행하세요.
""")

code("""# 3. 실행 파이프라인 준비: 코드와 결과는 Drive에, 실행 중인 복사본은 Colab VM에 둔다.
os.environ["PROJECT_ROOT"] = str(DRIVE_PROJECT)
pipeline_source = """ + repr(source) + """
Path("/content/press_pipeline.py").write_text(pipeline_source, encoding="utf-8")
sys.path.insert(0, "/content")
import press_pipeline as pp
print("준비 완료:", pp.__file__)
""")

code("""# 4. 데이터 품질 검사 및 그래프 생성
RUN_NAME = "colab-inspect"
out = ARTIFACT_DIR / RUN_NAME
out.mkdir(parents=True, exist_ok=True)
frames, quality = pp.read_data(DATA_DIR)
pp.write_json(out / "quality.json", quality)
pp.plot_data(frames, out)
print({name: {k: report[k] for k in ["rows", "duplicate_rows", "non_0_1_second_intervals"]} for name, report in quality.items()})
""")

code("""# 5. 전처리와 윈도 구성 확인
SPLIT = "guide"  # 추가 실험은 "strict"로 변경하고 별도 실행 이름을 사용할 것
data, scaler = pp.prepare(frames, SPLIT)
print({name: values.shape for name, values in data.items()})
""")

code("""# 6. 짧은 시험 학습: 3 epoch로 학습-평가-저장-재로딩까지 연결 확인
smoke_args = type("Args", (), {"seed": 42, "epochs": 3, "split": SPLIT, "threshold_method": "guide", "data_dir": DATA_DIR})()
smoke_out = ARTIFACT_DIR / "colab-smoke"
smoke_out.mkdir(parents=True, exist_ok=True)
pp.train(data, scaler, quality, smoke_args, smoke_out)
""")

md("""## 본 학습

시험 학습이 성공한 뒤 실행하세요. 최대 800 epoch이며 조기 종료할 수 있습니다. Colab 세션이 끊길 수 있으므로 학습 결과는 Google Drive에 저장합니다.
""")

code("""# 7. 가이드 분할 본 학습
full_args = type("Args", (), {"seed": 42, "epochs": 800, "split": SPLIT, "threshold_method": "guide", "data_dir": DATA_DIR})()
full_out = ARTIFACT_DIR / "colab-guide-full"
full_out.mkdir(parents=True, exist_ok=True)
pp.train(data, scaler, quality, full_args, full_out)
""")

code("""# 8. 저장된 모델과 scaler 재로딩 검증
pp.verify(DATA_DIR, ARTIFACT_DIR / "colab-guide-full")
""")

md("""## 선택 실험: strict 분할

가이드 재현과 구분되는 실험입니다. 원본 구간을 먼저 나눠 경계 윈도 중첩을 줄이고 검증 F1 최대 지점으로 임곗값을 정합니다. 결과는 별도 폴더에 저장됩니다.
""")

code("""strict_data, strict_scaler = pp.prepare(frames, "strict")
print({name: values.shape for name, values in strict_data.items()})
strict_args = type("Args", (), {"seed": 42, "epochs": 800, "split": "strict", "threshold_method": "f1", "data_dir": DATA_DIR})()
strict_out = ARTIFACT_DIR / "colab-strict-full"
strict_out.mkdir(parents=True, exist_ok=True)
pp.train(strict_data, strict_scaler, quality, strict_args, strict_out)
""")

md("""평가 결과, 설정과 학습 이력은 Drive의 `artifacts/<실행이름>/`에서 확인하세요. 기준 guide shape는 train `(14880,20,3)`, valid `(1180,20,3)`, test `(4180,20,3)`입니다. 데이터 시간 간격은 보간하지 않습니다. 현재 파일처럼 정상/이상 라벨이 각 파일에서 일정하면 이 실험만으로 고장 10초 전 탐지 성능을 입증할 수 없습니다. Colab의 Python/TensorFlow/GPU 가용성은 런타임마다 달라지며 실제 버전은 결과 설정에 기록됩니다.""")

nb = {"cells": cells,
      "metadata": {"colab": {"name": "소성가공_예지보전_Colab.ipynb", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/소성가공_예지보전_Colab.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
