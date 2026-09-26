"""KAMP 실습 2~6단계: inspect → prepare → train → verify.

원본 CSV는 읽기만 하고 모든 산출물은 Git에서 제외된 artifacts 아래에 쓴다.
TensorFlow는 학습/재로딩 시에만 불러오므로 설치 전에도 데이터 검사가 가능하다.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # GUI 없이 실행하고 그림은 PNG로 저장한다.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# Colab에서는 PROJECT_ROOT를 Google Drive 경로로 지정해 산출물을 런타임 종료 후에도 보존한다.
ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent)))
FEATURES = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SEQUENCE, HORIZON = 20, 100


def write_json(path, value):
    """numpy 객체는 호출부에서 기본 Python 자료형으로 변환해 전달한다."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_data(data_dir):
    """2단계: 스키마/라벨/시간/센서값 검사. 중복은 보고만 하고 원본을 보존한다."""
    frames, report = {}, {}
    for name, filename, count, label, date in [
        ("normal", "press_data_normal.csv", 20000, 0, "2022-07-12"),
        ("anomaly", "outlier_data.csv", 600, 1, "2022-07-17"),
    ]:
        path = data_dir / filename
        df = pd.read_csv(path, index_col=0)
        expected = ["TimeStamp"] + FEATURES + ["Equipment_state"]
        if list(df.columns) != expected or len(df) != count:
            raise ValueError(f"{filename}: 가이드의 열 구성 또는 행 수와 다릅니다.")
        if df.isna().any().any():
            raise ValueError(f"{filename}: 결측치를 확인하세요.")
        duplicates = int(df.duplicated().sum())
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="raise")
        for col in FEATURES + ["Equipment_state"]:
            df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[FEATURES].to_numpy()).all():
            raise ValueError(f"{filename}: 센서에 NaN 또는 무한대가 있습니다.")
        if not df["Equipment_state"].eq(label).all():
            raise ValueError(f"{filename}: 정상/이상 라벨이 예상과 다릅니다.")
        if not df.TimeStamp.is_monotonic_increasing:
            raise ValueError(f"{filename}: 시간 역전이 있습니다. 원본 순서를 확인하세요.")
        if not df.TimeStamp.dt.strftime("%Y-%m-%d").eq(date).all():
            raise ValueError(f"{filename}: 수집 날짜가 가이드와 다릅니다.")
        delta = df.TimeStamp.diff().dt.total_seconds().dropna()
        # 시간 간격의 이상은 자동 보간하지 않는다. 가이드 재현에 미치는 영향을 보고한다.
        report[name] = {
            "rows": len(df), "duplicate_rows": duplicates,
            "missing_cells": int(df.isna().sum().sum()),
            "non_0_1_second_intervals": int((~np.isclose(delta, 0.1)).sum()),
            "interval_counts": {str(k): int(v) for k, v in delta.value_counts().items()},
            "sensor_statistics": df[FEATURES].describe().to_dict(),
            "absolute_correlation": df[FEATURES].abs().corr().to_dict(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        frames[name] = df
    return frames, report


def plot_data(frames, out):
    """원본과 절댓값 변환 후의 세 센서를 나란히 비교한다."""
    for name, df in frames.items():
        fig, axes = plt.subplots(3, 2, figsize=(13, 8))
        for i, col in enumerate(FEATURES):
            axes[i, 0].plot(df.TimeStamp, df[col], linewidth=0.5)
            axes[i, 1].plot(df.TimeStamp, df[col].abs(), linewidth=0.5)
            axes[i, 0].set_title(col + " / raw")
            axes[i, 1].set_title(col + " / absolute")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / f"{name}_signals.png")
        plt.close(fig)


def windows(values, labels):
    """3단계: 가이드의 N-20-100개 윈도와 미래 라벨 인덱스를 정확히 따른다.

    입력은 [i:i+20], 라벨은 i+120이다. 라벨은 평가에만 쓰며 학습 목표는 X다.
    일정 라벨 파일이므로 이 연결 자체가 10초 사전 예측을 입증하지는 않는다.
    """
    count = len(values) - SEQUENCE - HORIZON
    if count <= 0:
        raise ValueError("윈도를 생성하려면 구간당 120행보다 많은 데이터가 필요합니다.")
    x = np.stack([values[i:i + SEQUENCE] for i in range(count)]).astype("float32")
    y = np.asarray(labels[SEQUENCE + HORIZON:], dtype="int64")
    return x, y


def prepare(frames, mode):
    """guide는 책의 분할, strict는 원본 구간을 먼저 분할해 경계 중첩을 방지한다."""
    normal, anomaly = frames["normal"], frames["anomaly"]
    scaler = MinMaxScaler()
    scaler.fit(normal[FEATURES].iloc[:15000].abs())

    def make(df):
        # 평가 센서값이 학습 범위를 넘으면 1보다 커질 수 있다. clip하지 않는다.
        return windows(scaler.transform(df[FEATURES].abs()), df.Equipment_state.to_numpy())

    train, _ = make(normal.iloc[:15000])
    if mode == "guide":
        nx, ny = make(normal.iloc[15000:])
        ax, ay = make(anomaly)
        vn, va = (nx[:880], ny[:880]), (ax[:300], ay[:300])
        tn, ta = (nx[880:], ny[880:]), (ax[300:], ay[300:])
    else:
        # 각 구간 내에서만 입력과 미래 라벨을 구성하므로 서로 원본 행을 공유하지 않는다.
        vn, tn = make(normal.iloc[15000:16000]), make(normal.iloc[16000:])
        va, ta = make(anomaly.iloc[:300]), make(anomaly.iloc[300:])
    data = {
        "train": train, "valid_normal": vn[0],
        "valid": np.concatenate([vn[0], va[0]]), "valid_y": np.concatenate([vn[1], va[1]]),
        "test": np.concatenate([tn[0], ta[0]]), "test_y": np.concatenate([tn[1], ta[1]]),
    }
    return data, scaler


def tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow를 현재 인터프리터에 설치하세요. README의 환경 설정을 참고하세요.") from exc
    return tf


def build_model(tf):
    """4단계: 20×3 입력을 압축한 뒤 같은 크기로 복원하는 LSTM-Autoencoder."""
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(SEQUENCE, len(FEATURES))),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.RepeatVector(SEQUENCE),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(len(FEATURES))),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def score(model, x):
    """가이드처럼 마지막 시점의 세 센서만 평균해 윈도별 재구성 오차를 구한다."""
    predicted = model.predict(x, batch_size=128, verbose=0)
    result = np.mean(np.square(x[:, -1, :] - predicted[:, -1, :]), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("재구성 오차가 유한하지 않습니다. 학습 상태를 확인하세요.")
    return result


def select_threshold(y, scores, method):
    """5단계: 검증셋만 사용한다. PR 곡선의 마지막 원소에는 대응 임곗값이 없다."""
    precision, recall, thresholds = metrics.precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)
    equal = np.flatnonzero(p == r)
    index = int(equal[0]) if method == "guide" and len(equal) else int(np.argmax(f1))
    # sklearn PR의 >= 판정과 평가 판정을 통일한다(책의 >와 경계 처리 차이).
    return float(thresholds[index]), {
        "requested_method": method,
        "used_method": "precision_equals_recall" if method == "guide" and len(equal) else "max_f1",
        "comparison": ">=", "precision": float(p[index]), "recall": float(r[index]),
    }


def evaluate(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "accuracy": float(metrics.accuracy_score(y, pred)),
        "precision": float(metrics.precision_score(y, pred, zero_division=0)),
        "recall": float(metrics.recall_score(y, pred, zero_division=0)),
        "f1": float(metrics.f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (tn + fp)),
    }


def verify(data_dir, out):
    """6단계: 저장된 모델·scaler·설정으로 원본부터 재처리하여 이전 점수와 비교한다.

    joblib 파일은 자신이 생성한 로컬 산출물만 읽는다.
    테스트 정답으로 임곗값을 재학습하지 않는다.
    """
    tf = tensorflow()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config["features"] != FEATURES or config["sequence"] != SEQUENCE or config["horizon"] != HORIZON:
        raise ValueError("저장 설정과 현재 코드의 입력 규격이 다릅니다.")
    frames, report = read_data(data_dir)
    if {k: v["sha256"] for k, v in report.items()} != config["data_sha256"]:
        raise ValueError("학습 시점과 원본 데이터가 다릅니다.")
    scaler = joblib.load(out / "scaler.joblib")
    # prepare를 다시 호출해 scaler를 재학습하지 않고 저장된 scaler만 사용한다.
    if config["split"] == "guide":
        segments = [(frames["normal"].iloc[15000:], 880), (frames["anomaly"], 300)]
    else:
        segments = [(frames["normal"].iloc[16000:], 0), (frames["anomaly"].iloc[300:], 0)]
    xs = []
    for df, skip in segments:
        x, _ = windows(scaler.transform(df[FEATURES].abs()), df.Equipment_state.to_numpy())
        xs.append(x[skip:])
    model = tf.keras.models.load_model(out / "model.h5", compile=False)
    restored = score(model, np.concatenate(xs))
    original = np.load(out / "test_scores.npy", allow_pickle=False)
    np.testing.assert_allclose(restored, original, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(restored >= config["threshold"], original >= config["threshold"])
    result = {"passed": True, "samples": len(restored), "max_absolute_error": float(np.max(np.abs(restored - original)))}
    write_json(out / "reload_check.json", result)
    print("재로딩 검증:", result)


def train(data, scaler, report, args, out):
    tf = tensorflow()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    model = build_model(tf)
    model.summary()
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=50),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00001, patience=120, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    started = time.perf_counter()
    history = model.fit(
        data["train"], data["train"], epochs=args.epochs, batch_size=128,
        validation_data=(data["valid_normal"], data["valid_normal"]), callbacks=callbacks,
        shuffle=True, verbose=2,
    )
    if not all(np.isfinite(values).all() for values in history.history.values()):
        raise ValueError("학습 loss 또는 기록에 NaN/무한대가 있습니다.")
    threshold, selection = select_threshold(data["valid_y"], score(model, data["valid"]), args.threshold_method)
    scores = score(model, data["test"])
    result = evaluate(data["test_y"], scores, threshold)
    config = {
        "features": FEATURES, "absolute_value": True, "sequence": SEQUENCE, "horizon": HORIZON,
        "score": "last_timestep_mse", "split": args.split, "seed": args.seed,
        "threshold": threshold, "threshold_selection": selection,
        "epochs_requested": args.epochs, "epochs_completed": len(history.history["loss"]),
        "seconds": time.perf_counter() - started,
        "data_sha256": {k: v["sha256"] for k, v in report.items()},
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["tensorflow", "numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
    }
    # HDF5는 TF 2.7과 최신 Keras에서 모두 읽을 수 있도록 선택했다.
    model.save(out / "model.h5", include_optimizer=False)
    joblib.dump(scaler, out / "scaler.joblib")
    np.save(out / "test_scores.npy", scores)
    write_json(out / "config.json", config)
    write_json(out / "metrics.json", result)
    write_json(out / "history.json", {k: [float(v) for v in values] for k, values in history.history.items()})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].legend()
    axes[0].set_title("Reconstruction loss")
    axes[1].scatter(np.arange(len(scores)), scores, c=data["test_y"], s=3)
    axes[1].axhline(threshold, color="red")
    axes[1].set_title("Test score / threshold")
    metrics.ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=["Normal", "Anomaly"]).plot(ax=axes[2], colorbar=False)
    fig.tight_layout()
    fig.savefig(out / "evaluation.png")
    plt.close(fig)
    print("평가 결과:", result)
    verify(args.data_dir, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "prepare", "train", "verify"])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", str(Path.home() / "datasets/kamp/fine-blanking-press/raw"))))
    parser.add_argument("--run", default="smoke", help="artifacts 아래 실행 폴더 이름")
    parser.add_argument("--split", choices=["guide", "strict"], default="guide")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-method", choices=["guide", "f1"], default="guide")
    args = parser.parse_args()
    if args.epochs < 1 or not args.run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run):
        parser.error("epochs는 양수, run은 영문/숫자/밑줄/하이픈 이름이어야 합니다.")
    out = ROOT / "artifacts" / args.run
    if args.command == "verify":
        verify(args.data_dir, out)
        return
    # 이전 학습 결과를 덮어쓰지 않도록 실행마다 새 이름을 사용한다.
    if (out / "config.json").exists():
        parser.error("이미 학습된 실행 폴더입니다. 다른 --run 이름을 사용하세요.")
    out.mkdir(parents=True, exist_ok=True)
    frames, report = read_data(args.data_dir)
    write_json(out / "quality.json", report)
    plot_data(frames, out)
    print("데이터 검사:", {k: {f: v[f] for f in ["rows", "duplicate_rows", "non_0_1_second_intervals"]} for k, v in report.items()})
    if args.command == "inspect":
        return
    data, scaler = prepare(frames, args.split)
    shapes = {k: list(v.shape) for k, v in data.items()}
    write_json(out / "shapes.json", {"split": args.split, "shapes": shapes})
    print("데이터 구성:", shapes)
    if args.command == "train":
        train(data, scaler, report, args, out)


if __name__ == "__main__":
    main()
