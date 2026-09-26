"""KAMP guidebook baseline pipeline, expressed as importable Python functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from kamp_config import (
    BATCH_SIZE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    LEARNING_RATE,
    OFFSET,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SEQUENCE,
    TRAIN_NORMAL_ROWS,
    VALID_ANOMALY_SEQUENCES,
    VALID_NORMAL_SEQUENCES,
)


def load_data(normal_csv, outlier_csv):
    """Guidebook code 3: load both CSV files with their first column as index."""
    normal = pd.read_csv(normal_csv, index_col=0)
    outlier = pd.read_csv(outlier_csv, index_col=0)
    return normal, outlier


def inspect_data(normal_data: pd.DataFrame, outlier_data: pd.DataFrame) -> dict:
    """Guidebook codes 4-11: return the basic checks without notebook display calls."""
    return {
        "normal_head": normal_data.head(),
        "outlier_head": outlier_data.head(),
        "normal_rows": len(normal_data),
        "outlier_rows": len(outlier_data),
        "normal_describe": normal_data.describe().T,
        "outlier_describe": outlier_data.describe().T,
        "normal_nulls": normal_data.isnull().sum(),
        "outlier_nulls": outlier_data.isnull().sum(),
    }


def preprocess(normal: pd.DataFrame, outlier: pd.DataFrame):
    """Guidebook codes 6, 14, 17, 18: copy data and convert selected values to abs."""
    normal_data = normal.copy()
    outlier_data = outlier.copy()
    normal_data[FEATURE_COLUMNS] = normal_data[FEATURE_COLUMNS].applymap(lambda x: abs(x))
    outlier_data[FEATURE_COLUMNS] = outlier_data[FEATURE_COLUMNS].applymap(lambda x: abs(x))
    return normal_data, outlier_data


def split_and_scale(normal_data: pd.DataFrame, outlier_data: pd.DataFrame):
    """Guidebook codes 19-20: chronological normal split and MinMax scaling."""
    x_normal = normal_data[FEATURE_COLUMNS]
    y_normal = normal_data[LABEL_COLUMN]
    x_anomaly = outlier_data[FEATURE_COLUMNS]
    y_anomaly = outlier_data[LABEL_COLUMN]

    x_train_normal = x_normal[:TRAIN_NORMAL_ROWS]
    y_train_normal = y_normal[:TRAIN_NORMAL_ROWS]
    x_test_normal = x_normal[TRAIN_NORMAL_ROWS:]
    y_test_normal = y_normal[TRAIN_NORMAL_ROWS:]
    x_test_anomaly = x_anomaly
    y_test_anomaly = y_anomaly

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_normal)
    x_test_normal_scaled = scaler.transform(x_test_normal)
    x_test_anomaly_scaled = scaler.transform(x_test_anomaly)

    return {
        "scaler": scaler,
        "x_train_scaled": x_train_scaled,
        "x_test_normal_scaled": x_test_normal_scaled,
        "x_test_anomaly_scaled": x_test_anomaly_scaled,
        "y_train_normal": np.array(y_train_normal),
        "y_test_normal": np.array(y_test_normal),
        "y_test_anomaly": np.array(y_test_anomaly),
    }


def make_sequences(values: np.ndarray, labels: np.ndarray):
    """Guidebook code 21: 20-step input and label 100 steps after the input window."""
    x_sequences, y_sequences = [], []
    for index in range(len(values) - SEQUENCE - OFFSET):
        x_sequences.append(values[index : index + SEQUENCE])
        y_sequences.append(labels[index + SEQUENCE + OFFSET])
    return np.array(x_sequences), np.array(y_sequences)


def prepare_datasets(normal_data: pd.DataFrame, outlier_data: pd.DataFrame):
    """Guidebook codes 19-24: scale, sequence, and construct validation/test sets."""
    split = split_and_scale(normal_data, outlier_data)
    x_train, y_train = make_sequences(split["x_train_scaled"], split["y_train_normal"])
    x_test_normal, y_test_normal = make_sequences(
        split["x_test_normal_scaled"], split["y_test_normal"]
    )
    x_test_anomal, y_test_anomal = make_sequences(
        split["x_test_anomaly_scaled"], split["y_test_anomaly"]
    )

    x_valid_normal, y_valid_normal = (
        x_test_normal[:VALID_NORMAL_SEQUENCES],
        y_test_normal[:VALID_NORMAL_SEQUENCES],
    )
    x_test_normal, y_test_normal = (
        x_test_normal[VALID_NORMAL_SEQUENCES:],
        y_test_normal[VALID_NORMAL_SEQUENCES:],
    )
    x_valid_anomal, y_valid_anomal = (
        x_test_anomal[:VALID_ANOMALY_SEQUENCES],
        y_test_anomal[:VALID_ANOMALY_SEQUENCES],
    )
    x_test_anomal, y_test_anomal = (
        x_test_anomal[VALID_ANOMALY_SEQUENCES:],
        y_test_anomal[VALID_ANOMALY_SEQUENCES:],
    )

    x_valid = np.vstack((x_valid_normal, x_valid_anomal))
    y_valid = np.hstack((y_valid_normal, y_valid_anomal))
    x_test = np.vstack((x_test_normal, x_test_anomal))
    y_test = np.hstack((y_test_normal, y_test_anomal))
    x_valid_0 = x_valid[y_valid == 0]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "x_valid_0": x_valid_0,
        "x_test": x_test,
        "y_test": y_test,
        "scaler": split["scaler"],
    }


def build_lstm_autoencoder(sequence: int = SEQUENCE, n_features: int = len(FEATURE_COLUMNS)):
    """Guidebook code 25 LSTM-Autoencoder architecture."""
    lstm_ae = models.Sequential()
    lstm_ae.add(layers.LSTM(64, input_shape=(sequence, n_features), return_sequences=True))
    lstm_ae.add(layers.LSTM(32, return_sequences=False))
    lstm_ae.add(layers.RepeatVector(sequence))
    lstm_ae.add(layers.LSTM(32, return_sequences=True))
    lstm_ae.add(layers.LSTM(64, return_sequences=True))
    lstm_ae.add(layers.TimeDistributed(layers.Dense(n_features)))
    return lstm_ae


def train_lstm_autoencoder(model, datasets, epochs: int = 800):
    """Guidebook code 26. Call explicitly; smoke tests do not call this function."""
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=EARLY_STOPPING_MIN_DELTA,
        patience=EARLY_STOPPING_PATIENCE,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    model.compile(loss="mse", optimizer=optimizers.Adam(LEARNING_RATE))
    return model.fit(
        datasets["x_train"],
        datasets["x_train"],
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[reduce_lr, early_stopping],
        validation_data=(datasets["x_valid_0"], datasets["x_valid_0"]),
    )


def flatten(x: np.ndarray) -> np.ndarray:
    """Guidebook code 28: retain only the final time step for MSE."""
    flattened = np.empty((x.shape[0], x.shape[2]))
    for index in range(x.shape[0]):
        flattened[index] = x[index, x.shape[1] - 1, :]
    return flattened


def find_threshold(model, x_valid: np.ndarray, y_valid: np.ndarray):
    """Guidebook code 29: precision-recall equality threshold selection."""
    valid_predictions = model.predict(x_valid)
    valid_mse = np.mean(np.power(flatten(x_valid) - flatten(valid_predictions), 2), axis=1)
    precision, recall, threshold = metrics.precision_recall_curve(list(y_valid), valid_mse)
    index_cnt = [
        count for count, (p_value, r_value) in enumerate(zip(precision, recall)) if p_value == r_value
    ][0]
    return threshold[index_cnt], precision, recall, threshold, valid_mse, index_cnt


def plot_threshold_curve(precision, recall, threshold, threshold_final, index_cnt):
    """Guidebook code 29 precision/recall threshold visualization."""
    plt.figure(figsize=(10, 7))
    plt.title("Precision/Recall Curve for threshold", fontsize=15)
    selected = threshold <= 0.2
    plt.plot(threshold[selected], precision[1:][selected], label="Precision")
    plt.plot(threshold[selected], recall[1:][selected], label="Recall")
    plt.plot(threshold_final, precision[index_cnt], "o", color="r", label="Optimal threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Precision/Recall")
    plt.legend()
    plt.show()


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray, threshold_final: float):
    """Guidebook codes 30-33: reconstruction MSE, confusion matrix, Accuracy, and F1."""
    test_predictions = model.predict(x_test)
    mse = np.mean(np.power(flatten(x_test) - flatten(test_predictions), 2), axis=1)
    pred_y = [1 if error > threshold_final else 0 for error in mse]
    return {
        "mse": mse,
        "pred_y": pred_y,
        "confusion_matrix": metrics.confusion_matrix(list(y_test), pred_y),
        "accuracy": metrics.accuracy_score(list(y_test), pred_y),
        "f1_score": metrics.f1_score(list(y_test), pred_y),
    }


def plot_evaluation(y_test: np.ndarray, results: dict, threshold_final: float):
    """Guidebook codes 30-31 reconstruction-error and confusion-matrix plots."""
    mse = results["mse"]
    plt.figure(figsize=(10, 7))
    plt.title("Reconstruction Error for both classes", fontsize=15)
    plt.plot(np.where(y_test == 0)[0], mse[y_test == 0], marker="o", linestyle="", label="Normal")
    plt.plot(np.where(y_test == 1)[0], mse[y_test == 1], marker="o", linestyle="", label="Anomaly")
    plt.axhline(threshold_final, 0, len(y_test), color="r", linestyle="--", label="Threshold for Anomaly")
    plt.legend()
    plt.ylabel("Reconstruction Error")
    plt.show()

    plt.figure(figsize=(7, 7))
    sns.heatmap(results["confusion_matrix"], xticklabels=[0, 1], yticklabels=[0, 1], annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()
