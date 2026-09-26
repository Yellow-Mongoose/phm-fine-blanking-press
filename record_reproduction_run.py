"""One-off result recorder for the unchanged KAMP baseline pipeline."""

import json
import time
from pathlib import Path

from kamp_config import EPOCHS, NORMAL_CSV, OUTLIER_CSV
from kamp_pipeline import (
    build_lstm_autoencoder,
    evaluate,
    find_threshold,
    load_data,
    prepare_datasets,
    preprocess,
    train_lstm_autoencoder,
)


def main():
    normal, outlier = load_data(NORMAL_CSV, OUTLIER_CSV)
    normal_data, outlier_data = preprocess(normal, outlier)
    datasets = prepare_datasets(normal_data, outlier_data)
    model = build_lstm_autoencoder()

    started = time.perf_counter()
    history = train_lstm_autoencoder(model, datasets, epochs=EPOCHS)
    elapsed_seconds = time.perf_counter() - started
    threshold, _, _, _, _, _ = find_threshold(model, datasets["x_valid"], datasets["y_valid"])
    results = evaluate(model, datasets["x_test"], datasets["y_test"], threshold)

    record = {
        "configured_max_epochs": EPOCHS,
        "actual_epochs": len(history.history["loss"]),
        "final_train_loss": history.history["loss"][-1],
        "final_validation_loss": history.history["val_loss"][-1],
        "threshold": threshold,
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "accuracy": results["accuracy"],
        "f1_score": results["f1_score"],
        "training_seconds": elapsed_seconds,
        "history": history.history,
    }
    Path("REPRODUCTION_TRAINING_HISTORY.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
