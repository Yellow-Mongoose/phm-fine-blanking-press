"""Run the guidebook baseline deliberately; training is opt-in because it uses 800 epochs."""

import argparse

from kamp_config import EPOCHS, NORMAL_CSV, OUTLIER_CSV
from kamp_pipeline import (
    build_lstm_autoencoder,
    evaluate,
    find_threshold,
    inspect_data,
    load_data,
    prepare_datasets,
    preprocess,
    plot_evaluation,
    plot_threshold_curve,
    train_lstm_autoencoder,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run the guidebook's 800-epoch training.")
    args = parser.parse_args()

    normal, outlier = load_data(NORMAL_CSV, OUTLIER_CSV)
    checks = inspect_data(normal, outlier)
    print(f"Normal rows: {checks['normal_rows']}")
    print(f"Outlier rows: {checks['outlier_rows']}")
    normal_data, outlier_data = preprocess(normal, outlier)
    datasets = prepare_datasets(normal_data, outlier_data)
    print("X_train:", datasets["x_train"].shape, "Y_train:", datasets["y_train"].shape)
    print("X_valid:", datasets["x_valid"].shape, "Y_valid:", datasets["y_valid"].shape)
    print("X_test:", datasets["x_test"].shape, "Y_test:", datasets["y_test"].shape)
    model = build_lstm_autoencoder()
    model.summary()

    if not args.train:
        print("Training not run. Re-run with --train to execute the guidebook's 800 epochs.")
        return

    train_lstm_autoencoder(model, datasets, epochs=EPOCHS)
    threshold_final, precision, recall, threshold, _, index_cnt = find_threshold(
        model, datasets["x_valid"], datasets["y_valid"]
    )
    plot_threshold_curve(precision, recall, threshold, threshold_final, index_cnt)
    results = evaluate(model, datasets["x_test"], datasets["y_test"], threshold_final)
    plot_evaluation(datasets["y_test"], results, threshold_final)
    print("Threshold:", threshold_final)
    print("Confusion matrix:\n", results["confusion_matrix"])
    print("Accuracy:", results["accuracy"])
    print("F1-Score:", results["f1_score"])


if __name__ == "__main__":
    main()
