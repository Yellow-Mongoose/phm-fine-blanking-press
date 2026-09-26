"""Non-training smoke test for the reproduced KAMP guidebook baseline."""

from kamp_config import NORMAL_CSV, OUTLIER_CSV
from kamp_pipeline import build_lstm_autoencoder, load_data, prepare_datasets, preprocess


def main():
    normal, outlier = load_data(NORMAL_CSV, OUTLIER_CSV)
    normal_data, outlier_data = preprocess(normal, outlier)
    datasets = prepare_datasets(normal_data, outlier_data)
    model = build_lstm_autoencoder()

    assert normal.shape == (20_000, 5)
    assert outlier.shape == (600, 5)
    assert datasets["x_train"].shape == (14_880, 20, 3)
    assert datasets["x_valid"].shape == (1_180, 20, 3)
    assert datasets["x_valid_0"].shape == (880, 20, 3)
    assert datasets["x_test"].shape == (4_180, 20, 3)
    assert model.input_shape == (None, 20, 3)
    assert model.output_shape == (None, 20, 3)

    print("CSV load: PASS", normal.shape, outlier.shape)
    print("Sequence/split shapes: PASS", datasets["x_train"].shape, datasets["x_valid"].shape, datasets["x_test"].shape)
    print("LSTM-Autoencoder creation: PASS", model.input_shape, model.output_shape)


if __name__ == "__main__":
    main()
