# Controlled Experiment Contract

## Objective

Compare the baseline **single-encoder LSTM-AE** with a **dual-encoder sensor-fusion LSTM-AE**. The only intended experimental change is the encoder topology:

```text
AI0_Vibration ─┐
               ├─ vibration encoder ─┐
AI1_Vibration ─┘                     ├─ latent concat ─ decoder ─ 3-channel reconstruction
AI2_Current ───── current encoder ───┘
```

No attention mechanism, fusion loss, extra feature engineering, or test-driven tuning is permitted in this experiment.

## Source baseline

- Reference notebook: `../../pytorch_reproduction.ipynb`
- Reference model: `LSTMAutoencoder` in notebook cell 17.
- This directory must not modify the reference notebook.

## Data-path contract

The experiment notebook is located in `experiment/sensor_fusion/`. Its default data directory must be resolved by `experiment_config.resolve_project_root()`, which searches upward from an explicit starting directory for the repository layout. It must not rely on a hard-coded absolute path or blindly assume the current working directory:

```python
PROJECT_ROOT = resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data"
```

The notebook must display the resolved project/data paths and assert that both files exist before loading:

- `data/press_data_normal.csv`
- `data/outlier_data.csv`

Raw CSV files are not copied into `sensor_fusion/` and must not be committed.

## Frozen input contract

- Feature order: `AI0_Vibration`, `AI1_Vibration`, `AI2_Current`.
- Labels: `Equipment_state`; normal CSV must contain 20,000 rows of label `0`, and outlier CSV 600 rows of label `1`.
- Each feature is transformed by absolute value before scaling.
- `MinMaxScaler` is fit only on the first 15,000 normal rows.
- The same fitted scaler transforms the remaining normal data and all anomaly data. Transformed anomaly values are not clipped.
- Sequence length: 20.
- Label offset: 100.
- Window rule: for input starting at `i`, use `x[i:i + 20]` and label `y[i + 20 + 100]`; number of windows is `len(x) - 20 - 100`.

## Frozen split contract

| Split | Source | Window count | Labels |
| --- | --- | ---: | --- |
| Train | first 15,000 normal rows | 14,880 | normal only |
| Validation | remaining normal + anomaly | 1,180 | 880 normal, 300 anomaly |
| Test | remaining validation-source windows | 4,180 | 4,000 normal, 180 anomaly |

Expected tensor shapes are:

| Tensor | Shape |
| --- | --- |
| `X_train` | `(14880, 20, 3)` |
| `X_valid` | `(1180, 20, 3)` |
| `X_test` | `(4180, 20, 3)` |
| `X_valid_0` | `(880, 20, 3)` |

## Frozen training contract

- Autoencoder training uses only `X_train`; validation loss uses only `X_valid_0`.
- Loss: mean squared reconstruction error over all sequence positions and channels (`nn.MSELoss`).
- Epoch limit: 800; batch size: 128; learning rate: 0.001; seed: 42.
- Optimizer, learning-rate schedule, initialization, and early-stopping behavior must remain aligned with the reference notebook unless an implementation necessity is documented separately. They are not tuning variables.
- The new model must print its module structure, total parameters, and trainable parameters.

## Frozen dual-encoder topology

- Vibration encoder: `2 → 64 → 16` LSTM, consuming `AI0_Vibration` and `AI1_Vibration` in that order.
- Current encoder: `1 → 32 → 16` LSTM, consuming `AI2_Current`.
- Fusion: concatenate the two final 16-dimensional hidden vectors to form a 32-dimensional fused latent.
- Decoder: preserve the reference dimensions `32 → 32 → 64 → 3`; its output ordering remains `AI0_Vibration`, `AI1_Vibration`, `AI2_Current`.
- Every LSTM uses the same initialization/bias convention as the reference notebook.
- This topology has 63,171 trainable parameters—the same trainable count reported by the reference single-encoder model. It has 64,067 total parameters, including 896 intentionally frozen recurrent-bias parameters.

## Frozen score, threshold, and test contract

- Anomaly score is the mean squared reconstruction error of the **last timestep only**, averaged across the three original channels.
- Select threshold solely from validation scores using the reference notebook's first exact `precision == recall` point from `sklearn.metrics.precision_recall_curve`.
- Preserve the strict anomaly decision rule: `score > threshold`.
- Test labels/scores may be used only once for final evaluation; they must not influence threshold selection, architecture size, or training hyperparameters.
- Report Accuracy, Precision, Recall, F1, confusion matrix, FP count, FN count, validation/test score distributions, threshold curve, and selected threshold.

## Comparison and interpretation contract

- Use the same prepared arrays and the same evaluation functions for both models.
- The comparison table must state parameter counts and all requested test metrics for the baseline and dual-encoder models.
- A result difference is evidence only for this dataset, split, seed, and configuration. It must not be described as a general superiority claim.
- Potential improvements to validation design, split hygiene, threshold selection, or other methodology belong in a separate limitations section and are not implemented in this experiment.

## Phase-1 verification record

The contracts above were transcribed from cells 4, 10, 13, 15, 17, 21, and 23 of `../../pytorch_reproduction.ipynb` on 2026-09-29. The reference notebook's recorded full 800-epoch output uses the single-encoder topology and reports 63,171 trainable parameters.
