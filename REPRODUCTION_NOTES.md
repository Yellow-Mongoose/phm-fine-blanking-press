# KAMP 소성가공 예지보전 가이드 baseline 재현 노트

## Source and scope

- Source: `Guidebook_소성가공 예지보전 AI 데이터셋.pdf`, code 3-33 (guidebook pp. 22-40).
- The Python files reproduce the guidebook's chronological split, scaling, sequence construction, LSTM-Autoencoder, callbacks, reconstruction MSE, precision-recall threshold selection, confusion matrix, Accuracy, and F1 calculation.
- `smoke_test.py` never calls training. `run_baseline.py` also requires an explicit `--train` before it invokes the guidebook's 800-epoch fit.

## Guidebook code ambiguity / minimal correction

1. In guidebook code 21, the first two `range(...)` expressions are visually rendered as `len(...) - sequence 100`; the anomaly expression clearly shows `- sequence - 100`. The implementation uses `len(values) - sequence - 100` for all three cases. This is the only syntactic interpretation consistent with the immediately following `index + sequence + 100` label offset and with the documented 20-step sequence / 100-step offset.
2. The guidebook names anomaly sequence variables `X_test_anomal` / `Y_test_anomal` (without the final `y`) and subsequently uses the same spelling. The implementation retains that internal spelling only where it maps the guidebook steps, while function names use conventional English.
3. The guidebook's threshold rule selects the first exact `precision == recall` point. That exact equality can be absent for a different trained random state; this intentionally remains unchanged, so it can raise `IndexError` rather than silently substituting another rule.

## Environment differences

- The guidebook package-install cell specifies `numpy==1.20.0`; this workspace's locked `environment.yml` uses `numpy==1.19.5` because it is compatible with Windows TensorFlow 2.7.0 in the established Python 3.9.7 environment.
- The guidebook uses notebook magics and inline displays. Those have been converted into plain Python functions and CLI output; data processing, model architecture, split boundaries, callback values, batch size, and 800 epoch setting are unchanged.
- TensorFlow random seeds are not set because the guidebook does not set them. Consequently trained threshold/metrics can vary between runs, as the guidebook itself notes.
