# KAMP baseline actual-training reproduction result

## Status

No final reproducibility metrics are recorded for this run.

Two full-training processes were started with the unchanged pipeline and `epochs=800`. The first process ran from approximately 00:29 to 01:32 but its final stdout was not retained by the execution wrapper. A second run used `record_reproduction_run.py` to write the final history and metrics to `REPRODUCTION_TRAINING_HISTORY.json`; that process exited without creating the JSON file. Therefore the final threshold, confusion matrix, Accuracy, and F1 cannot be asserted from this execution.

The baseline pipeline was not changed to force a result. In particular, the guidebook's exact `precision == recall` threshold selection was retained. That rule is a likely post-training failure point because it can have no exact matching pair for a particular unseeded training run, as already documented in `REPRODUCTION_NOTES.md`; the lost process stderr cannot establish the exact exception.

## One-epoch end-to-end smoke test

This test completed `fit -> threshold -> evaluate` without a runtime error:

| Item | Observed value |
| --- | ---: |
| Epochs | 1 |
| Train loss | 0.04052620381116867 |
| Validation loss | 0.026890737935900688 |
| Threshold | 0.053006983505958484 |
| Confusion matrix | `[[3807, 193], [81, 99]]` |
| Accuracy | 0.9344497607655502 |
| F1-score | 0.4194915254237288 |
| Fit time | 12.244701 seconds |

## Guidebook comparison

The guidebook reference values (`threshold ~= 0.0035`, `[[3922, 78], [26, 154]]`, Accuracy ~= 97.51%, F1 ~= 74.76%) cannot be compared against an asserted full-run result because the full-run metrics were not persisted. No hyperparameter, split, random seed, or threshold rule was changed to target those values.
