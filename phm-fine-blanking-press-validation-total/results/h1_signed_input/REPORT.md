# Absolute vs Signed LSTM-AE ablation report

## Result

The single-seed result **partially supports H1** under this fixed protocol. Signed F1 changed from 0.6933 to 0.7595; recall changed from 0.8667 to 0.8333. This is descriptive evidence only, not a statistical claim.

## Test comparison

| index | Precision | Recall | F1 | FPR | FP | FN | Threshold | Best epoch | Best validation loss | Training seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absolute | 0.577778 | 0.866667 | 0.693333 | 0.0285 | 114 | 24 | 0.0028958 | 784 | 0.00251386 | 1104.61 |
| signed | 0.697674 | 0.833333 | 0.759494 | 0.01625 | 65 | 30 | 0.00237339 | 790 | 0.00158128 | 1113.42 |
| signed_minus_absolute | 0.119897 | -0.0333333 | 0.0661603 | -0.01225 | -49 | 6 | -0.00052241 | 6 | -0.000932575 | 8.81031 |

## Observation

Both models used the same rows, window indices, labels, model architecture (63,171 trainable parameters), seed, initial weights, shuffle order, and training controls. Their only intended input difference is applying `abs()` before each model's separately fitted normal-train MinMaxScaler.

## Interpretation

Absolute and Signed reconstruction MSE scales are not directly comparable because the inputs and scalers differ. The valid direct comparison is the prediction/metric result on the shared test windows using each model's validation-only threshold.

## Prediction changes

| index | outcome | count | share_of_test |
| --- | --- | --- | --- |
| 0 | unchanged | 4009 | 0.959091 |
| 1 | signed_removed_FP | 95 | 0.0227273 |
| 2 | signed_new_FP | 46 | 0.0110048 |
| 3 | absolute_correct_signed_wrong | 18 | 0.00430622 |
| 4 | absolute_wrong_signed_correct | 12 | 0.00287081 |

## Limitations

This preserves the source protocol's time-gap-agnostic windows, validation/test input-boundary overlap, and `label_offset=100`. Thresholds come only from validation, but the protocol is exploratory rather than leakage-resistant. One seed does not establish statistical significance or generalization. A larger reconstruction error does not by itself establish a causal sensor fault.
