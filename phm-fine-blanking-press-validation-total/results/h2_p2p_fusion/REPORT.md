# H2 — Signed LSTM-AE + W20 vibration P2P fusion

## Execution record

- CUDA inference completed on the saved Signed checkpoint; no LSTM training was launched by this experiment.
- The stored Signed scaler was absent, so it was deterministically reconstructed by fitting `MinMaxScaler` to the original **signed** normal rows 0–14,999, exactly as the first experiment specifies. This provenance is in `preflight.json`.
- Validation selected alpha **2.0** by highest F1 across `[0, 0.5, 1.0, 2.0]` (tie: smaller alpha). Test data were not used for selection.
- Thresholds use normal-validation empirical FPR <= 1%; with 880 normal validation windows, 8 false positives are allowed and strict `score > threshold` is applied.

## Test result

| method | alpha | precision | recall | f1 | false_positive_rate | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | nan | 0.8750 | 0.5444 | 0.6712 | 0.0035 | 14 | 82 |
| B | nan | 1.0000 | 0.6944 | 0.8197 | 0.0000 | 0 | 55 |
| C | 2.0000 | 1.0000 | 0.6944 | 0.8197 | 0.0000 | 0 | 55 |

Relative to A, C changed Precision +0.1250, Recall +0.1500, F1 +0.1484, FPR -0.0035, FP -14, and FN -27.

## Interpretation

H2 is **supports** for this fixed checkpoint and protocol: P2P-only and the selected fusion both achieved the recorded Test metrics above. The selected fusion coincides with P2P-only here because the P2P score dominates after its documented score-IQR floor is applied.

The normalized AE/P2P Pearson correlation on Test is 0.5927; this is descriptive only, not proof of complementarity. Prediction-level evidence is in `prediction_changes.csv`, not correlation alone.

## Important limitation

The one-sided P2P score is zero for most normal-train windows, giving it IQR=0. The pre-registered 1e-6 floor avoids division by zero but magnifies nonzero P2P excesses. Thus fusion alpha > 0 largely ranks by P2P, which must temper any claim that the two signals combine smoothly. This exploratory result also preserves time-gap-agnostic windows and validation/test boundary overlap; it is not an independent generalization, early-warning, or statistical-significance claim.
