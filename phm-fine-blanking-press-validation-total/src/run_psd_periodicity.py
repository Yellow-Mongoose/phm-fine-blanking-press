"""Validation-3: signed LSTM-AE error with current PSD/autocorrelation features.

The default action is a non-destructive preflight.  It never trains a model and
never scores the test set.  `--run-validation` is deliberately separate from
`--run-test`, so a test evaluation cannot happen by accident.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import nn

FEATURES = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
EXPECTED_COLUMNS = ["TimeStamp", *FEATURES, "Equipment_state"]
FS_HZ, GAP_SECONDS, EPS = 10.0, 0.150, 1e-12
ALPHAS = (0.0, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Config:
    sequence: int = 20
    label_offset: int = 100
    train_rows: int = 15000
    valid_normal: int = 880
    valid_anomaly: int = 300
    autocorr_lags: tuple = tuple(range(1, 11))
    fpr_target: float = 0.01


def default_json(value):
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default_json), encoding="utf-8")


def load_csvs(data_dir: Path):
    normal = pd.read_csv(data_dir / "press_data_normal.csv", index_col=0)
    anomaly = pd.read_csv(data_dir / "outlier_data.csv", index_col=0)
    for name, frame, rows, state in (("normal", normal, 20000, 0), ("outlier", anomaly, 600, 1)):
        if list(frame.columns) != EXPECTED_COLUMNS or len(frame) != rows:
            raise ValueError(f"Unexpected {name} schema/row count")
        if not frame.Equipment_state.eq(state).all() or frame.isna().any().any():
            raise ValueError(f"Unexpected {name} labels or missing values")
        if not np.isfinite(frame[FEATURES].to_numpy()).all():
            raise ValueError(f"Non-finite sensor values in {name}")
    return normal, anomaly


def make_manifest(normal, anomaly, cfg: Config):
    """Exactly preserve the source's segment-wise W20 and label_offset indexing."""
    rows = []
    specs = (("train", "normal", normal, 0, cfg.train_rows - cfg.sequence - cfg.label_offset),
             ("valid", "normal", normal, cfg.train_rows, cfg.valid_normal),
             ("test", "normal", normal, cfg.train_rows + cfg.valid_normal,
              len(normal) - cfg.train_rows - cfg.sequence - cfg.label_offset - cfg.valid_normal),
             ("valid", "anomaly", anomaly, 0, cfg.valid_anomaly),
             ("test", "anomaly", anomaly, cfg.valid_anomaly,
              len(anomaly) - cfg.sequence - cfg.label_offset - cfg.valid_anomaly))
    for split, source, frame, first_start, count in specs:
        for start in range(first_start, first_start + count):
            end, label_row = start + cfg.sequence - 1, start + cfg.sequence + cfg.label_offset
            rows.append(dict(split=split, source=source, input_start_row=start, input_end_row=end,
                             label_row=label_row, label=int(frame.Equipment_state.iloc[label_row]),
                             input_start_time=frame.TimeStamp.iloc[start], input_end_time=frame.TimeStamp.iloc[end],
                             label_time=frame.TimeStamp.iloc[label_row]))
    return pd.DataFrame(rows)


def timestamp_diagnostics(frame: pd.DataFrame, manifest: pd.DataFrame, source: str, cfg: Config):
    ts = pd.to_datetime(frame.TimeStamp, errors="raise")
    delta = ts.diff().dt.total_seconds().to_numpy()[1:]
    # A W20 is eligible only if all 19 internal intervals are <= 150 ms.
    contiguous = np.ones(len(manifest), dtype=bool)
    selected = manifest.source.eq(source).to_numpy()
    for pos, start in zip(np.flatnonzero(selected), manifest.loc[selected, "input_start_row"]):
        intervals = delta[int(start):int(start) + cfg.sequence - 1]
        contiguous[pos] = len(intervals) == cfg.sequence - 1 and bool(np.all(intervals <= GAP_SECONDS))
    irregular = delta[np.isfinite(delta) & ~np.isclose(delta, 1 / FS_HZ, rtol=0, atol=1e-6)]
    return contiguous, {"rows": len(frame), "timestamp_monotonic": bool(ts.is_monotonic_increasing),
                        "nominal_interval_seconds": 1 / FS_HZ, "gap_threshold_seconds": GAP_SECONDS,
                        "interval_min_seconds": float(np.nanmin(delta)), "interval_max_seconds": float(np.nanmax(delta)),
                        "intervals_over_150ms": int((delta > GAP_SECONDS).sum()),
                        "intervals_not_100ms": int(len(irregular)),
                        "unique_intervals_seconds": pd.Series(delta).value_counts().head(20).to_dict()}


def raw_windows(frame: pd.DataFrame, manifest: pd.DataFrame, source: str):
    selected = manifest.source.eq(source).to_numpy()
    positions = np.flatnonzero(selected)
    data = frame.AI2_Current.to_numpy(dtype=np.float64)
    windows = np.stack([data[int(manifest.input_start_row.iloc[p]):int(manifest.input_end_row.iloc[p]) + 1]
                        for p in positions])
    return positions, windows


def psd_features(windows: np.ndarray):
    """Mean remove -> Hann -> rFFT periodogram; DC excluded and relative power used."""
    n = windows.shape[1]
    centered = windows - windows.mean(axis=1, keepdims=True)
    tapered = centered * np.hanning(n)[None, :]
    power = np.abs(np.fft.rfft(tapered, axis=1)) ** 2
    power = power[:, 1:]
    frequencies = np.fft.rfftfreq(n, d=1 / FS_HZ)[1:]
    total = power.sum(axis=1)
    valid = np.isfinite(total) & (total > EPS)
    relative = np.full_like(power, np.nan)
    relative[valid] = power[valid] / total[valid, None]
    entropy = np.full(len(windows), np.nan)
    entropy[valid] = -np.sum(relative[valid] * np.log(np.maximum(relative[valid], EPS)), axis=1) / np.log(power.shape[1])
    peak = np.zeros(len(windows), dtype=int)
    peak[valid] = np.argmax(relative[valid], axis=1)
    dominant_frequency = np.full(len(windows), np.nan); dominant_ratio = np.full(len(windows), np.nan)
    dominant_frequency[valid] = frequencies[peak[valid]]; dominant_ratio[valid] = relative[np.flatnonzero(valid), peak[valid]]
    return {"spectral_entropy": entropy, "dominant_frequency_hz": dominant_frequency,
            "dominant_relative_power": dominant_ratio, "total_power": total, "psd_relative": relative,
            "frequencies_hz": frequencies, "valid": valid}


def autocorrelation(windows: np.ndarray, lags):
    centered = windows - windows.mean(axis=1, keepdims=True)
    energy = np.sum(centered ** 2, axis=1)
    valid = np.isfinite(energy) & (energy > EPS)
    values = np.full((len(windows), len(lags)), np.nan)
    for j, lag in enumerate(lags):
        values[valid, j] = np.sum(centered[valid, :-lag] * centered[valid, lag:], axis=1) / energy[valid]
    return values, valid


def robust_reference(values, name: str, percentile: float):
    values = np.asarray(values, dtype=float); finite = values[np.isfinite(values)]
    if not len(finite): raise ValueError(f"{name}: no finite normal-train values")
    median, threshold = np.median(finite), np.percentile(finite, percentile)
    iqr = np.percentile(finite, 75) - np.percentile(finite, 25)
    if not np.isfinite(iqr) or iqr <= EPS:
        raise ValueError(f"{name}: normal-train IQR is zero/too small ({iqr}); score intentionally not fabricated")
    return {"n": len(finite), "median": float(median), "percentile": percentile,
            "reference_quantile": float(threshold), "iqr": float(iqr)}


def one_sided_high(values, ref):
    return np.maximum(0.0, (np.asarray(values) - ref["reference_quantile"]) / ref["iqr"])


def one_sided_low(values, ref):
    return np.maximum(0.0, (ref["reference_quantile"] - np.asarray(values)) / ref["iqr"])


def normalize_from_normal_train(values, normal_train_scores, name: str):
    """Robust normalization used before score fusion; never substitutes a tiny divisor."""
    ref = robust_reference(normal_train_scores, name, 50)
    return (np.asarray(values) - ref["median"]) / ref["iqr"], ref


def fusion_score(ae_score, periodicity_score, normal_train_ae, normal_train_periodicity, alpha, name):
    """C/D fusion implementation; callers record the raised IQR exception verbatim."""
    ae_norm, ae_ref = normalize_from_normal_train(ae_score, normal_train_ae, "ae_score")
    periodicity_norm, periodicity_ref = normalize_from_normal_train(periodicity_score, normal_train_periodicity, name)
    return ae_norm + alpha * periodicity_norm, {"ae": ae_ref, name: periodicity_ref, "alpha": alpha}


class LSTMAutoencoder(nn.Module):
    def __init__(self, sequence=20):
        super().__init__(); self.sequence = sequence
        self.encoder1, self.encoder2 = nn.LSTM(3, 64, batch_first=True), nn.LSTM(64, 32, batch_first=True)
        self.decoder1, self.decoder2 = nn.LSTM(32, 32, batch_first=True), nn.LSTM(32, 64, batch_first=True)
        self.output = nn.Linear(64, 3)
    def forward(self, x):
        x, _ = self.encoder1(x); _, (h, _) = self.encoder2(x); x = h[-1].unsqueeze(1).repeat(1, self.sequence, 1)
        x, _ = self.decoder1(x); x, _ = self.decoder2(x); return self.output(x)


def require_cuda():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; no CPU fallback or retraining is permitted")
    return torch.device("cuda:0")


def load_signed_artifacts(checkpoint: Path, scaler_path: Path, cfg):
    if not checkpoint.is_file() or not scaler_path.is_file():
        missing = [str(p) for p in (checkpoint, scaler_path) if not p.is_file()]
        raise FileNotFoundError("Signed checkpoint/scaler unavailable: " + "; ".join(missing))
    device = require_cuda(); payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("mode") not in (None, "signed"): raise ValueError("Checkpoint is not marked as signed")
    model = LSTMAutoencoder(cfg.sequence).to(device); model.load_state_dict(payload["model_state_dict"]); model.eval()
    scaler = joblib.load(scaler_path)
    if getattr(scaler, "n_features_in_", None) != 3: raise ValueError("Scaler does not have three sensor features")
    return model, scaler, device, {"checkpoint": str(checkpoint), "scaler": str(scaler_path),
                                   "checkpoint_mode": payload.get("mode"), "checkpoint_epoch": payload.get("epoch")}


def ae_scores(model, scaler, device, frame, manifest, source):
    rows = frame[FEATURES].to_numpy(); scaled = scaler.transform(rows)
    idx = np.flatnonzero(manifest.source.eq(source).to_numpy())
    x = np.stack([scaled[int(manifest.input_start_row.iloc[p]):int(manifest.input_end_row.iloc[p]) + 1] for p in idx]).astype(np.float32)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), 64): outputs.append(model(torch.as_tensor(x[start:start + 64], device=device)).cpu().numpy())
    recon = np.concatenate(outputs); score = np.mean((x[:, -1] - recon[:, -1]) ** 2, axis=1)
    return idx, score, x, recon


def fpr_threshold(normal_validation_scores, target):
    values = np.asarray(normal_validation_scores); values = values[np.isfinite(values)]
    if not len(values): raise ValueError("No finite normal validation scores for FPR threshold")
    # `higher` makes empirical FPR <= target when scores are thresholded with >.
    try:
        return float(np.quantile(values, 1 - target, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, 1 - target, interpolation="higher"))


def score_metrics(labels, scores, threshold):
    pred = (np.asarray(scores) > threshold).astype(int); cm = metrics.confusion_matrix(labels, pred, labels=[0, 1])
    return {"threshold": float(threshold), "precision": float(metrics.precision_score(labels, pred, zero_division=0)),
            "recall": float(metrics.recall_score(labels, pred, zero_division=0)), "f1": float(metrics.f1_score(labels, pred, zero_division=0)),
            "false_positive_rate": float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else None,
            "false_negatives": int(cm[1, 0]), "confusion_matrix": cm.tolist()}, pred


def plot_feature_distributions(table, out):
    for column in ("spectral_entropy", "dominant_frequency_hz", "autocorrelation"):
        plt.figure(figsize=(7, 4))
        for label, name in ((0, "normal"), (1, "anomaly")):
            x = table.loc[table.label.eq(label), column].dropna()
            if len(x): plt.hist(x, bins=30, alpha=.55, density=True, label=name)
        plt.title(column); plt.legend(); plt.tight_layout(); plt.savefig(out / f"distribution_{column}.png", dpi=140); plt.close()


def preflight(data_dir, out, checkpoint, scaler_path):
    cfg = Config(); normal, anomaly = load_csvs(data_dir); manifest = make_manifest(normal, anomaly, cfg)
    normal_ok, normal_time = timestamp_diagnostics(normal, manifest, "normal", cfg)
    anomaly_ok, anomaly_time = timestamp_diagnostics(anomaly, manifest, "anomaly", cfg)
    eligible = normal_ok & anomaly_ok  # only one source applies per row; the other starts true
    # Correct source-specific mask (above arrays are full manifest and set only their own rows).
    eligible = np.where(manifest.source.eq("normal"), normal_ok, anomaly_ok)
    manifest["continuous_w20"] = eligible
    feature_table = manifest.copy()
    for source, frame in (("normal", normal), ("anomaly", anomaly)):
        pos, windows = raw_windows(frame, manifest, source); p = psd_features(windows); ac, ac_valid = autocorrelation(windows, cfg.autocorr_lags)
        for key in ("spectral_entropy", "dominant_frequency_hz", "dominant_relative_power", "total_power"):
            feature_table.loc[pos, key] = p[key]
        feature_table.loc[pos, "psd_valid"] = p["valid"]
        feature_table.loc[pos, "autocorr_valid"] = ac_valid
        for j, lag in enumerate(cfg.autocorr_lags): feature_table.loc[pos, f"acf_lag_{lag}"] = ac[:, j]
    feature_table["psd_valid"] = feature_table["psd_valid"].astype(bool)
    feature_table["autocorr_valid"] = feature_table["autocorr_valid"].astype(bool)
    feature_table["feature_valid"] = feature_table.continuous_w20 & feature_table.psd_valid & feature_table.autocorr_valid
    train = feature_table.split.eq("train") & feature_table.feature_valid
    entropy_ref = robust_reference(feature_table.loc[train, "spectral_entropy"], "spectral entropy", 95)
    mean_acf = feature_table.loc[train, [f"acf_lag_{x}" for x in cfg.autocorr_lags]].mean()
    selected_lag = int(mean_acf.idxmax().removeprefix("acf_lag_"))
    feature_table["autocorrelation"] = feature_table[f"acf_lag_{selected_lag}"]
    autocorr_ref = robust_reference(feature_table.loc[train, "autocorrelation"], "autocorrelation", 5)
    feature_table["psd_score"] = one_sided_high(feature_table.spectral_entropy, entropy_ref)
    feature_table["autocorr_score"] = one_sided_low(feature_table.autocorrelation, autocorr_ref)
    fusion_scale = {}
    for name in ("psd_score", "autocorr_score"):
        try:
            fusion_scale[name] = {"available": True, **robust_reference(feature_table.loc[train, name], name, 50)}
        except ValueError as exc:
            fusion_scale[name] = {"available": False, "error": str(exc),
                                  "zero_or_nonfinite_train_scores": int((~np.isfinite(feature_table.loc[train, name]) | (feature_table.loc[train, name] == 0)).sum())}
    artifact = {"available": False}
    try:
        _, _, device, artifact = load_signed_artifacts(checkpoint, scaler_path, cfg)
        artifact.update({"available": True, "cuda": str(device), "gpu": torch.cuda.get_device_name(device)})
    except Exception as exc:
        artifact.update({"error_type": type(exc).__name__, "error": str(exc), "cuda_available": bool(torch.cuda.is_available())})
    feature_table.to_csv(out / "preflight_features.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(out / "split_manifest_continuity.csv", index=False, encoding="utf-8-sig")
    report = {"status": "PASS_WITH_ARTIFACT_BLOCK" if not artifact["available"] else "PASS",
              "scope": "Preflight only: no model training, validation inference, or test evaluation was run.",
              "config": asdict(cfg), "frequency": {"fs_hz": FS_HZ, "w20_resolution_hz": FS_HZ / cfg.sequence,
                  "method": "mean removal, Hann, rFFT periodogram, DC exclusion, relative power"},
              "time_continuity": {"normal": normal_time, "anomaly": anomaly_time,
                  "all_windows_before_filter": int(len(manifest)), "continuous_w20": int(manifest.continuous_w20.sum()),
                  "feature_valid_w20": int(feature_table.feature_valid.sum()),
                  "by_split_source": {f"{split}/{source}": {k: int(v) for k, v in values.items()}
                                      for (split, source), values in feature_table.groupby(["split", "source"])["feature_valid"].agg(["size", "sum"]).to_dict("index").items()},
                  "note": "Intervals differing from 100 ms are reported; no interpolation is performed."},
              "psd_numeric_stability": {"invalid_psd_windows": int((~feature_table.psd_valid).sum()),
                  "entropy_min": float(feature_table.spectral_entropy.min()), "entropy_max": float(feature_table.spectral_entropy.max()),
                  "entropy_reference_normal_train": entropy_ref},
              "autocorrelation": {"candidate_lags": list(cfg.autocorr_lags), "normal_train_mean_by_lag": mean_acf.to_dict(),
                  "selected_lag": selected_lag, "score": "max(0, (normal_train_q05 - acf_at_selected_lag) / normal_train_IQR)",
                  "reference": autocorr_ref},
              "scores": {"psd": "max(0, (entropy - normal_train_q95) / normal_train_IQR)",
                  "normalization_for_fusion": "Each AE/PSD/autocorrelation score is robust-normalized as (score - normal_train_median) / normal_train_IQR; denominator <= 1e-12 raises an error.",
                  "feature_score_normalization_readiness": fusion_scale,
                  "fusion": "C = normalized_AE + alpha*normalized_PSD; D = normalized_AE + alpha*normalized_autocorrelation; alpha in [0, .5, 1, 2].",
                  "threshold": "For every method/candidate, threshold is the higher empirical 99th percentile of its eligible normal-validation scores; prediction is score > threshold."},
              "artifact_loading": artifact,
              "comparison_alignment": "All A/B/C/D must be built from the same feature_valid manifest rows; validation/test evaluation asserts identical index and labels."}
    write_json(out / "preflight.json", report); return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "preflight")
    parser.add_argument("--signed-checkpoint", type=Path, default=root.parent / "phm-fine-blanking-press-val" / "results" / "gpu_ablation_20261004" / "signed" / "checkpoint_best.pt")
    parser.add_argument("--signed-scaler", type=Path, default=root.parent / "phm-fine-blanking-press-val" / "results" / "gpu_ablation_20261004" / "signed" / "scaler.joblib")
    parser.add_argument("--run-validation", action="store_true", help="Reserved for approved validation inference; not implemented in this preflight delivery.")
    parser.add_argument("--run-test", action="store_true", help="Refused: final test evaluation is outside the approved current scope.")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_validation or args.run_test:
        raise SystemExit("This delivery is intentionally preflight-only. Obtain approval before enabling inference/evaluation.")
    try:
        report = preflight(args.data_dir, args.output_dir, args.signed_checkpoint, args.signed_scaler)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=default_json))
        return 0
    except Exception as exc:
        failure = {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc), "python": sys.version,
                   "platform": platform.platform(), "torch": torch.__version__}
        write_json(args.output_dir / "preflight.json", failure); print(json.dumps(failure, ensure_ascii=False, indent=2)); return 1


if __name__ == "__main__": raise SystemExit(main())
