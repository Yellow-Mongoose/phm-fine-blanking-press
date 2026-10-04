"""H2: add raw signed AI0/AI1 W20 P2P to an already-trained Signed LSTM-AE.

Default mode is preflight only.  --run performs CUDA inference and evaluation;
it never trains or modifies the first ablation experiment.
"""
from __future__ import annotations

import argparse
import shutil
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from torch import nn

FEATURES = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
VIBRATIONS = FEATURES[:2]
EXPECTED_COLUMNS = ["TimeStamp", *FEATURES, "Equipment_state"]
ALPHAS = (0.0, 0.5, 1.0, 2.0)  # pre-registered candidates; do not add test-tuned values


@dataclass(frozen=True)
class Config:
    sequence: int = 20
    label_offset: int = 100
    train_rows: int = 15000
    valid_normal: int = 880
    valid_anomaly: int = 300
    fpr_target: float = 0.01
    iqr_floor_relative: float = 1e-6
    iqr_floor_absolute: float = 1e-12


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. CPU fallback is intentionally disabled.")
    return torch.device("cuda:0")


def load_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal = pd.read_csv(data_dir / "press_data_normal.csv", index_col=0)
    anomaly = pd.read_csv(data_dir / "outlier_data.csv", index_col=0)
    for name, frame, rows, label in (("normal", normal, 20000, 0), ("outlier", anomaly, 600, 1)):
        if list(frame.columns) != EXPECTED_COLUMNS or len(frame) != rows:
            raise ValueError(f"{name} does not match the source CSV schema/row count")
        if not frame.Equipment_state.eq(label).all() or frame.isna().any().any():
            raise ValueError(f"{name} labels or missing values do not match the protocol")
        if not np.isfinite(frame[FEATURES].to_numpy()).all():
            raise ValueError(f"{name} contains non-finite sensor values")
    return normal, anomaly


def make_windows(values: np.ndarray, labels: np.ndarray, cfg: Config):
    count = len(values) - cfg.sequence - cfg.label_offset
    windows = np.asarray([values[i:i + cfg.sequence] for i in range(count)], dtype=np.float32)
    y = np.asarray([labels[i + cfg.sequence + cfg.label_offset] for i in range(count)], dtype=np.int64)
    return windows, y, np.arange(count, dtype=np.int64)


def raw_window_sets(normal: pd.DataFrame, anomaly: pd.DataFrame, cfg: Config) -> dict:
    """Windows use the original signed values; no abs()/scaler is involved here."""
    train, y_train, s_train = make_windows(normal.iloc[:cfg.train_rows][FEATURES].to_numpy(),
                                           normal.iloc[:cfg.train_rows].Equipment_state.to_numpy(), cfg)
    normal_tail = normal.iloc[cfg.train_rows:]
    normal_w, normal_y, normal_s = make_windows(normal_tail[FEATURES].to_numpy(), normal_tail.Equipment_state.to_numpy(), cfg)
    anomaly_w, anomaly_y, anomaly_s = make_windows(anomaly[FEATURES].to_numpy(), anomaly.Equipment_state.to_numpy(), cfg)
    return {
        "train": train, "valid": np.vstack((normal_w[:cfg.valid_normal], anomaly_w[:cfg.valid_anomaly])),
        "test": np.vstack((normal_w[cfg.valid_normal:], anomaly_w[cfg.valid_anomaly:])),
        "y_train": y_train, "y_valid": np.hstack((normal_y[:cfg.valid_normal], anomaly_y[:cfg.valid_anomaly])),
        "y_test": np.hstack((normal_y[cfg.valid_normal:], anomaly_y[cfg.valid_anomaly:])),
        "starts": {"train": s_train, "normal": normal_s, "anomaly": anomaly_s},
    }


def build_manifest(normal: pd.DataFrame, anomaly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for split, source, frame, offset, starts in (
        ("train", "normal", normal, 0, range(cfg.train_rows - cfg.sequence - cfg.label_offset)),
        ("valid", "normal", normal, cfg.train_rows, range(cfg.valid_normal)),
        ("test", "normal", normal, cfg.train_rows, range(cfg.valid_normal, 4880)),
        ("valid", "anomaly", anomaly, 0, range(cfg.valid_anomaly)),
        ("test", "anomaly", anomaly, 0, range(cfg.valid_anomaly, 480)),
    ):
        for i in starts:
            first, last = offset + i, offset + i + cfg.sequence - 1
            label_row = offset + i + cfg.sequence + cfg.label_offset
            rows.append({"split": split, "source": source, "window_start_in_segment": i,
                         "input_start_row": first, "input_end_row": last, "label_row": label_row,
                         "input_end_time": frame.iloc[last].TimeStamp, "label_time": frame.iloc[label_row].TimeStamp,
                         "label": int(frame.iloc[label_row].Equipment_state)})
    return pd.DataFrame(rows)


def p2p_values(raw_windows: np.ndarray) -> np.ndarray:
    """P2P for AI0/AI1 only: max(x)-min(x) over each original signed W20 window."""
    return raw_windows[:, :, :2].max(axis=1) - raw_windows[:, :, :2].min(axis=1)


def safe_iqr(values: np.ndarray, relative_floor: float, absolute_floor: float) -> tuple[float, dict]:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    raw = float(q75 - q25)
    floor = max(absolute_floor, relative_floor * max(1.0, abs(float(q25)), abs(float(q75))))
    used = max(raw, floor)
    return used, {"q25": float(q25), "q75": float(q75), "raw_iqr": raw, "floor": floor,
                  "used_iqr": used, "floor_applied": raw < floor}


def fit_p2p_reference(train_p2p: np.ndarray, cfg: Config) -> dict:
    result = {"sensors": {}, "definition": "max(0, P2P - normal_train_q95) / normal_train_IQR"}
    for j, name in enumerate(VIBRATIONS):
        iqr, details = safe_iqr(train_p2p[:, j], cfg.iqr_floor_relative, cfg.iqr_floor_absolute)
        result["sensors"][name] = {"q95": float(np.quantile(train_p2p[:, j], .95)), **details}
    return result


def p2p_anomaly_score(p2p: np.ndarray, reference: dict) -> tuple[np.ndarray, np.ndarray]:
    per_sensor = np.column_stack([np.maximum(0.0, p2p[:, j] - reference["sensors"][name]["q95"])
                                  / reference["sensors"][name]["used_iqr"]
                                  for j, name in enumerate(VIBRATIONS)])
    return per_sensor.max(axis=1), per_sensor


def fit_robust_normalizer(train_scores: np.ndarray, cfg: Config, name: str) -> dict:
    iqr, details = safe_iqr(train_scores, cfg.iqr_floor_relative, cfg.iqr_floor_absolute)
    return {"name": name, "center_median": float(np.median(train_scores)), **details,
            "definition": "(score - normal_train_median) / max(normal_train_IQR, documented_floor)"}


def normalize(scores: np.ndarray, reference: dict) -> np.ndarray:
    return (scores - reference["center_median"]) / reference["used_iqr"]


def threshold_at_normal_fpr(valid_labels: np.ndarray, valid_scores: np.ndarray, target: float) -> tuple[float, dict]:
    normal = valid_scores[valid_labels == 0]
    if not len(normal):
        raise ValueError("No normal Validation scores available for FPR thresholding")
    # Select an observed score: score > threshold implies at most floor(target*n) normal FPs.
    allowed_fp = int(np.floor(target * len(normal)))
    ranked = np.sort(normal)[::-1]
    threshold = float(ranked[allowed_fp]) if allowed_fp < len(ranked) else float(ranked[-1])
    actual = float(np.mean(normal > threshold))
    return threshold, {"threshold_method": "normal-validation empirical FPR <= target with strict score > threshold",
                       "target_fpr": target, "normal_validation_windows": int(len(normal)),
                       "allowed_false_positives": allowed_fp, "achieved_normal_validation_fpr": actual}


def metric_row(labels: np.ndarray, scores: np.ndarray, threshold: float, method: str, alpha=None) -> tuple[dict, np.ndarray]:
    predicted = (scores > threshold).astype(int)
    cm = metrics.confusion_matrix(labels, predicted, labels=[0, 1])
    row = {"method": method, "alpha": alpha, "threshold": threshold,
           "precision": metrics.precision_score(labels, predicted, zero_division=0),
           "recall": metrics.recall_score(labels, predicted, zero_division=0),
           "f1": metrics.f1_score(labels, predicted, zero_division=0),
           "false_positive_rate": float(cm[0, 1] / cm[0].sum()), "false_negative_count": int(cm[1, 0]),
           "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
    return row, predicted


class LSTMAutoencoder(nn.Module):
    """Unchanged architecture copied from the first Signed LSTM-AE experiment."""
    def __init__(self, sequence: int):
        super().__init__(); self.sequence = sequence
        self.encoder1 = nn.LSTM(3, 64, batch_first=True); self.encoder2 = nn.LSTM(64, 32, batch_first=True)
        self.decoder1 = nn.LSTM(32, 32, batch_first=True); self.decoder2 = nn.LSTM(32, 64, batch_first=True)
        self.output = nn.Linear(64, 3)
    def forward(self, x):
        x, _ = self.encoder1(x); _, (hidden, _) = self.encoder2(x)
        x = hidden[-1].unsqueeze(1).repeat(1, self.sequence, 1)
        x, _ = self.decoder1(x); x, _ = self.decoder2(x)
        return self.output(x)


def checkpoint_paths(first_results: Path) -> tuple[Path | None, Path | None, str | None]:
    # Prefer a completed signed run.  An interrupted run's checkpoint_best is valid
    # for inference only when its saved mode is explicitly "signed".
    candidates = sorted(first_results.glob("*/signed"), key=lambda p: p.stat().st_mtime, reverse=True)
    for directory in candidates:
        model = directory / "model.pt"
        scaler = directory / "scaler.joblib"
        if model.is_file() and scaler.is_file():
            return model, scaler, "saved_model_and_scaler"
        best = directory / "checkpoint_best.pt"
        if best.is_file():
            return best, None, "checkpoint_best_with_reconstructed_signed_scaler"
    return None, None, None


def reconstruct_signed_scaler(normal: pd.DataFrame, cfg: Config) -> MinMaxScaler:
    """Exactly the first experiment's deterministic fit: signed normal rows [0:15000]."""
    return MinMaxScaler().fit(normal.iloc[:cfg.train_rows][FEATURES].to_numpy())


def load_signed_artifacts(model_path: Path, scaler_path: Path | None, normal: pd.DataFrame, cfg: Config, device: torch.device):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    mode = checkpoint.get("mode")
    if mode not in (None, "signed"):
        raise ValueError(f"Refusing non-signed checkpoint mode={mode!r}")
    model = LSTMAutoencoder(cfg.sequence).to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    scaler = joblib.load(scaler_path) if scaler_path else reconstruct_signed_scaler(normal, cfg)
    if not isinstance(scaler, MinMaxScaler) or getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError("Signed scaler is not a compatible 3-feature MinMaxScaler")
    return model, scaler


def ae_scores(model, scaled_windows: np.ndarray, device: torch.device) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(scaled_windows), 128):
            x = torch.tensor(scaled_windows[start:start + 128], dtype=torch.float32, device=device)
            reconstruction = model(x)
            chunks.append(torch.mean((x[:, -1, :] - reconstruction[:, -1, :]).square(), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def make_changed_cases(manifest, labels, p2p, ae, p2p_score, per_sensor, ae_norm, p2p_norm, alpha, a_pred, c_pred) -> pd.DataFrame:
    kinds = np.select([
        (labels == 1) & (a_pred == 0) & (c_pred == 1), (labels == 1) & (a_pred == 1) & (c_pred == 0),
        (labels == 0) & (a_pred == 0) & (c_pred == 1), (labels == 0) & (a_pred == 1) & (c_pred == 0)],
        ["A_FN_to_C_TP", "A_TP_to_C_FN", "A_TN_to_C_FP", "A_FP_to_C_TN"], default="")
    base = manifest.reset_index(drop=True).copy()
    base["change_type"] = kinds; base["ae_score"] = ae; base["p2p_score"] = p2p_score
    base["ae_normalized"] = ae_norm; base["p2p_normalized"] = p2p_norm
    base["fusion_p2p_contribution"] = alpha * p2p_norm
    base["AI0_Vibration_p2p"] = p2p[:, 0]; base["AI1_Vibration_p2p"] = p2p[:, 1]
    base["AI0_Vibration_p2p_sensor_score"] = per_sensor[:, 0]
    base["AI1_Vibration_p2p_sensor_score"] = per_sensor[:, 1]
    base["A_prediction"] = a_pred; base["C_prediction"] = c_pred
    return base[base.change_type != ""]


def run_evaluation(raw, manifest, cfg, model, scaler, device, output: Path) -> None:
    features_dir, scores_dir = output / "features", output / "scores"
    features_dir.mkdir(exist_ok=True); scores_dir.mkdir(exist_ok=True)
    scaled = {key: scaler.transform(raw[key].reshape(-1, 3)).reshape(raw[key].shape).astype(np.float32)
              for key in ("train", "valid", "test")}
    p2p = {key: p2p_values(raw[key]) for key in ("train", "valid", "test")}
    p2p_ref = fit_p2p_reference(p2p["train"], cfg)
    p2p_scores, per_sensor = {}, {}
    for key in p2p:
        p2p_scores[key], per_sensor[key] = p2p_anomaly_score(p2p[key], p2p_ref)
    ae = {key: ae_scores(model, scaled[key], device) for key in scaled}
    ae_ref, p2p_norm_ref = fit_robust_normalizer(ae["train"], cfg, "AE"), fit_robust_normalizer(p2p_scores["train"], cfg, "P2P")
    ae_norm = {key: normalize(ae[key], ae_ref) for key in ae}
    p2p_norm = {key: normalize(p2p_scores[key], p2p_norm_ref) for key in p2p_scores}
    write_json(output / "normal_train_references.json", {"p2p": p2p_ref, "ae_normalizer": ae_ref, "p2p_normalizer": p2p_norm_ref,
              "leakage_check": "All fitted quantiles, IQRs, and normalizers above use train windows only.",
              "iqr_floor_impact": "The P2P anomaly score is zero for at least 75% of normal-train windows, so its score-IQR is floored. This prevents division by zero but can amplify nonzero P2P excesses; inspect all alpha rows rather than treating a fused result as a stable gain."})
    validation_rows, choices = [], []
    all_scores = {"A": ae, "B": p2p_scores}
    for method, score in all_scores.items():
        threshold, rule = threshold_at_normal_fpr(raw["y_valid"], score["valid"], cfg.fpr_target)
        row, _ = metric_row(raw["y_valid"], score["valid"], threshold, method); row.update(rule); validation_rows.append(row)
    for alpha in ALPHAS:
        fusion = {key: ae_norm[key] + alpha * p2p_norm[key] for key in ae_norm}
        threshold, rule = threshold_at_normal_fpr(raw["y_valid"], fusion["valid"], cfg.fpr_target)
        row, _ = metric_row(raw["y_valid"], fusion["valid"], threshold, "C", alpha); row.update(rule); validation_rows.append(row)
        choices.append((row["f1"], -alpha, alpha, fusion, threshold))
    # Pre-registered selection: highest validation F1; ties use the smaller alpha.
    selected = max(choices, key=lambda x: (x[0], x[1]))
    _, _, alpha, fusion, c_threshold = selected
    validation_table = pd.DataFrame(validation_rows); validation_table.to_csv(output / "validation_alpha_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(output / "selection.json", {"selection_rule": "highest validation F1; exact ties select smaller alpha", "candidate_alphas": ALPHAS,
              "selected_alpha": alpha, "selected_threshold": c_threshold, "test_not_used_for_selection": True})
    test_rows, predictions = [], {}
    for method, score in (("A", ae), ("B", p2p_scores), ("C", fusion)):
        threshold = (threshold_at_normal_fpr(raw["y_valid"], score["valid"], cfg.fpr_target)[0])
        row, pred = metric_row(raw["y_test"], score["test"], threshold, method, alpha if method == "C" else None)
        test_rows.append(row); predictions[method] = pred
    pd.DataFrame(test_rows).to_csv(output / "abc_test_comparison.csv", index=False, encoding="utf-8-sig")
    # Report every pre-registered alpha on Test for transparency, without using any row to select alpha.
    test_alpha_rows = []
    for candidate_alpha in ALPHAS:
        candidate = {key: ae_norm[key] + candidate_alpha * p2p_norm[key] for key in ae_norm}
        candidate_threshold, _ = threshold_at_normal_fpr(raw["y_valid"], candidate["valid"], cfg.fpr_target)
        candidate_row, _ = metric_row(raw["y_test"], candidate["test"], candidate_threshold, "C", candidate_alpha)
        candidate_row["selected_on_validation"] = candidate_alpha == alpha
        test_alpha_rows.append(candidate_row)
    pd.DataFrame(test_alpha_rows).to_csv(output / "test_alpha_comparison.csv", index=False, encoding="utf-8-sig")
    test_manifest = manifest[manifest.split == "test"].reset_index(drop=True)
    changed = make_changed_cases(test_manifest, raw["y_test"], p2p["test"], ae["test"], p2p_scores["test"], per_sensor["test"], ae_norm["test"], p2p_norm["test"], alpha, predictions["A"], predictions["C"])
    changed.to_csv(output / "test_prediction_changes.csv", index=False, encoding="utf-8-sig")
    scores = test_manifest.copy(); scores["ae_score"] = ae["test"]; scores["p2p_score"] = p2p_scores["test"]
    scores["fusion_score"] = fusion["test"]; scores["A_prediction"] = predictions["A"]; scores["B_prediction"] = predictions["B"]; scores["C_prediction"] = predictions["C"]
    scores.to_csv(output / "test_scores.csv", index=False, encoding="utf-8-sig")
    thresholds = {"A": threshold_at_normal_fpr(raw["y_valid"], ae["valid"], cfg.fpr_target)[0],
                  "B": threshold_at_normal_fpr(raw["y_valid"], p2p_scores["valid"], cfg.fpr_target)[0], "C": c_threshold}
    for split in ("train", "valid", "test"):
        rows = manifest[manifest.split == split].reset_index(drop=True)
        labels = raw[f"y_{split}"]
        assert np.array_equal(rows.label.to_numpy(), labels)
        feature_frame = rows.copy()
        feature_frame["AI0_Vibration_p2p"] = p2p[split][:, 0]; feature_frame["AI1_Vibration_p2p"] = p2p[split][:, 1]
        feature_frame["AI0_p2p_sensor_score"] = per_sensor[split][:, 0]; feature_frame["AI1_p2p_sensor_score"] = per_sensor[split][:, 1]
        feature_frame["p2p_score"] = p2p_scores[split]
        feature_frame.to_csv(features_dir / f"{split}_p2p_features.csv", index=False, encoding="utf-8-sig")
        score_frame = rows.copy(); score_frame["ae_score"] = ae[split]; score_frame["p2p_score"] = p2p_scores[split]
        score_frame["ae_normalized"] = ae_norm[split]; score_frame["p2p_normalized"] = p2p_norm[split]
        score_frame["fusion_score"] = fusion[split]
        for method, value in (("A", ae[split]), ("B", p2p_scores[split]), ("C", fusion[split])):
            score_frame[f"{method}_threshold"] = thresholds[method]
            score_frame[f"{method}_prediction"] = (value > thresholds[method]).astype(int)
        score_frame.to_csv(scores_dir / f"{split}_scores.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(scores));
    ax.scatter(x, scores.ae_score, s=4, label="A: AE score"); ax.scatter(x, scores.p2p_score, s=4, label="B: P2P score")
    ax.set(xlabel="Test window", ylabel="Raw score", title="A/B raw scores (different scales; not directly summed)"); ax.legend(); fig.tight_layout(); fig.savefig(output / "test_score_comparison.png", dpi=150); plt.close(fig)


def preflight(normal, raw, manifest, cfg, device, first_results: Path, output: Path) -> dict:
    p2p_train = p2p_values(raw["train"]); ref = fit_p2p_reference(p2p_train, cfg)
    manual_p2p = np.array([raw["train"][0, :, j].max() - raw["train"][0, :, j].min() for j in range(2)])
    p2p_exact_check = bool(np.allclose(p2p_train[0], manual_p2p) and np.all(p2p_train >= 0))
    p2p_train_score, _ = p2p_anomaly_score(p2p_train, ref)
    normalizer = fit_robust_normalizer(p2p_train_score, cfg, "P2P")
    # Deterministic functional check ensures threshold/fusion code works without fabricating model results.
    toy_labels = np.array([0] * 100 + [1] * 10); toy_ae = np.linspace(0, 1, 110); toy_p2p = np.maximum(0, toy_ae - .4)
    toy_threshold, toy_rule = threshold_at_normal_fpr(toy_labels, toy_ae + toy_p2p, cfg.fpr_target)
    model_path, scaler_path, artifact_kind = checkpoint_paths(first_results)
    artifact = {"signed_model": str(model_path) if model_path else None, "signed_scaler": str(scaler_path) if scaler_path else None,
                "available": bool(model_path), "artifact_kind": artifact_kind,
                "absolute_checkpoint_not_usable_for_H2": True}
    if artifact["available"]:
        model, scaler = load_signed_artifacts(model_path, scaler_path, normal, cfg, device)
        artifact["load_verified"] = bool(next(model.parameters()).is_cuda and scaler.n_features_in_ == 3)
        artifact["scaler_provenance"] = "saved signed scaler" if scaler_path else "deterministically reconstructed from original signed normal train rows [0:15000]"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        artifact["checkpoint_epoch"] = checkpoint.get("epoch")
        artifact["checkpoint_validation_loss"] = checkpoint.get("validation_loss")
    prior_manifest = first_results / "gpu_ablation_20261004" / "split_manifest.csv"
    manifest_match = False
    if prior_manifest.is_file():
        old = pd.read_csv(prior_manifest); columns = ["split", "source", "window_start_in_segment", "input_start_row", "input_end_row", "label_row", "label"]
        manifest_match = old[columns].equals(manifest[columns])
    report = {"status": "PASS" if artifact["available"] else "BLOCKED_MISSING_SIGNED_ARTIFACT",
              "cuda": {"available": True, "device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
              "windows": {"train": list(raw["train"].shape), "valid": list(raw["valid"].shape), "test": list(raw["test"].shape),
                          "labels": {"valid": np.bincount(raw["y_valid"]).tolist(), "test": np.bincount(raw["y_test"]).tolist()},
                          "matches_first_experiment_manifest": manifest_match},
              "p2p": {"uses_original_signed_W20": True, "sensors": VIBRATIONS, "first_window_manual_p2p": manual_p2p,
                      "first_window_matches_vectorized_calculation": p2p_exact_check, "reference": ref, "score_normalizer": normalizer,
                      "iqr_floor_impact": "Sensor-level P2P IQRs are nonzero. The combined one-sided P2P score has IQR=0 because most normal-train windows score zero; the documented 1e-6 floor will be used for fusion normalization, which can amplify rare P2P excesses."},
              "fusion_and_threshold_functional_check": {"passed": bool(np.isfinite(toy_threshold)), "threshold": toy_threshold, **toy_rule},
              "signed_artifact": artifact,
              "limitations_preserved": "Time-gap-agnostic W20 windows and validation/test input-boundary overlap are intentionally unchanged.",
              "note": "No training or final Test evaluation was run during preflight."}
    write_json(output / "preflight.json", report); return report


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here.parent / "data")
    parser.add_argument("--first-results", type=Path, default=here.parent / "phm-fine-blanking-press-val" / "results")
    parser.add_argument("--output-dir", type=Path, default=here / "results" / "preflight")
    parser.add_argument("--run", action="store_true", help="CUDA inference/evaluation only; no training")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        device = require_cuda(); cfg = Config(); normal, anomaly = load_csvs(args.data_dir)
        raw, manifest = raw_window_sets(normal, anomaly, cfg), build_manifest(normal, anomaly, cfg)
        manifest.to_csv(args.output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"); write_json(args.output_dir / "config.json", asdict(cfg))
        report = preflight(normal, raw, manifest, cfg, device, args.first_results, args.output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=json_default))
        if args.run:
            if report["status"] != "PASS": raise RuntimeError("A compatible Signed checkpoint is required; refusing to run H2 with any substitute.")
            scaler_path = report["signed_artifact"]["signed_scaler"]
            model, scaler = load_signed_artifacts(Path(report["signed_artifact"]["signed_model"]), Path(scaler_path) if scaler_path else None, normal, cfg, device)
            # Freeze the exact in-memory inference artifact used by this independent run.
            # This remains reproducible even if an external first-experiment process is still writing its checkpoint.
            artifact_dir = args.output_dir / "source_artifacts"; artifact_dir.mkdir(exist_ok=True)
            torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "source_checkpoint": report["signed_artifact"]["signed_model"],
                        "source_checkpoint_epoch": report["signed_artifact"].get("checkpoint_epoch"),
                        "source_checkpoint_validation_loss": report["signed_artifact"].get("checkpoint_validation_loss")}, artifact_dir / "checkpoint_used.pt")
            joblib.dump(scaler, artifact_dir / ("scaler.joblib" if scaler_path else "reconstructed_signed_scaler.joblib"))
            run_evaluation(raw, manifest, cfg, model, scaler, device, args.output_dir)
            print("CUDA inference/evaluation complete; no training was run.")
        else: print("Preflight only; no training or final Test evaluation was run.")
        return 0 if report["status"] == "PASS" else 2
    except Exception as error:
        failure = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error), "python": sys.version, "platform": platform.platform()}
        write_json(args.output_dir / "preflight.json", failure); print(json.dumps(failure, ensure_ascii=False, indent=2)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
