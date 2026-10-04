"""Minimal absolute-value versus signed-input LSTM-AE ablation.

Default mode is deliberately preflight-only.  Training starts only with --run.
The split protocol intentionally reproduces pytorch_reproduction.ipynb, including
its time-gap-agnostic windows and overlap at the validation/test input boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
EXPECTED_COLUMNS = ["TimeStamp", *FEATURES, "Equipment_state"]


@dataclass(frozen=True)
class Config:
    sequence: int = 20
    label_offset: int = 100
    train_rows: int = 15000
    valid_normal: int = 880
    valid_anomaly: int = 300
    epochs: int = 800
    batch_size: int = 128
    learning_rate: float = 0.001
    lr_factor: float = 0.7
    lr_patience: int = 50
    es_min_delta: float = 0.00001
    es_patience: int = 120
    seed: int = 42


class ThresholdRuleNotFound(ValueError):
    """The notebook's exact precision == recall rule did not yield a threshold."""


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Kept as in the source notebook: deterministic algorithms are not enabled.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. CPU fallback is intentionally disabled.")
    return torch.device("cuda:0")


def load_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal_path, anomaly_path = data_dir / "press_data_normal.csv", data_dir / "outlier_data.csv"
    if not normal_path.is_file() or not anomaly_path.is_file():
        raise FileNotFoundError(f"Both CSV files are required in {data_dir}")
    normal = pd.read_csv(normal_path, index_col=0)
    anomaly = pd.read_csv(anomaly_path, index_col=0)
    for name, frame, expected_rows, label in (("normal", normal, 20000, 0), ("outlier", anomaly, 600, 1)):
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"{name} columns differ: {list(frame.columns)}")
        if len(frame) != expected_rows or not frame.Equipment_state.eq(label).all():
            raise ValueError(f"{name} row count or Equipment_state labels differ from the notebook protocol")
        if frame.isna().any().any() or not np.isfinite(frame[FEATURES].to_numpy()).all():
            raise ValueError(f"{name} has missing or non-finite sensor values")
    return normal, anomaly


def make_windows(values: np.ndarray, labels: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, window_labels, starts = [], [], []
    for i in range(len(values) - cfg.sequence - cfg.label_offset):
        windows.append(values[i:i + cfg.sequence])
        window_labels.append(labels[i + cfg.sequence + cfg.label_offset])
        starts.append(i)
    return np.asarray(windows, dtype=np.float32), np.asarray(window_labels, dtype=np.int64), np.asarray(starts, dtype=np.int64)


def build_manifest(normal: pd.DataFrame, anomaly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    specs = (
        ("train", "normal", normal, 0, cfg.train_rows, 0, cfg.train_rows - cfg.sequence - cfg.label_offset),
        ("valid_test", "normal", normal, cfg.train_rows, len(normal), cfg.valid_normal,
         len(normal) - cfg.train_rows - cfg.sequence - cfg.label_offset),
        ("valid_test", "anomaly", anomaly, 0, len(anomaly), cfg.valid_anomaly,
         len(anomaly) - cfg.sequence - cfg.label_offset),
    )
    for group, source, frame, segment_start, _, validation_count, count in specs:
        for i in range(count):
            split = "train" if group == "train" else ("valid" if i < validation_count else "test")
            first, last = segment_start + i, segment_start + i + cfg.sequence - 1
            label_row = segment_start + i + cfg.sequence + cfg.label_offset
            rows.append({"split": split, "source": source, "window_start_in_segment": i,
                         "input_start_row": first, "input_end_row": last, "label_row": label_row,
                         "input_end_time": frame.iloc[last].TimeStamp, "label_time": frame.iloc[label_row].TimeStamp,
                         "label": int(frame.iloc[label_row].Equipment_state)})
    return pd.DataFrame(rows)


def transformed_frame(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    result = frame.copy()
    if mode == "absolute":
        result.loc[:, FEATURES] = result[FEATURES].abs()
    elif mode != "signed":
        raise ValueError(mode)
    return result


def prepare_variant(normal: pd.DataFrame, anomaly: pd.DataFrame, cfg: Config, mode: str) -> dict:
    normal_t, anomaly_t = transformed_frame(normal, mode), transformed_frame(anomaly, mode)
    # Fit only on the transformed normal train partition; transforms may exceed [0, 1].
    train_input = normal_t.loc[:cfg.train_rows - 1, FEATURES].to_numpy()
    scaler = MinMaxScaler().fit(train_input)
    train = scaler.transform(train_input)
    normal_test = scaler.transform(normal_t.loc[cfg.train_rows:, FEATURES].to_numpy())
    anomaly_test = scaler.transform(anomaly_t[FEATURES].to_numpy())
    x_train, y_train, train_starts = make_windows(train, normal_t.Equipment_state.iloc[:cfg.train_rows].to_numpy(), cfg)
    xn, yn, normal_starts = make_windows(normal_test, normal_t.Equipment_state.iloc[cfg.train_rows:].to_numpy(), cfg)
    xa, ya, anomaly_starts = make_windows(anomaly_test, anomaly_t.Equipment_state.to_numpy(), cfg)
    return {"mode": mode, "X_train": x_train, "Y_train": y_train,
            "X_valid": np.vstack((xn[:cfg.valid_normal], xa[:cfg.valid_anomaly])),
            "Y_valid": np.hstack((yn[:cfg.valid_normal], ya[:cfg.valid_anomaly])),
            "X_test": np.vstack((xn[cfg.valid_normal:], xa[cfg.valid_anomaly:])),
            "Y_test": np.hstack((yn[cfg.valid_normal:], ya[cfg.valid_anomaly:])),
            "X_valid_0": xn[:cfg.valid_normal], "scaler": scaler,
            "window_starts": {"train": train_starts, "normal": normal_starts, "anomaly": anomaly_starts},
            "transformed_normal": normal_t, "transformed_anomaly": anomaly_t, "scaler_fit_values": train_input}


class LSTMAutoencoder(nn.Module):
    """Exact architecture and initialization used by the reproduction notebook."""
    def __init__(self, sequence: int):
        super().__init__()
        self.sequence = sequence
        self.encoder1 = nn.LSTM(3, 64, batch_first=True)
        self.encoder2 = nn.LSTM(64, 32, batch_first=True)
        self.decoder1 = nn.LSTM(32, 32, batch_first=True)
        self.decoder2 = nn.LSTM(32, 64, batch_first=True)
        self.output = nn.Linear(64, 3)
        for layer in (self.encoder1, self.encoder2, self.decoder1, self.decoder2):
            nn.init.xavier_uniform_(layer.weight_ih_l0)
            nn.init.orthogonal_(layer.weight_hh_l0)
            nn.init.zeros_(layer.bias_ih_l0)
            with torch.no_grad():
                layer.bias_ih_l0[layer.hidden_size:2 * layer.hidden_size].fill_(1)
            nn.init.zeros_(layer.bias_hh_l0)
            layer.bias_hh_l0.requires_grad_(False)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.encoder1(x)
        _, (hidden, _) = self.encoder2(x)
        x = hidden[-1].unsqueeze(1).repeat(1, self.sequence, 1)
        x, _ = self.decoder1(x)
        x, _ = self.decoder2(x)
        return self.output(x)


def make_seeded_model(cfg: Config, device: torch.device) -> LSTMAutoencoder:
    # fork_rng prevents construction of A from advancing B's initial RNG state.
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        return LSTMAutoencoder(cfg.sequence).to(device)


class KerasAdam(torch.optim.Optimizer):
    """Notebook's Keras-Adam-compatible implementation (epsilon=1e-7)."""
    def __init__(self, params, lr: float = .001):
        super().__init__(params, dict(lr=lr, beta1=.9, beta2=.999, eps=1e-7, step=0))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            group["step"] += 1
            anchor = group["params"][0]
            b1, b2 = [torch.tensor(group[k], dtype=anchor.dtype, device=anchor.device) for k in ("beta1", "beta2")]
            alpha = torch.tensor(group["lr"], dtype=anchor.dtype, device=anchor.device) * (1 - b2.pow(group["step"])).sqrt() / (1 - b1.pow(group["step"]))
            for param in group["params"]:
                if param.grad is None:
                    continue
                if not self.state[param]:
                    self.state[param]["m"], self.state[param]["v"] = torch.zeros_like(param), torch.zeros_like(param)
                m, v = self.state[param]["m"], self.state[param]["v"]
                m.add_((param.grad - m) * (1 - b1))
                v.add_((param.grad.square() - v) * (1 - b2))
                param.add_(-alpha * m / (v.sqrt() + group["eps"]))


def state_dicts_equal(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.equal(a.state_dict()[key].cpu(), b.state_dict()[key].cpu()) for key in a.state_dict())


def preflight(normal: pd.DataFrame, anomaly: pd.DataFrame, cfg: Config, device: torch.device, output: Path) -> dict:
    absolute, signed = (prepare_variant(normal, anomaly, cfg, mode) for mode in ("absolute", "signed"))
    expected_shapes = {"X_train": (14880, 20, 3), "X_valid": (1180, 20, 3), "X_test": (4180, 20, 3), "X_valid_0": (880, 20, 3)}
    for item in (absolute, signed):
        for key, shape in expected_shapes.items():
            assert item[key].shape == shape, (item["mode"], key, item[key].shape)
    same_indices = all(np.array_equal(absolute["window_starts"][key], signed["window_starts"][key])
                       for key in absolute["window_starts"])
    same_labels = all(np.array_equal(absolute[key], signed[key]) for key in ("Y_train", "Y_valid", "Y_test"))
    abs_is_absolute = all(np.array_equal(absolute[f"transformed_{name}"][FEATURES].to_numpy(), frame[FEATURES].abs().to_numpy())
                          for name, frame in (("normal", normal), ("anomaly", anomaly)))
    signed_is_raw = all(np.array_equal(signed[f"transformed_{name}"][FEATURES].to_numpy(), frame[FEATURES].to_numpy())
                        for name, frame in (("normal", normal), ("anomaly", anomaly)))
    scaler_ok = {}
    for item in (absolute, signed):
        values, scaler = item["scaler_fit_values"], item["scaler"]
        scaler_ok[item["mode"]] = bool(np.allclose(scaler.data_min_, values.min(axis=0)) and
                                         np.allclose(scaler.data_max_, values.max(axis=0)) and
                                         scaler.n_samples_seen_ == cfg.train_rows)
    model_a, model_b = make_seeded_model(cfg, device), make_seeded_model(cfg, device)
    parameter_counts = [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in (model_a, model_b)]
    input_batch = torch.tensor(absolute["X_train"][:2], dtype=torch.float32, device=device)
    model_a.train(); reconstruction = model_a(input_batch); loss = nn.MSELoss()(reconstruction, input_batch)
    loss.backward()
    cuda_forward_backward = reconstruction.shape == input_batch.shape and all(
        p.grad is not None for p in model_a.parameters() if p.requires_grad)
    shuffle_a = torch.randperm(len(absolute["X_train"]), generator=torch.Generator().manual_seed(cfg.seed))
    shuffle_b = torch.randperm(len(signed["X_train"]), generator=torch.Generator().manual_seed(cfg.seed))
    report = {"status": "PASS", "cuda": {"available": True, "device": str(device), "gpu": torch.cuda.get_device_name(device),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
              "csv": {"normal_rows": len(normal), "outlier_rows": len(anomaly), "columns": EXPECTED_COLUMNS},
              "splits": {key: list(value) for key, value in expected_shapes.items()},
              "labels": {"train": {"0": int((absolute["Y_train"] == 0).sum())},
                         "valid": {str(k): int(v) for k, v in zip(*np.unique(absolute["Y_valid"], return_counts=True))},
                         "test": {str(k): int(v) for k, v in zip(*np.unique(absolute["Y_test"], return_counts=True))}},
              "same_raw_rows_and_window_indices": same_indices, "same_labels": same_labels,
              "preprocessing": {"absolute_is_abs": abs_is_absolute, "signed_is_raw": signed_is_raw, "scaler_fit_only_normal_train": scaler_ok},
              "model": {"same_structure_and_trainable_parameters": parameter_counts[0] == parameter_counts[1],
                        "trainable_parameter_counts": parameter_counts, "same_initial_weights": state_dicts_equal(model_a, model_b)},
              "same_shuffle_order": bool(torch.equal(shuffle_a, shuffle_b)), "cuda_forward_backward": bool(cuda_forward_backward),
              "note": "No model training was run during preflight."}
    checks = [same_indices, same_labels, abs_is_absolute, signed_is_raw, all(scaler_ok.values()),
              parameter_counts[0] == parameter_counts[1], report["model"]["same_initial_weights"],
              report["same_shuffle_order"], cuda_forward_backward]
    if not all(checks):
        report["status"] = "FAIL"
    write_json(output / "preflight.json", report)
    pd.DataFrame(build_manifest(normal, anomaly, cfg)).to_csv(output / "split_manifest.csv", index=False, encoding="utf-8-sig")
    return report


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval(); chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            chunks.append(model(torch.tensor(x[start:start + 32], dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(chunks)


def last_step_mse(x: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    return np.mean(np.power(x[:, -1, :] - reconstruction[:, -1, :], 2), axis=1)


def select_threshold(labels: np.ndarray, scores: np.ndarray):
    precision, recall, thresholds = metrics.precision_recall_curve(list(labels), scores)
    matches = [i for i, (p, r) in enumerate(zip(precision[:-1], recall[:-1])) if p == r]
    if not matches:
        raise ThresholdRuleNotFound("The source notebook's exact precision == recall rule found no threshold.")
    i = matches[0]
    return float(thresholds[i]), i, precision, recall, thresholds


def train_variant(data: dict, cfg: Config, device: torch.device, output: Path) -> dict:
    model = make_seeded_model(cfg, device)
    optimizer = KerasAdam([p for p in model.parameters() if p.requires_grad], lr=float(np.float32(cfg.learning_rate)))
    loader = DataLoader(TensorDataset(torch.tensor(data["X_train"], dtype=torch.float32)), batch_size=cfg.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), num_workers=0)
    valid_loader = DataLoader(TensorDataset(torch.tensor(data["X_valid_0"], dtype=torch.float32)), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    loss_fn, best_loss, lr_best, early_wait, lr_wait = nn.MSELoss(), float("inf"), float("inf"), 0, 0
    history, started, best_weights = [], time.perf_counter(), None
    for epoch in range(cfg.epochs):
        epoch_start, current_lr, total = time.perf_counter(), optimizer.param_groups[0]["lr"], 0.0
        model.train()
        for (x,) in loader:
            x = x.to(device); optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(x), x); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(x)
        train_loss = total / len(data["X_train"]); total = 0.0; model.eval()
        with torch.inference_mode():
            for (x,) in valid_loader:
                x = x.to(device); total += float(loss_fn(model(x), x)) * len(x)
        valid_loss = total / len(data["X_valid_0"])
        if valid_loss < lr_best - 1e-4: lr_best, lr_wait = valid_loss, 0
        else:
            lr_wait += 1
            if lr_wait >= cfg.lr_patience:
                optimizer.param_groups[0]["lr"] = float(np.float32(optimizer.param_groups[0]["lr"] * cfg.lr_factor)); lr_wait = 0
        early_wait += 1
        if valid_loss + cfg.es_min_delta < best_loss:
            best_loss, early_wait = valid_loss, 0; best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_weights, "config": asdict(cfg), "mode": data["mode"],
                        "epoch": epoch, "validation_loss": valid_loss}, output / "checkpoint_best.pt")
        history.append({"epoch": epoch, "loss": train_loss, "val_loss": valid_loss, "lr": current_lr, "epoch_seconds": time.perf_counter() - epoch_start})
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        write_json(output / "training_progress.json", {"last_completed_epoch": epoch + 1, "best_validation_loss": best_loss,
                   "early_stop_wait": early_wait, "lr_wait": lr_wait, "learning_rate": optimizer.param_groups[0]["lr"]})
        logging.info("%s epoch=%d train_loss=%.8f valid_loss=%.8f lr=%.8g", data["mode"], epoch + 1, train_loss, valid_loss, current_lr)
        if early_wait >= cfg.es_patience and epoch > 0:
            model.load_state_dict(best_weights); break
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "mode": data["mode"]}, output / "model.pt")
    joblib.dump(data["scaler"], output / "scaler.joblib")
    summary = {"seconds": time.perf_counter() - started, "epochs_run": len(history), "epoch_limit": cfg.epochs,
               "best_val_loss": min(x["val_loss"] for x in history)}
    write_json(output / "training.json", summary)
    return {"model": model, **summary}


def evaluate_variant(data: dict, model: nn.Module, device: torch.device, output: Path) -> dict:
    valid_reconstruction = predict(model, data["X_valid"], device)
    test_reconstruction = predict(model, data["X_test"], device)
    valid_scores = last_step_mse(data["X_valid"], valid_reconstruction)
    test_scores = last_step_mse(data["X_test"], test_reconstruction)
    np.savez_compressed(output / "reconstructions.npz", valid_input=data["X_valid"], valid_reconstruction=valid_reconstruction,
                        test_input=data["X_test"], test_reconstruction=test_reconstruction)
    valid_sensor_mse = np.square(data["X_valid"][:, -1, :] - valid_reconstruction[:, -1, :])
    test_sensor_mse = np.square(data["X_test"][:, -1, :] - test_reconstruction[:, -1, :])
    try:
        threshold, index, precision, recall, curve_thresholds = select_threshold(data["Y_valid"], valid_scores)
    except ThresholdRuleNotFound as error:
        result = {"threshold_available": False, "threshold_error": str(error)}
        pd.DataFrame({"label": data["Y_valid"], "anomaly_score": valid_scores, **{f"{name}_mse": valid_sensor_mse[:, i] for i, name in enumerate(FEATURES)}}).to_csv(output / "valid_predictions.csv", index=False)
        pd.DataFrame({"label": data["Y_test"], "anomaly_score": test_scores, **{f"{name}_mse": test_sensor_mse[:, i] for i, name in enumerate(FEATURES)}}).to_csv(output / "test_predictions.csv", index=False)
        write_json(output / "metrics.json", result); return result
    predicted = (test_scores > threshold).astype(int)
    cm = metrics.confusion_matrix(data["Y_test"], predicted, labels=[0, 1])
    result = {"threshold_available": True, "threshold": threshold, "validation_curve_precision": float(precision[index]),
              "validation_curve_recall": float(recall[index]), "f1": metrics.f1_score(data["Y_test"], predicted, zero_division=0),
              "precision": metrics.precision_score(data["Y_test"], predicted, zero_division=0), "recall": metrics.recall_score(data["Y_test"], predicted, zero_division=0),
              "normal_false_positive_rate": float(cm[0, 1] / cm[0].sum()), "confusion_matrix": cm.tolist()}
    write_json(output / "metrics.json", result)
    pd.DataFrame({"label": data["Y_valid"], "anomaly_score": valid_scores, "predicted_label": (valid_scores > threshold).astype(int), **{f"{name}_mse": valid_sensor_mse[:, i] for i, name in enumerate(FEATURES)}}).to_csv(output / "valid_predictions.csv", index=False)
    pd.DataFrame({"label": data["Y_test"], "anomaly_score": test_scores, "predicted_label": predicted, **{f"{name}_mse": test_sensor_mse[:, i] for i, name in enumerate(FEATURES)}}).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(output / "confusion_matrix.csv")
    pd.DataFrame({"threshold": curve_thresholds, "precision": precision[:-1], "recall": recall[:-1]}).to_csv(output / "threshold_curve.csv", index=False)
    return result


def run_experiment(normal, anomaly, cfg, device, root: Path) -> None:
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device), "device": str(device)}
    records = []
    for mode in ("absolute", "signed"):
        output = root / mode; output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", asdict(cfg))
        write_json(output / "environment.json", environment)
        data = prepare_variant(normal, anomaly, cfg, mode); trained = train_variant(data, cfg, device, output)
        result = evaluate_variant(data, trained["model"], device, output)
        result.update({"mode": mode, "training_seconds": trained["seconds"], "epochs_run": trained["epochs_run"], "best_val_loss": trained["best_val_loss"]})
        write_json(output / "result.json", result); records.append(result)
    comparison = pd.DataFrame(records).set_index("mode")
    if all(comparison.threshold_available):
        for metric in ("f1", "precision", "recall", "normal_false_positive_rate"):
            comparison.loc["signed_minus_absolute", metric] = comparison.loc["signed", metric] - comparison.loc["absolute", metric]
    comparison.to_csv(root / "ablation_comparison.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--run", action="store_true", help="After successful preflight, train both variants on CUDA.")
    args = parser.parse_args(); cfg = Config(); args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.output_dir / "run.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_json(args.output_dir / "config.json", asdict(cfg))
    try:
        device = require_cuda(); configure_seed(cfg.seed); normal, anomaly = load_csvs(args.data_dir)
        report = preflight(normal, anomaly, cfg, device, args.output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=json_default)); logging.info("Preflight: %s", report["status"])
        if report["status"] != "PASS": return 2
        if args.run:
            run_experiment(normal, anomaly, cfg, device, args.output_dir)
            print(f"Training complete: {args.output_dir}")
        else:
            print("Preflight only; no training was run. Add --run only after approval.")
    except Exception as error:
        failure = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error), "python": sys.version,
                   "platform": platform.platform(), "torch_version": torch.__version__}
        write_json(args.output_dir / "preflight.json", failure); logging.exception("Stopped")
        print(json.dumps(failure, ensure_ascii=False, indent=2)); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
