"""Frozen configuration shared by the sensor-fusion experiment and its smoke test.

This module intentionally contains no data loading or model code.  Its purpose is
to make the baseline-equivalent settings and repository-relative paths explicit.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union


FEATURES = ("AI0_Vibration", "AI1_Vibration", "AI2_Current")
VIBRATION_FEATURES = FEATURES[:2]
CURRENT_FEATURES = FEATURES[2:]
NORMAL_FILENAME = "press_data_normal.csv"
OUTLIER_FILENAME = "outlier_data.csv"


@dataclass(frozen=True)
class ExperimentConfig:
    """Values copied from the reference notebook; none are tuning parameters here."""

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


def config_dict() -> dict:
    """Return the frozen settings in a JSON-friendly form for run metadata."""
    return asdict(ExperimentConfig())


def resolve_project_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Find the repository root from a notebook or command-line working directory.

    A valid root contains both ``data/`` and ``experiment/sensor_fusion/``.  ``start`` exists
    for a deliberate user override and for deterministic tests.
    """
    candidate = Path.cwd() if start is None else Path(start)
    candidate = candidate.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "data").is_dir() and (directory / "experiment" / "sensor_fusion").is_dir():
            return directory
    raise FileNotFoundError(
        "Could not find the project root containing both 'data/' and 'experiment/sensor_fusion/'. "
        "Start the notebook inside the cloned repository or pass an explicit start path."
    )


def resolve_data_paths(start: Optional[Union[str, Path]] = None) -> tuple[Path, Path]:
    """Return and validate the repository-relative normal/outlier CSV paths."""
    data_dir = resolve_project_root(start) / "data"
    normal_path = data_dir / NORMAL_FILENAME
    outlier_path = data_dir / OUTLIER_FILENAME
    missing = [str(path) for path in (normal_path, outlier_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required experiment data file(s) not found: " + ", ".join(missing))
    return normal_path, outlier_path
