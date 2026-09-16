"""Shared paths and writers for the preprocessing stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"
RAW_ROOT = DATA_ROOT / "Raw"
INTERIM_ROOT = DATA_ROOT / "Interim"
PROCESSED_ROOT = DATA_ROOT / "Processed"
METADATA_ROOT = DATA_ROOT / "Metadata"
FIGURE_ROOT = PROJECT_ROOT / "Outputs" / "Figures" / "0-DataPreprocess"


def write_audit(path: Path, values: dict) -> None:
    """Write a preprocessing audit as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def aggregate_events(
    events: pd.DataFrame,
    output_path: Path,
    observed_start_s: float,
    observed_end_s: float,
    window_seconds: int = 3600,
) -> pd.DataFrame:
    """Aggregate event rows into a complete per-type count grid."""
    first_window = int(observed_start_s // window_seconds)
    last_window = int(observed_end_s // window_seconds)
    groups = sorted(events["task_type"].dropna().unique())
    counts = (
        events.assign(window=(events.arrival_time_s // window_seconds).astype("int64"))
        .groupby(["task_type", "window"])
        .size()
    )
    index = pd.MultiIndex.from_product(
        [groups, range(first_window, last_window + 1)],
        names=["task_type", "window"],
    )
    result = counts.reindex(index, fill_value=0).rename("arrival_count").reset_index()
    result["window_start_s"] = result.pop("window") * window_seconds
    result["window_seconds"] = window_seconds
    result["coverage"] = np.where(
        (result.window_start_s < observed_start_s)
        | (result.window_start_s + window_seconds > observed_end_s),
        "partial_boundary",
        "within_recorded_span_not_independently_verified",
    )
    if int(result.arrival_count.sum()) != len(events):
        raise ValueError("Aggregated count does not equal retained event rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result
