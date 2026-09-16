"""Fit simple Alibaba arrival baselines and evaluate rolling forecasts."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT / "Data" / "Processed" / "alibaba_gpu_2020" / "arrivals_hourly.csv"
)
TABLE_ROOT = PROJECT_ROOT / "Outputs" / "Tables"
FIGURE_ROOT = PROJECT_ROOT / "Outputs" / "Figures" / "1-FitForecast"

TASK_TYPES = ("bert", "ctr", "nmt")
WINDOW_SECONDS = 3600
INITIAL_TRAIN_FRACTION = 0.5
ROLLING_WINDOW_HOURS = 24
EWMA_ALPHA = 0.2
ZOOM_HOURS = 72
REQUIRED_COLUMNS = {"task_type", "window_start_s", "arrival_count"}

PREDICTION_COLUMNS = {
    "same_hour_mean": "same_hour_mean_prediction",
    "poisson_rate": "poisson_rate_prediction",
    "rolling_poisson_24h": "rolling_poisson_24h_prediction",
    "ewma_0_2": "ewma_0_2_prediction",
    "seasonal_naive_24h": "seasonal_naive_24h_prediction",
}

NEW_PLOT_METHODS = (
    ("rolling_poisson_24h_prediction", "24 小时滑动泊松"),
    ("ewma_0_2_prediction", "指数平滑（α=0.2）"),
    ("seasonal_naive_24h_prediction", "前一天同小时"),
)


def validate_task_series(data: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Validate and sort one complete hourly count series."""
    series = data.loc[
        data["task_type"].eq(task_type),
        ["task_type", "window_start_s", "arrival_count"],
    ].copy()
    if series.empty:
        raise ValueError(f"Missing task type: {task_type}")

    for column in ("window_start_s", "arrival_count"):
        series[column] = pd.to_numeric(series[column], errors="coerce")
        if series[column].isna().any():
            raise ValueError(f"{task_type}: {column} contains missing or nonnumeric values")

    if not np.equal(series["window_start_s"], np.floor(series["window_start_s"])).all():
        raise ValueError(f"{task_type}: window_start_s must contain integer seconds")
    if not np.equal(series["arrival_count"], np.floor(series["arrival_count"])).all():
        raise ValueError(f"{task_type}: arrival_count must contain integer counts")
    if series["arrival_count"].lt(0).any():
        raise ValueError(f"{task_type}: arrival_count contains negative values")
    if series["window_start_s"].duplicated().any():
        raise ValueError(f"{task_type}: duplicate hourly windows found")

    series["window_start_s"] = series["window_start_s"].astype("int64")
    series["arrival_count"] = series["arrival_count"].astype("int64")
    series = series.sort_values("window_start_s").reset_index(drop=True)

    gaps = series["window_start_s"].diff().dropna()
    invalid = gaps.ne(WINDOW_SECONDS)
    if invalid.any():
        first_bad_index = invalid[invalid].index[0]
        previous = int(series.loc[first_bad_index - 1, "window_start_s"])
        current = int(series.loc[first_bad_index, "window_start_s"])
        raise ValueError(
            f"{task_type}: hourly sequence is not continuous between "
            f"{previous} and {current}"
        )

    initial_hours = int(len(series) * INITIAL_TRAIN_FRACTION)
    if initial_hours < ROLLING_WINDOW_HOURS or initial_hours == len(series):
        raise ValueError(
            f"{task_type}: at least {ROLLING_WINDOW_HOURS} initial hours "
            "and one evaluation hour are required"
        )
    return series


def load_input(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the processed Alibaba table and retain the three target types."""
    data = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing input columns: {', '.join(missing_columns)}")
    return pd.concat(
        [validate_task_series(data, task_type) for task_type in TASK_TYPES],
        ignore_index=True,
    )


def rolling_forecast(series: pd.DataFrame, task_type: str) -> tuple[pd.DataFrame, pd.Series]:
    """Predict each evaluation hour before adding its observed count to history."""
    series = validate_task_series(series, task_type)
    split_index = int(len(series) * INITIAL_TRAIN_FRACTION)
    history = series.iloc[:split_index]

    history_hours = (
        history["window_start_s"].to_numpy(dtype="int64") // WINDOW_SECONDS
    ) % 24
    history_values = history["arrival_count"].to_numpy(dtype="float64")
    hourly_sums = np.bincount(history_hours, weights=history_values, minlength=24)
    hourly_counts = np.bincount(history_hours, minlength=24)
    if (hourly_counts == 0).any():
        raise ValueError(f"{task_type}: initial history does not cover all 24 daily hours")

    total_sum = float(history_values.sum())
    total_count = len(history_values)
    recent_values = deque(
        history_values[-ROLLING_WINDOW_HOURS:].tolist(), maxlen=ROLLING_WINDOW_HOURS
    )
    recent_sum = float(sum(recent_values))
    ewma = float(history_values[0])
    for value in history_values[1:]:
        ewma = EWMA_ALPHA * value + (1 - EWMA_ALPHA) * ewma
    records: list[dict[str, float | int | str]] = []

    for row in series.iloc[split_index:].itertuples(index=False):
        hour = (int(row.window_start_s) // WINDOW_SECONDS) % 24
        actual = int(row.arrival_count)
        records.append(
            {
                "task_type": task_type,
                "window_start_s": int(row.window_start_s),
                "actual_count": actual,
                "same_hour_mean_prediction": float(
                    hourly_sums[hour] / hourly_counts[hour]
                ),
                "poisson_rate_prediction": float(total_sum / total_count),
                "rolling_poisson_24h_prediction": float(
                    recent_sum / ROLLING_WINDOW_HOURS
                ),
                "ewma_0_2_prediction": ewma,
                "seasonal_naive_24h_prediction": float(recent_values[0]),
            }
        )

        # Update only after forecasting, so the current observation cannot leak.
        recent_sum += actual - recent_values[0]
        recent_values.append(actual)
        ewma = EWMA_ALPHA * actual + (1 - EWMA_ALPHA) * ewma
        hourly_sums[hour] += actual
        hourly_counts[hour] += 1
        total_sum += actual
        total_count += 1

    return pd.DataFrame.from_records(records), history["arrival_count"]


def error_metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    """Return point-forecast errors using prediction minus observation."""
    error = prediction.to_numpy(dtype="float64") - actual.to_numpy(dtype="float64")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_error": float(np.mean(error)),
    }


def summarize_metrics(
    predictions: pd.DataFrame, initial_history: pd.Series, task_type: str
) -> pd.DataFrame:
    """Summarize forecast errors and an initial-history Poisson diagnostic."""
    training_mean = float(initial_history.mean())
    training_variance = float(initial_history.var(ddof=1))
    variance_mean_ratio = (
        training_variance / training_mean if training_mean > 0 else np.nan
    )
    rows = []
    for method, prediction_column in PREDICTION_COLUMNS.items():
        rows.append(
            {
                "task_type": task_type,
                "method": method,
                "initial_history_hours": len(initial_history),
                "evaluation_hours": len(predictions),
                **error_metrics(
                    predictions["actual_count"], predictions[prediction_column]
                ),
                "training_mean": training_mean,
                "training_variance": training_variance,
                "training_variance_mean_ratio": variance_mean_ratio,
            }
        )
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    """Use a Chinese-capable font when available."""
    windows_font = Path("C:/Windows/Fonts/msyh.ttc")
    if windows_font.exists():
        font_manager.fontManager.addfont(str(windows_font))
        family = "Microsoft YaHei"
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    methods: tuple[tuple[str, str], ...] = (
        ("same_hour_mean_prediction", "同一时刻均值"),
        ("poisson_rate_prediction", "固定率泊松"),
    ),
    title: str = "Alibaba 三类作业：逐小时滚动预测",
) -> None:
    """Plot observed counts against selected one-hour-ahead forecasts."""
    configure_plotting()
    figure, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
    for axis, task_type in zip(axes, TASK_TYPES):
        subset = predictions[predictions["task_type"].eq(task_type)]
        relative_days = (
            subset["window_start_s"] - subset["window_start_s"].iloc[0]
        ) / 86400
        axis.plot(
            relative_days,
            subset["actual_count"],
            color="#333333",
            linewidth=0.8,
            label="真实值",
        )
        for column, label in methods:
            axis.plot(relative_days, subset[column], linewidth=1.0, label=label)
        axis.set_title(task_type, loc="left", fontweight="bold")
        axis.set_xlabel("评估阶段相对时间（天）")
        axis.set_ylabel("数量 / 小时")
        axis.grid(alpha=0.18)
        axis.legend(loc="upper right")

    figure.suptitle(title, fontsize=18, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_zoomed_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    task_type: str = "nmt",
    hours: int = ZOOM_HOURS,
) -> None:
    """Plot a readable close-up of the final evaluation hours."""
    subset = predictions[predictions["task_type"].eq(task_type)].tail(hours)
    if len(subset) < hours:
        raise ValueError(f"{task_type}: fewer than {hours} evaluation hours")

    configure_plotting()
    relative_hours = np.arange(-hours + 1, 1)
    figure, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
    axis.plot(
        relative_hours,
        subset["actual_count"],
        color="#222222",
        linewidth=1.6,
        marker="o",
        markersize=3,
        label="真实值",
        zorder=4,
    )
    for column, label in NEW_PLOT_METHODS:
        axis.plot(relative_hours, subset[column], linewidth=1.2, label=label)
    axis.set_title(
        f"Alibaba {task_type}：评估末 {hours} 小时局部放大",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_xlabel("距评估结束（小时）")
    axis.set_ylabel("数量 / 小时")
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left", ncols=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    data = load_input()
    prediction_parts = []
    metric_parts = []
    for task_type in TASK_TYPES:
        predictions, history = rolling_forecast(data, task_type)
        prediction_parts.append(predictions)
        metric_parts.append(summarize_metrics(predictions, history, task_type))

    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    all_metrics = pd.concat(metric_parts, ignore_index=True)

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    figure_stamp = datetime.now().strftime("%m%d_%H%S")
    predictions_path = TABLE_ROOT / "alibaba_rolling_predictions.csv"
    metrics_path = TABLE_ROOT / "alibaba_rolling_metrics.csv"
    figure_path = FIGURE_ROOT / f"Alibaba_滚动预测对比_{figure_stamp}.png"
    new_figure_path = FIGURE_ROOT / f"Alibaba_基础模型对比_{figure_stamp}.png"
    single_method_figures = (
        (
            NEW_PLOT_METHODS[0],
            FIGURE_ROOT
            / f"Alibaba_24小时滑动泊松_真实值对比_{figure_stamp}.png",
            "Alibaba 三类作业：24 小时滑动泊松与真实值",
        ),
        (
            NEW_PLOT_METHODS[1],
            FIGURE_ROOT / f"Alibaba_EWMA_真实值对比_{figure_stamp}.png",
            "Alibaba 三类作业：指数平滑与真实值",
        ),
        (
            NEW_PLOT_METHODS[2],
            FIGURE_ROOT
            / f"Alibaba_前一天同小时_真实值对比_{figure_stamp}.png",
            "Alibaba 三类作业：前一天同小时与真实值",
        ),
    )
    zoom_figure_path = (
        FIGURE_ROOT / f"Alibaba_基础模型对比_nmt末72小时_{figure_stamp}.png"
    )
    all_predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    all_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig", na_rep="")
    plot_predictions(all_predictions, figure_path)
    plot_predictions(
        all_predictions,
        new_figure_path,
        methods=NEW_PLOT_METHODS,
        title="Alibaba 三类作业：基础模型逐小时预测对比",
    )
    for method, output_path, title in single_method_figures:
        plot_predictions(
            all_predictions,
            output_path,
            methods=(method,),
            title=title,
        )
    plot_zoomed_predictions(all_predictions, zoom_figure_path)

    print(all_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nPredictions: {predictions_path}")
    print(f"Metrics:     {metrics_path}")
    print(f"Figure:      {figure_path}")
    print(f"New figure:  {new_figure_path}")
    for _, output_path, _ in single_method_figures:
        print(f"Method plot: {output_path}")
    print(f"Zoom figure: {zoom_figure_path}")


if __name__ == "__main__":
    main()
