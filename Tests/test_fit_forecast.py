"""Tests for the stage-1 rolling forecast baselines."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "Scripts" / "1-FitForecast" / "run.py"
)
SPEC = importlib.util.spec_from_file_location("fit_forecast_run", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
fit_forecast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit_forecast)


def make_series(values: list[int], task_type: str = "bert") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_type": task_type,
            "window_start_s": np.arange(len(values), dtype="int64") * 3600,
            "arrival_count": values,
        }
    )


class FitForecastTests(unittest.TestCase):
    def test_rolling_predictions_use_only_past_values(self) -> None:
        values = list(range(24)) + [24] * 24
        predictions, history = fit_forecast.rolling_forecast(
            make_series(values), "bert"
        )

        self.assertEqual(len(history), 24)
        self.assertEqual(len(predictions), 24)
        self.assertEqual(predictions.iloc[0]["same_hour_mean_prediction"], 0.0)
        self.assertEqual(predictions.iloc[0]["poisson_rate_prediction"], 11.5)
        self.assertEqual(predictions.iloc[1]["same_hour_mean_prediction"], 1.0)
        self.assertEqual(predictions.iloc[1]["poisson_rate_prediction"], 12.0)

        changed_future = list(range(24)) + [999] + [0] * 23
        changed, _ = fit_forecast.rolling_forecast(
            make_series(changed_future), "bert"
        )
        for column in fit_forecast.PREDICTION_COLUMNS.values():
            self.assertEqual(predictions.iloc[0][column], changed.iloc[0][column])

    def test_constant_series_predicts_constant_for_every_method(self) -> None:
        predictions, history = fit_forecast.rolling_forecast(
            make_series([7] * 48), "bert"
        )
        for column in fit_forecast.PREDICTION_COLUMNS.values():
            np.testing.assert_allclose(predictions[column], 7)
        summary = fit_forecast.summarize_metrics(predictions, history, "bert")
        self.assertEqual(len(summary), 5)
        np.testing.assert_allclose(
            summary[["mae", "rmse", "mean_error"]], 0, atol=1e-12
        )

    def test_daily_repeat_and_rolling_poisson(self) -> None:
        predictions, _ = fit_forecast.rolling_forecast(
            make_series(list(range(24)) * 2), "bert"
        )
        self.assertEqual(predictions.iloc[0]["rolling_poisson_24h_prediction"], 11.5)
        self.assertEqual(predictions.iloc[0]["seasonal_naive_24h_prediction"], 0)
        self.assertEqual(predictions.iloc[1]["seasonal_naive_24h_prediction"], 1)
        self.assertAlmostEqual(
            predictions.iloc[1]["rolling_poisson_24h_prediction"], 11.5
        )

    def test_level_shift_updates_new_models_after_observation(self) -> None:
        predictions, _ = fit_forecast.rolling_forecast(
            make_series([0] * 24 + [24] + [0] * 23), "bert"
        )
        for column in fit_forecast.PREDICTION_COLUMNS.values():
            self.assertEqual(predictions.iloc[0][column], 0)
        self.assertEqual(predictions.iloc[1]["rolling_poisson_24h_prediction"], 1)
        self.assertAlmostEqual(predictions.iloc[1]["ewma_0_2_prediction"], 4.8)
        self.assertEqual(predictions.iloc[1]["seasonal_naive_24h_prediction"], 0)
        self.assertAlmostEqual(predictions.iloc[2]["ewma_0_2_prediction"], 3.84)

    def test_error_metrics(self) -> None:
        metrics = fit_forecast.error_metrics(
            pd.Series([1, 3]), pd.Series([2, 5])
        )
        self.assertEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.5))
        self.assertEqual(metrics["mean_error"], 1.5)

    def test_all_zero_history_is_valid(self) -> None:
        predictions, history = fit_forecast.rolling_forecast(
            make_series([0] * 48), "bert"
        )
        summary = fit_forecast.summarize_metrics(predictions, history, "bert")
        self.assertTrue((predictions.iloc[:, 3:] == 0).all().all())
        self.assertTrue((summary[["mae", "rmse", "mean_error"]] == 0).all().all())
        self.assertTrue(summary["training_variance_mean_ratio"].isna().all())

    def test_missing_hour_is_rejected(self) -> None:
        data = make_series([1] * 48).drop(index=10)
        with self.assertRaisesRegex(ValueError, "not continuous"):
            fit_forecast.validate_task_series(data, "bert")

    def test_duplicate_hour_is_rejected(self) -> None:
        data = make_series([1] * 48)
        data = pd.concat([data, data.iloc[[10]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_forecast.validate_task_series(data, "bert")

    def test_negative_count_is_rejected(self) -> None:
        data = make_series([1] * 48)
        data.loc[10, "arrival_count"] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            fit_forecast.validate_task_series(data, "bert")


if __name__ == "__main__":
    unittest.main()
