"""Validate processed count totals against preprocessing audits."""

import json
from pathlib import Path

import pandas as pd

from common import METADATA_ROOT, PROCESSED_ROOT


DATASETS = (
    "alibaba_2020",
    "philly_2017",
    "google_2011",
    "google_2019",
    "azure_2019",
)


def read_audit(dataset: str, name: str = "audit.json") -> dict:
    return json.loads((METADATA_ROOT / dataset / name).read_text(encoding="utf-8"))


def count_sum(path: Path, column: str = "arrival_count") -> int:
    data = pd.read_csv(path)
    if (data[column] < 0).any():
        raise ValueError(f"Negative count in {path}")
    return int(data[column].sum())


def check(dataset: str) -> tuple[int, int]:
    if dataset == "alibaba_2020":
        return (
            count_sum(PROCESSED_ROOT / "alibaba_gpu_2020" / "arrivals_hourly.csv"),
            read_audit("alibaba_gpu_2020")["tagged_valid_jobs"],
        )
    if dataset == "philly_2017":
        return (
            count_sum(
                PROCESSED_ROOT / "microsoft_philly_2017" / "arrivals_hourly.csv"
            ),
            read_audit("microsoft_philly_2017")["valid_jobs"],
        )
    if dataset == "google_2011":
        return (
            count_sum(PROCESSED_ROOT / "google_cluster_2011" / "arrivals_hourly.csv"),
            read_audit("google_cluster_2011")["valid_distinct_jobs"],
        )
    if dataset == "google_2019":
        return (
            count_sum(
                PROCESSED_ROOT
                / "google_cluster_2019"
                / "cell_a"
                / "arrivals_hourly.csv"
            ),
            read_audit("google_cluster_2019", "cell_a_audit.json")[
                "valid_distinct_jobs"
            ],
        )
    if dataset == "azure_2019":
        audit = read_audit("azure_functions_2019")
        return (
            count_sum(
                PROCESSED_ROOT
                / "azure_functions_2019"
                / "trigger_arrivals_minute.csv.gz",
                "invocation_count",
            ),
            sum(day["invocations"] for day in audit["days"]),
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def validate(datasets: list[str] | tuple[str, ...] = DATASETS) -> None:
    for dataset in datasets:
        actual, expected = check(dataset)
        if actual != expected:
            raise ValueError(
                f"{dataset}: processed total {actual:,} != audit total {expected:,}"
            )
        print(f"{dataset}: {actual:,} records/counts verified")


if __name__ == "__main__":
    validate()
