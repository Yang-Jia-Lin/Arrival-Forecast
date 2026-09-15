"""Preprocess Google Cluster Data 2011 job arrivals."""

import pandas as pd

from common import INTERIM_ROOT, METADATA_ROOT, PROCESSED_ROOT, RAW_ROOT
from common import aggregate_events, write_audit


def main() -> None:
    columns = [
        "time",
        "missing_info",
        "job_id",
        "event_type",
        "user",
        "scheduling_class",
        "job_name",
        "logical_job_name",
    ]
    files = sorted((RAW_ROOT / "google_cluster_2011" / "job_events").glob("*.csv.gz"))
    if len(files) != 500:
        raise ValueError(f"Expected 500 job-event shards, found {len(files)}")

    parts = []
    event_rows = 0
    max_event_time = 0
    for path in files:
        frame = pd.read_csv(
            path,
            header=None,
            names=columns,
            dtype={
                "job_id": "string",
                "user": "string",
                "job_name": "string",
                "logical_job_name": "string",
            },
        )
        event_rows += len(frame)
        max_event_time = max(max_event_time, int(frame.time.max()))
        parts.append(frame.loc[frame.event_type.eq(0)].copy())

    submits = pd.concat(parts, ignore_index=True)
    stats = {
        "files": len(files),
        "event_rows": event_rows,
        "submit_rows": len(submits),
        "submit_time_lt_600s": int((submits.time < 600e6).sum()),
        "submit_missing_info": int(submits.missing_info.notna().sum()),
        "max_event_time_s": max_event_time / 1e6,
    }
    submits = submits[submits.time.ge(600e6) & submits.missing_info.isna()].copy()
    before = len(submits)
    submits = submits.sort_values("time").drop_duplicates("job_id", keep="first")
    stats["repeated_submit_rows_removed"] = before - len(submits)
    submits["arrival_time_s"] = submits.time / 1e6
    submits["task_type"] = "scheduling_class_" + submits.scheduling_class.fillna(
        -1
    ).astype(int).astype(str)
    events = submits[
        ["job_id", "task_type", "arrival_time_s", "user", "logical_job_name"]
    ]

    interim = INTERIM_ROOT / "google_cluster_2011"
    (interim / "samples").mkdir(parents=True, exist_ok=True)
    events.to_csv(interim / "arrivals.csv.gz", index=False, compression="gzip")
    events.head(100).to_csv(
        interim / "samples" / "first100.csv", index=False, encoding="utf-8-sig"
    )
    aggregate_events(
        events,
        PROCESSED_ROOT / "google_cluster_2011" / "arrivals_hourly.csv",
        600,
        max_event_time / 1e6,
    )
    stats.update(
        valid_distinct_jobs=len(events),
        class_counts=events.task_type.value_counts().to_dict(),
        time_min_s=float(events.arrival_time_s.min()),
    )
    write_audit(METADATA_ROOT / "google_cluster_2011" / "audit.json", stats)


if __name__ == "__main__":
    main()
