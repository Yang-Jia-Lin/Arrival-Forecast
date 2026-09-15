"""Preprocess Microsoft Philly GPU Cluster Trace 2017."""

import json

import pandas as pd

from common import INTERIM_ROOT, METADATA_ROOT, PROCESSED_ROOT, RAW_ROOT
from common import aggregate_events, write_audit


def main() -> None:
    jobs = json.loads(
        (RAW_ROOT / "microsoft_philly_2017" / "cluster_job_log.json").read_text(
            encoding="utf-8"
        )
    )
    frame = pd.DataFrame(
        [
            {
                key: job.get(key)
                for key in ["jobid", "submitted_time", "vc", "user", "status"]
            }
            for job in jobs
        ]
    )
    arrival = pd.to_datetime(frame.submitted_time, errors="coerce")
    stats = {
        "raw_jobs": len(frame),
        "unique_job_ids": int(frame.jobid.nunique()),
        "invalid_submission": int(arrival.isna().sum()),
        "statuses": frame.status.value_counts(dropna=False).to_dict(),
        "distinct_vcs": int(frame.vc.nunique()),
        "timezone_note": "submitted_time is timezone-naive; retained as source local wall time, no UTC claim",
    }
    frame["arrival_datetime"] = arrival
    frame = (
        frame.dropna(subset=["jobid", "arrival_datetime"])
        .sort_values("arrival_datetime")
        .drop_duplicates("jobid")
    )
    origin = frame.arrival_datetime.min().normalize()
    frame["arrival_time_s"] = (frame.arrival_datetime - origin).dt.total_seconds()
    frame["task_type"] = "DNN_training"
    events = frame.rename(columns={"jobid": "job_id"})[
        [
            "job_id",
            "task_type",
            "arrival_datetime",
            "arrival_time_s",
            "vc",
            "user",
            "status",
        ]
    ]

    interim = INTERIM_ROOT / "microsoft_philly_2017"
    (interim / "samples").mkdir(parents=True, exist_ok=True)
    events.to_csv(interim / "arrivals.csv.gz", index=False, compression="gzip")
    events.head(100).to_csv(
        interim / "samples" / "first100.csv", index=False, encoding="utf-8-sig"
    )
    aggregate_events(
        events,
        PROCESSED_ROOT / "microsoft_philly_2017" / "arrivals_hourly.csv",
        events.arrival_time_s.min(),
        events.arrival_time_s.max(),
    )
    stats.update(
        valid_jobs=len(events),
        first_submission=str(frame.arrival_datetime.min()),
        last_submission=str(frame.arrival_datetime.max()),
        origin=str(origin),
    )
    write_audit(METADATA_ROOT / "microsoft_philly_2017" / "audit.json", stats)


if __name__ == "__main__":
    main()
