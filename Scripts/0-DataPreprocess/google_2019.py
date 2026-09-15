"""Preprocess Google Cluster Data 2019 cell a job arrivals."""

import json

import pandas as pd
import pyarrow.parquet as pq

from common import INTERIM_ROOT, METADATA_ROOT, PROCESSED_ROOT, RAW_ROOT
from common import aggregate_events, write_audit


def main() -> None:
    source = RAW_ROOT / "google_cluster_2019" / "cell_a" / "collection_events"
    files = sorted(source.glob("*.parquet.gz"))
    inventory = json.loads(
        (METADATA_ROOT / "google_cluster_2019" / "collection_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        item["name"]: int(item["size"])
        for item in inventory["items"]
        if item["name"].endswith(".parquet.gz")
    }
    actual = {path.name: path.stat().st_size for path in files}
    if actual != expected:
        raise ValueError("Google 2019 cell a shards do not match the inventory")

    columns = [
        "time",
        "type",
        "collection_id",
        "scheduling_class",
        "missing_type",
        "collection_type",
        "parent_collection_id",
    ]
    parts = []
    event_rows = 0
    max_event_time = 0
    job_submit_rows = 0
    initial_rows = 0
    missing_rows = 0
    nonjob_rows = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=100_000, columns=columns
        ):
            frame = batch.to_pandas()
            event_rows += len(frame)
            max_event_time = max(max_event_time, int(frame.time.max()))
            submits = frame[frame.type.eq(0)]
            nonjob_rows += int(submits.collection_type.ne(0).sum())
            submits = submits[submits.collection_type.eq(0)]
            job_submit_rows += len(submits)
            initial_rows += int(submits.time.eq(0).sum())
            missing_rows += int(submits.missing_type.fillna(0).ne(0).sum())
            parts.append(
                submits[
                    submits.time.gt(0) & submits.missing_type.fillna(0).eq(0)
                ].copy()
            )

    jobs = pd.concat(parts, ignore_index=True)
    before = len(jobs)
    jobs = jobs.sort_values("time").drop_duplicates("collection_id", keep="first")
    jobs["arrival_time_s"] = jobs.time / 1e6
    jobs["task_type"] = "scheduling_class_" + jobs.scheduling_class.fillna(-1).astype(
        int
    ).astype(str)
    events = jobs[
        ["collection_id", "task_type", "arrival_time_s", "parent_collection_id"]
    ].rename(columns={"collection_id": "job_id"})
    events["job_id"] = events.job_id.astype("int64")
    events["parent_collection_id"] = events.parent_collection_id.astype("Int64")

    interim = INTERIM_ROOT / "google_cluster_2019" / "cell_a"
    (interim / "samples").mkdir(parents=True, exist_ok=True)
    events.to_csv(interim / "arrivals.csv.gz", index=False, compression="gzip")
    events.head(100).to_csv(
        interim / "samples" / "first100.csv", index=False, encoding="utf-8-sig"
    )
    processed = PROCESSED_ROOT / "google_cluster_2019" / "cell_a"
    aggregate_events(
        events, processed / "arrivals_hourly.csv", 600, max_event_time / 1e6
    )
    roots = events[events.parent_collection_id.isna()].copy()
    aggregate_events(
        roots,
        processed / "arrivals_no_recorded_parent_hourly.csv",
        600,
        max_event_time / 1e6,
    )
    write_audit(
        METADATA_ROOT / "google_cluster_2019" / "cell_a_audit.json",
        {
            "scope": "cell a, all collection-event Parquet shards only; other cells and instance tables not downloaded",
            "files": len(files),
            "compressed_bytes": sum(path.stat().st_size for path in files),
            "all_event_rows": event_rows,
            "job_submit_rows": job_submit_rows,
            "nonjob_submit_rows_excluded": nonjob_rows,
            "submit_time_zero_rows_excluded": initial_rows,
            "submit_missing_type_nonzero_excluded": missing_rows,
            "duplicate_valid_job_submits_removed": before - len(jobs),
            "valid_distinct_jobs": len(events),
            "class_counts": events.task_type.value_counts().to_dict(),
            "min_valid_submit_s": events.arrival_time_s.min(),
            "max_event_time_s": max_event_time / 1e6,
            "jobs_with_parent": int(events.parent_collection_id.notna().sum()),
            "jobs_without_recorded_parent": len(roots),
            "analysis_start_s": 600,
        },
    )


if __name__ == "__main__":
    main()
