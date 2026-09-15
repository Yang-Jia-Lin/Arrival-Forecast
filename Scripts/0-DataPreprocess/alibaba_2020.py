"""Preprocess Alibaba GPU Cluster Trace 2020."""

import pandas as pd

from common import INTERIM_ROOT, METADATA_ROOT, PROCESSED_ROOT, RAW_ROOT
from common import aggregate_events, write_audit


def main() -> None:
    raw = RAW_ROOT / "alibaba_gpu_2020"
    jobs = pd.read_csv(
        raw / "pai_job_table.csv",
        header=None,
        names=["job_name", "job_id", "user", "status", "start_time", "end_time"],
    )
    tags = pd.read_csv(
        raw / "pai_group_tag_table.csv",
        header=None,
        names=["job_id", "tag_user", "gpu_type_spec", "group", "workload"],
    )
    stats = {
        "job_rows": len(jobs),
        "unique_job_ids": int(jobs.job_id.nunique()),
        "unique_job_names": int(jobs.job_name.nunique()),
        "tag_rows": len(tags),
        "tag_unique_job_ids": int(tags.job_id.nunique()),
        "tagged_rows": int(tags.workload.notna().sum()),
        "statuses": jobs.status.value_counts(dropna=False).to_dict(),
        "missing_submission": int(jobs.start_time.isna().sum()),
        "nonpositive_submission": int(jobs.start_time.le(0).sum()),
        "duplicate_job_id_rows": int(jobs.job_id.duplicated().sum()),
    }

    conflicts = tags.dropna(subset=["workload"]).groupby("job_id").workload.nunique()
    conflict_ids = conflicts[conflicts > 1].index
    stats["conflicting_label_job_ids"] = len(conflict_ids)
    unique_tags = (
        tags[~tags.job_id.isin(conflict_ids)]
        .sort_values("workload", na_position="last")
        .drop_duplicates("job_id")
    )
    jobs = (
        jobs.dropna(subset=["job_id", "start_time"])
        .loc[lambda frame: frame.start_time.gt(0)]
        .sort_values("start_time")
        .drop_duplicates("job_id")
        .merge(
            unique_tags[["job_id", "workload", "group"]],
            on="job_id",
            how="left",
            validate="one_to_one",
        )
    )
    events = jobs.rename(
        columns={"start_time": "arrival_time_s", "workload": "task_type"}
    )[["job_id", "job_name", "user", "status", "task_type", "group", "arrival_time_s"]]

    interim = INTERIM_ROOT / "alibaba_gpu_2020"
    (interim / "samples").mkdir(parents=True, exist_ok=True)
    events.to_csv(interim / "arrivals.csv.gz", index=False, compression="gzip")
    tagged = events.dropna(subset=["task_type"])
    tagged.head(100).to_csv(
        interim / "samples" / "tagged_first100.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stats.update(
        valid_jobs=len(events),
        tagged_valid_jobs=len(tagged),
        labelled_fraction=len(tagged) / len(events),
        workload_counts=tagged.task_type.value_counts().to_dict(),
        min_submission_s=float(events.arrival_time_s.min()),
        max_submission_s=float(events.arrival_time_s.max()),
    )
    aggregate_events(
        tagged,
        PROCESSED_ROOT / "alibaba_gpu_2020" / "arrivals_hourly.csv",
        events.arrival_time_s.min(),
        events.arrival_time_s.max(),
    )
    write_audit(METADATA_ROOT / "alibaba_gpu_2020" / "audit.json", stats)


if __name__ == "__main__":
    main()
