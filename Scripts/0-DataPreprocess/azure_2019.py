"""Preprocess Azure Functions Trace 2019 invocation counts."""

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import METADATA_ROOT, PROCESSED_ROOT, RAW_ROOT, write_audit


def main() -> None:
    totals = []
    selected_rows = []
    days = []
    selected_keys = None
    all_keys = set()
    archive = RAW_ROOT / "azure_functions_2019" / "azurefunctions_dataset2019.tar.xz"
    with tarfile.open(archive, "r|xz") as source:
        for member in source:
            name = Path(member.name).name
            if not member.isfile() or not name.startswith(
                "invocations_per_function_md.anon.d"
            ):
                continue
            day = int(name.split(".d")[-1].split(".")[0])
            daily = {}
            day_keys = set()
            row_count = 0
            negative_cells = 0
            duplicate_ids = 0
            candidates = []
            payload = io.BytesIO(source.extractfile(member).read())
            for chunk in pd.read_csv(payload, chunksize=1500):
                minute_columns = [str(index) for index in range(1, 1441)]
                values = chunk[minute_columns].to_numpy(dtype=np.int64)
                row_count += len(chunk)
                negative_cells += int((values < 0).sum())
                ids = (
                    chunk[["HashOwner", "HashApp", "HashFunction"]]
                    .astype(str)
                    .agg("/".join, axis=1)
                )
                for key in ids:
                    duplicate_ids += int(key in day_keys)
                    day_keys.add(key)
                all_keys.update(day_keys)
                for trigger, indices in chunk.groupby(
                    "Trigger", dropna=False
                ).indices.items():
                    daily[str(trigger)] = daily.get(
                        str(trigger), np.zeros(1440, dtype=np.int64)
                    ) + values[indices].sum(axis=0)
                if selected_keys is None:
                    row_totals = values.sum(axis=1)
                    for index in np.argsort(row_totals)[-3:]:
                        candidates.append(
                            (
                                int(row_totals[index]),
                                ids.iloc[index],
                                str(chunk.Trigger.iloc[index]),
                                values[index].copy(),
                            )
                        )
                else:
                    for index, key in enumerate(ids):
                        if key in selected_keys:
                            selected_rows.append(
                                pd.DataFrame(
                                    {
                                        "function_id": key,
                                        "trigger": str(chunk.Trigger.iloc[index]),
                                        "day": day,
                                        "minute": np.arange(1, 1441),
                                        "invocation_count": values[index],
                                    }
                                )
                            )
            if selected_keys is None:
                best = sorted(candidates, key=lambda item: item[0], reverse=True)[:3]
                selected_keys = {item[1] for item in best}
                for _, key, trigger, values in best:
                    selected_rows.append(
                        pd.DataFrame(
                            {
                                "function_id": key,
                                "trigger": trigger,
                                "day": day,
                                "minute": np.arange(1, 1441),
                                "invocation_count": values,
                            }
                        )
                    )
            for trigger, values in daily.items():
                totals.append(
                    pd.DataFrame(
                        {
                            "trigger": trigger,
                            "day": day,
                            "minute": np.arange(1, 1441),
                            "invocation_count": values,
                        }
                    )
                )
            days.append(
                {
                    "day": day,
                    "function_rows": row_count,
                    "duplicate_function_ids": duplicate_ids,
                    "negative_count_cells": negative_cells,
                    "invocations": sum(
                        int(values.sum()) for values in daily.values()
                    ),
                }
            )

    minute = pd.concat(totals)
    minute["window_start_s"] = (minute.day - 1) * 86400 + (minute.minute - 1) * 60
    minute["window_seconds"] = 60
    output = PROCESSED_ROOT / "azure_functions_2019"
    output.mkdir(parents=True, exist_ok=True)
    minute.to_csv(
        output / "trigger_arrivals_minute.csv.gz", index=False, compression="gzip"
    )
    pd.concat(selected_rows).to_csv(
        output / "selected_functions_minute.csv.gz", index=False, compression="gzip"
    )
    hourly = (
        minute.assign(hour=(minute.window_start_s // 3600).astype(int))
        .groupby(["trigger", "hour"], as_index=False)
        .invocation_count.sum()
    )
    hourly["window_start_s"] = hourly.pop("hour") * 3600
    hourly.to_csv(
        output / "trigger_arrivals_hourly.csv", index=False, encoding="utf-8-sig"
    )
    if int(minute.invocation_count.sum()) != sum(item["invocations"] for item in days):
        raise ValueError("Azure aggregate total does not match daily totals")
    write_audit(
        METADATA_ROOT / "azure_functions_2019" / "audit.json",
        {
            "days": sorted(days, key=lambda item: item["day"]),
            "distinct_function_keys": len(all_keys),
            "triggers": sorted(minute.trigger.unique()),
            "selection": "top 3 functions by first encountered day total; identity is not business type",
            "selected_function_keys": sorted(selected_keys),
            "counting_note": "published invocation counts are recorded after execution; not raw external arrival timestamps",
        },
    )


if __name__ == "__main__":
    main()
