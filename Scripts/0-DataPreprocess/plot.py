"""Plot processed arrival counts without smoothing or spike removal."""

from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from common import FIGURE_ROOT, PROCESSED_ROOT, PROJECT_ROOT


def configure_plotting() -> None:
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


def main() -> None:
    configure_plotting()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = [
        (
            PROCESSED_ROOT / "alibaba_gpu_2020" / "arrivals_hourly.csv",
            "task_type",
            "arrival_count",
            "阿里 2020：有标签作业的提交数量",
            True,
        ),
        (
            PROCESSED_ROOT / "microsoft_philly_2017" / "arrivals_hourly.csv",
            "task_type",
            "arrival_count",
            "Philly：训练作业提交总量",
            False,
        ),
        (
            PROCESSED_ROOT / "google_cluster_2011" / "arrivals_hourly.csv",
            "task_type",
            "arrival_count",
            "Google 2011：各调度类别的作业提交量",
            False,
        ),
        (
            PROCESSED_ROOT / "google_cluster_2019" / "cell_a" / "arrivals_hourly.csv",
            "task_type",
            "arrival_count",
            "Google 2019 cell a：各调度类别的作业提交量（含子作业）",
            False,
        ),
        (
            PROCESSED_ROOT / "azure_functions_2019" / "trigger_arrivals_hourly.csv",
            "trigger",
            "invocation_count",
            "Azure 2019：各触发方式的调用计数",
            False,
        ),
    ]
    figure, axes = plt.subplots(5, 1, figsize=(15, 17), constrained_layout=True)
    summaries = []
    for axis, (path, group_column, value_column, title, shift_to_first) in zip(
        axes, datasets
    ):
        data = pd.read_csv(path)
        top_groups = (
            data.groupby(group_column)[value_column]
            .sum()
            .sort_values(ascending=False)
            .head(3)
            .index
        )
        shift = data.window_start_s.min() if shift_to_first else 0
        for group in top_groups:
            subset = data[data[group_column].eq(group)]
            axis.plot(
                (subset.window_start_s - shift) / 86400,
                subset[value_column],
                linewidth=0.7,
                label=str(group).replace("scheduling_class_", "调度类别 "),
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("相对时间（天）")
        axis.set_ylabel("数量 / 小时")
        axis.grid(alpha=0.18)
        axis.legend(loc="upper right", fontsize=9)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 5))
        for group, subset in data.groupby(group_column):
            summaries.append(
                {
                    "file": path.relative_to(PROJECT_ROOT).as_posix(),
                    "group": group,
                    "total_count": int(subset[value_column].sum()),
                    "hour_windows": len(subset),
                    "zero_fraction": float(subset[value_column].eq(0).mean()),
                    "max_hour_count": int(subset[value_column].max()),
                    "mean_hour_count": float(subset[value_column].mean()),
                }
            )
    figure.suptitle("五项数据集：初步计数曲线", fontsize=19, fontweight="bold")
    figure.get_layout_engine().set(rect=(0, 0.025, 1, 0.975))
    figure.text(
        0.03,
        0.012,
        "原始计数按小时汇总；未平滑、未删除尖峰。每图最多展示总量前三组。",
        fontsize=10,
        color="#555555",
    )
    stamp = datetime.now().strftime("%m%d_%H%S")
    figure.savefig(
        FIGURE_ROOT / f"五项数据集_初步到达曲线_{stamp}.png", dpi=140
    )
    plt.close(figure)

    summary_path = PROCESSED_ROOT / "summary" / "series_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
