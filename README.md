# Arrival Forecast

面向公开任务日志，生成统一的任务到达序列，并研究具有明确数学表达和参数含义的拟合与预测方法。当前仅完成阶段 0：数据预处理。

## 目录

- `Data/Raw`：已下载的发布方原始数据，只读使用。
- `Data/Interim`：清洗后的事件级记录，一行表示一个任务到达事件。
- `Data/Processed`：按分钟或小时聚合的计数序列。
- `Data/Metadata`：字段说明、来源清单和清洗审计。
- `Scripts/0-preprocess`：当前阶段的全部可执行代码。
- `Src`：留给后续拟合、预测和评估的可复用核心代码。
- `Outputs`：图、表、报告和日志。

## 预处理脚本

- `run.py`：统一入口，可运行全部或指定数据集，并自动校验结果。
- `alibaba_2020.py`：Alibaba 2020 作业清洗、标签关联和小时聚合。
- `philly_2017.py`：Philly 2017 训练作业清洗和小时聚合。
- `google_2011.py`：Google 2011 首次有效 `SUBMIT` 提取和小时聚合。
- `google_2019.py`：Google 2019 cell a 首次有效作业 `SUBMIT` 提取和小时聚合。
- `azure_2019.py`：Azure Functions 2019 分钟计数读取和小时聚合。
- `validate.py`：核对处理结果总数与审计记录。
- `plot.py`：可选的五项数据集概览图及统计摘要。
- `common.py`：本阶段共用的路径、聚合和审计写入函数。

## 当前数据口径

- Alibaba：一个 `job` 的 `start_time` 作为提交时间；仅 `workload` 非空的子集可按明确标签分组。
- Philly：一个训练 `job` 的 `submitted_time`；调度重试不重复计数。
- Google 2011：一个 `job` 的首次有效 `SUBMIT`；调度类别不是业务类型。
- Google 2019：cell a 中 `collection_type=JOB` 的首次有效 `SUBMIT`；另提供无已记录父作业口径。
- Azure 2019：发布方提供的分钟调用计数；计数在函数执行后记录，不是逐请求原始到达时间戳。

## 使用

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 全部数据集：预处理 + 校验
python Scripts/0-preprocess/run.py

# 只处理指定数据集
python Scripts/0-preprocess/run.py google_2011 google_2019

# 全部数据集并重新生成概览图
python Scripts/0-preprocess/run.py --plot

# 不重复预处理，只校验已有结果
python Scripts/0-preprocess/validate.py
```

原始数据和事件级中间数据默认不提交到 Git；小时计数等较小的模型输入可按需要版本管理。
