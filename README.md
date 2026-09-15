# Arrival Forecast

面向公开集群与云函数日志，统一构造任务到达计数序列，为后续可解释的函数拟合与滚动预测提供数据基础。研究目标是：按任务类型或可用分组，利用历史到达数量拟合具有明确数学表达和参数含义的函数，并预测未来时间窗口内的到达数量。

## 当前进度

目前已完成阶段 0（数据预处理）：原始数据核验、有效到达提取、去重、时间单位统一、固定窗口聚合、审计校验和探索性绘图。

尚未完成数学函数拟合、未来窗口预测、滚动验证或调度仿真。当前图表展示的是清洗后的真实计数，连线不代表拟合或预测结果；数据未做平滑、去尖峰、去趋势、标准化或插值。

## 数据集与统一口径

| 数据集 | 一次到达的定义 | 分组含义 | 已校验总量 |
| --- | --- | --- | ---: |
| Alibaba GPU 2020 | 一个有明确 workload 标签的 `job`，以 `start_time` 作为提交时间 | `bert`、`ctr`、`nmt` 等 workload | 102,445 个作业 |
| Microsoft Philly 2017 | 一个训练 `job`，以 `submitted_time` 作为到达时间；调度重试不重复计数 | 统一记为 `DNN_training` | 117,325 个作业 |
| Google Cluster 2011 | 一个 `job` 的首次有效 `SUBMIT` | `scheduling_class`，表示调度敏感性而非业务类型 | 668,087 个作业 |
| Google Cluster 2019 | cell a 中 `collection_type=JOB` 的首次有效 `SUBMIT` | `scheduling_class`；另提供无已记录父作业口径 | 4,713,501 个作业 |
| Azure Functions 2019 | 发布方记录的每分钟函数调用计数 | `Trigger`，不是业务类型 | 12,495,810,846 次调用 |

注意：Azure 数据是函数执行后记录的分钟计数，不是逐请求原始到达时间戳；Google 的 `scheduling_class` 和 Azure 的 `Trigger` 也不能直接解释为业务任务类型。

## 项目结构

```text
Arrival_Forecast/
├─ Data/
│  ├─ Raw/          # 发布方原始数据，只读使用，不提交 Git
│  ├─ Interim/      # 清洗后的事件级记录，不提交 Git
│  ├─ Processed/    # 分钟/小时聚合的模型输入序列
│  └─ Metadata/     # 来源、字段、清洗口径和审计记录
├─ Scripts/
│  └─ 0-DataPreprocess/  # 阶段 0 的处理、校验与绘图脚本
├─ Src/             # 后续拟合、预测和评估的可复用代码
├─ Outputs/
│  ├─ Figures/      # 图
│  ├─ Tables/       # 表格
│  └─ Logs/         # 本地检查日志，不提交 Git
├─ requirements.txt
└─ README.md
```

## 环境准备

建议使用 Python 3.10 或更高版本。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 运行方法

```powershell
# 处理全部五项数据，并校验聚合总量
python Scripts/0-DataPreprocess/run.py

# 只处理指定数据集；可选名称见下方
python Scripts/0-DataPreprocess/run.py google_2011 google_2019

# 处理全部数据、校验并重新生成概览图
python Scripts/0-DataPreprocess/run.py --plot

# 不重复预处理，只校验已有的 Processed 与 Metadata
python Scripts/0-DataPreprocess/validate.py

# 不重复预处理，只根据已有 Processed 数据重新绘图
python Scripts/0-DataPreprocess/plot.py
```

可选数据集名称：`alibaba_2020`、`philly_2017`、`google_2011`、`google_2019`、`azure_2019`。

运行预处理前，需要将各数据集的原始文件放在对应的 `Data/Raw/<dataset>/` 目录中。原始数据体积较大，不随仓库提交；确切来源、文件清单、哈希或发布方说明保存在 `Data/Metadata/`。

## 主要脚本

- `run.py`：统一入口，可运行全部或指定数据集，并在处理后自动校验。
- `alibaba_2020.py`：Alibaba 作业清洗、标签关联和小时聚合。
- `philly_2017.py`：Philly 训练作业清洗和小时聚合。
- `google_2011.py`：Google 2011 首次有效 `SUBMIT` 提取和小时聚合。
- `google_2019.py`：Google 2019 cell a 首次有效作业 `SUBMIT` 提取和小时聚合。
- `azure_2019.py`：Azure Functions 分钟计数读取、触发方式汇总和小时聚合。
- `validate.py`：核对处理结果总量与审计记录。
- `plot.py`：生成五项数据集的探索性计数曲线与统计摘要。
- `common.py`：共享路径、固定窗口聚合和审计写入函数。

## 版本管理约定

- `Data/Raw/` 和 `Data/Interim/` 属于可重新获取或生成的大文件，默认忽略。
- `Data/Processed/` 和 `Data/Metadata/` 体积较小，用于复现实验口径，可按需纳入版本管理。
- `Outputs/Figures/` 与 `Outputs/Tables/` 可保存需要共享的结果；`Outputs/Logs/` 和归档副本默认忽略。
