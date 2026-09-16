# Arrival Forecast

面向公开集群与云函数日志，统一构造任务到达计数序列，为后续可解释的函数拟合与滚动预测提供数据基础。研究目标是：按任务类型或可用分组，利用历史到达数量拟合具有明确数学表达和参数含义的函数，并预测未来时间窗口内的到达数量。

## 当前进度

目前已完成阶段 0（数据预处理）：原始数据核验、有效到达提取、去重、时间单位统一、固定窗口聚合、审计校验和探索性绘图。阶段 1 已针对 Alibaba 的 `bert`、`ctr`、`nmt` 三类作业实现逐小时滚动预测基线和误差分析。

五项数据集概览图仍只展示清洗后的真实计数，连线不代表拟合或预测结果；数据未做平滑、去尖峰、去趋势、标准化或插值。当前阶段也尚未实现更复杂的时变到达率、参数化趋势/周期函数或调度仿真。

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
│  ├─ 0-DataPreprocess/  # 阶段 0 的处理、校验与绘图脚本
│  └─ 1-FitForecast/     # 阶段 1 的拟合、滚动预测与误差分析
├─ Src/             # 后续拟合、预测和评估的可复用代码
├─ Tests/           # 标准库单元测试
├─ Outputs/
│  ├─ Figures/
│  │  ├─ 0-DataPreprocess/  # 阶段 0 图
│  │  └─ 1-FitForecast/     # 阶段 1 图
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

# Alibaba 三类作业：拟合、逐小时滚动预测和误差分析
python Scripts/1-FitForecast/run.py

# 运行阶段 1 单元测试
python -m unittest discover -s Tests -v
```

可选数据集名称：`alibaba_2020`、`philly_2017`、`google_2011`、`google_2019`、`azure_2019`。

运行预处理前，需要将各数据集的原始文件放在对应的 `Data/Raw/<dataset>/` 目录中。原始数据体积较大，不随仓库提交；确切来源、文件清单、哈希或发布方说明保存在 `Data/Metadata/`。

## 阶段 1：Alibaba 拟合与滚动预测

阶段 1 读取 `Data/Processed/alibaba_gpu_2020/arrivals_hourly.csv`。每类序列以前 50% 小时作为初始历史，后 50% 用于逐小时评估；预测当前小时后才将其真实计数加入历史，避免未来信息泄漏。

当前比较五种简单点预测方法；后面三种新增方法的参数在评估前固定，不使用测试段调参：

- **同一时刻均值**：使用此前所有相同日内小时的平均计数，
  \(\hat y_t=\operatorname{mean}\{y_s:s<t,\ h(s)=h(t)\}\)。由于源数据是相对时间，\(h(t)\) 表示轨迹中 `window_start_s // 3600 mod 24` 的位置，不代表真实时区钟点。
- **泊松到达率**：假设下一小时 \(Y_t\sim\operatorname{Poisson}(\lambda_t)\)，并用此前所有小时计数的均值估计 \(\hat\lambda_t\)；点预测为 \(E[Y_t]=\hat\lambda_t\)，不进行随机抽样。
- **24 小时滑动泊松**：假设下一小时 \(Y_t\sim\operatorname{Poisson}(\lambda_t)\)，用最近 24 小时估计 \(\hat\lambda_t=\frac{1}{24}\sum_{k=1}^{24}y_{t-k}\)；点预测取 \(\hat\lambda_t\)。因此它的点预测也等于最近 24 小时移动平均，不进行随机抽样。
- **指数平滑（EWMA）**：从初始历史的第一个计数开始递推 \(m_s=0.2y_s+0.8m_{s-1}\)，用 \(\hat y_t=m_{t-1}\) 预测下一小时。
- **前一天同小时**：用 \(\hat y_t=y_{t-24}\) 预测，不估计额外参数。

脚本对五种方法使用完全相同的评估小时，报告 MAE、RMSE、平均误差（预测值减真实值），并给出初始历史计数的方差／均值比作为泊松假设的描述性诊断。该比值不是严格的分布检验；各方法使用的历史范围或时间信息不同，因此滑动泊松比固定率泊松误差更低，只能说明近期水平可能更有预测价值，不能单独证明泊松分布正确。

输出文件：

- `Outputs/Tables/alibaba_rolling_predictions.csv`：每个评估小时的真实值和五个预测值。
- `Outputs/Tables/alibaba_rolling_metrics.csv`：各任务类型、五种方法的误差和诊断量。
- `Outputs/Figures/0-DataPreprocess/五项数据集_初步到达曲线_<mmdd>_<hhss>.png`：阶段 0 的五项数据概览图。
- `Outputs/Figures/1-FitForecast/Alibaba_滚动预测对比_<mmdd>_<hhss>.png`：保留原有两种基线的对比图。
- `Outputs/Figures/1-FitForecast/Alibaba_基础模型对比_<mmdd>_<hhss>.png`：真实值与三种新增方法的对比图。
- `Outputs/Figures/1-FitForecast/Alibaba_24小时滑动泊松_真实值对比_<mmdd>_<hhss>.png`：滑动泊松与真实值的独立对比图。
- `Outputs/Figures/1-FitForecast/Alibaba_EWMA_真实值对比_<mmdd>_<hhss>.png`：指数平滑与真实值的独立对比图。
- `Outputs/Figures/1-FitForecast/Alibaba_前一天同小时_真实值对比_<mmdd>_<hhss>.png`：前一天同小时与真实值的独立对比图。
- `Outputs/Figures/1-FitForecast/Alibaba_基础模型对比_nmt末72小时_<mmdd>_<hhss>.png`：三种新增方法在 `nmt` 最后 72 个评估小时的局部放大图。

图片统一按 `数据集_图片名_mmdd_hhss.png` 命名，其中 `mmdd` 为月日，`hhss` 为小时和秒；同一次脚本运行生成的图片共享同一时间戳。阶段 0 的图片保存在 `Outputs/Figures/0-DataPreprocess/`，阶段 1 的图片保存在 `Outputs/Figures/1-FitForecast/`。

## 主要脚本

- `Scripts/0-DataPreprocess/run.py`：预处理统一入口，可运行全部或指定数据集，并在处理后自动校验。
- `Scripts/1-FitForecast/run.py`：Alibaba 三类作业的基线拟合、滚动预测、误差汇总和绘图入口。
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
