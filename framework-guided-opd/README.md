# Framework-Guided OPD

本项目在本地 Qwen3 模型与 GSM8K 上比较传统 OPD 和 Framework-Guided OPD。所有模型与数据均从服务器本地路径读取，不进行下载。

> **Research status: paused (2026-09-25).** 这是小规模探索性实验，不是已经验证有效或可直接部署的框架蒸馏方案。目前不继续扩大训练或审核；代码与结果作为可追溯的阶段性记录保留。

## 当前收尾实验：人工确认框架的四组对照

研究设想是由 Teacher 给出解题框架，让 Student 依据框架生成解答并接受 on-policy distillation（OPD）；对照组使用普通 OPD。收尾实验将“训练时是否使用框架”和“测试时是否提供框架”分别控制，因此有四组。这里测试时提供的是**固定、经人工筛选且参考答案辅助产生的框架**，不是 Teacher 在未知答案的测试题上在线生成的框架。

2026-09-25 的 50 题实验使用此前人工审核的 100 道题：最终准入 86 道，排除 14 道；36 道用于两组 Student 各自从同一 Qwen3-1.7B 基座重新训练，50 道用于两种测试条件，训练与测试题目不重叠。OPD 评分 Teacher 为冻结的 Qwen3-4B。两组各只训练一轮、同一个随机种子；严格正确要求最终答案符合 `#### number` 格式，宽松数值正确仅作诊断。

| 训练方式 | 测试时框架 | 严格正确 | 宽松数值正确（诊断） | 最终答案 token 均值 | 触及 2048 token 上限 |
|---|---|---:|---:|---:|---:|
| 框架 OPD | 不提供 | 30/50（60%） | 46/50（92%） | 173.3 | 1 |
| 框架 OPD | 提供 | 47/50（94%） | 47/50（94%） | 127.4 | 0 |
| 普通 OPD | 不提供 | 33/50（66%） | 45/50（90%） | 137.1 | 0 |
| 普通 OPD | 提供 | 47/50（94%） | 47/50（94%） | 135.0 | 0 |

结果来自已完成运行 `outputs/closeout-full-FZgvRxMy/report/summary.json`（原始运行目录保存在研究服务器，未上传模型、逐题数据或完整输出）。两组训练在“提供框架”的测试条件下同为 47/50，**没有观察到框架 OPD 训练相对普通 OPD 的额外准确率优势**。提供框架后严格格式正确率明显提高，但无框架时宽松数值正确率已经达到 45–46/50；严格与宽松之间的差距说明大部分表面提升不能直接解释为数学推理能力提升。

### 局限与暂停原因

- 这 100 道题曾参与框架审核规则修改和人工复核，是**已见开发集**；50 道测试题虽未进入本轮 Student 训练，仍不能称为独立未见测试集或证明泛化。
- 固定框架在产生和筛选时使用了参考解答与人工判断，属于特权／近似 oracle 条件。本轮没有检验未知答案时 Teacher 能否稳定生成正确框架，不能当作完整可部署方案的成绩。
- 样本量小、仅一个随机种子，Student 每组只用 36 道题进行短程训练；未报告跨种子方差或统计显著性。GSM8K 这组短题在无框架时数值正确率已经较高，存在明显天花板效应。
- 最终答案格式对严格指标影响很大。宽松指标可能从非标准输出中提取数字，仅用于定位问题，不能代替严格准确率；有框架也可能引导模型忠实执行错误计算。
- 上一轮 20 题收尾实验使用 60 道训练题；本轮改为 36 训练／50 测试，不能把跨轮分数变化归因于框架或训练方法。先前 v3/v4 审核器仍有误拒、漏检和语义覆盖边界，实验产物未被声明为训练可用的通用自动审核器。

因此本选题**暂时搁置**：目前证据不足以支持“框架引导 OPD 比普通 OPD 更有效”的核心主张。若未来恢复，应先建立独立未见、难度更有区分度的测试集，验证答案盲框架生成质量，并做多随机种子及格式／数学正确性分离评测；这些是后续建议，尚未执行。

当前收尾入口为 [`closeout_experiment.py`](closeout_experiment.py)、[`run_closeout.sh`](run_closeout.sh)，数据门禁与回归测试见 [`test_closeout.py`](test_closeout.py)，运行环境和冻结输入说明见 [`CLOSEOUT_RUNBOOK.md`](CLOSEOUT_RUNBOOK.md)。本次提交不包含 Qwen 权重、收尾实验的人工标注源文件、冻结划分或完整预测；没有这些材料不能仅凭 GitHub 克隆重现上述数值。旧结果和失败记录保留在服务器，不在此处重跑。

## 历史 v3/v4 实验链路（归档说明，非当前结果）

1. Qwen3-4B 在“题目 + 参考解答”的特权信息下生成抽象框架候选。
2. 候选先通过结构与 purity 检查，再让冻结的 4B 严格按框架解题；只有严格答案匹配参考答案时，才发布为 v3 框架标签。
3. 使用 v3 标签训练只看题目即可生成框架的 Qwen3-4B LoRA。
4. 重新训练 Vanilla OPD 与 Guided OPD。Student rollout 在首个完整 `#### number` 行后停止，不再学习答案后的重复垃圾 token。
5. 运行 v4 受控消融：无框架、空框架、固定 fallback、答案盲生成框架和 reference-aware oracle 框架。

OPD 的评分 Teacher 始终是冻结的 Qwen3-4B base。框架 Teacher 是加载了框架 LoRA 的另一份 4B。Student 只在自己实际生成的 solution token 上计算蒸馏损失；prompt 和框架 token 不参与 loss。`beta=1.0` 对应 reverse-KL OPD。

## 服务器资源

```text
项目:    /root/blockdata/framework-guided-opd/framework-guided-opd
Student: /root/eb-public/huggingface-models/Qwen/Qwen3-1.7B
Teacher: /root/eb-public/huggingface-models/Qwen/Qwen3-4B
GSM8K:  /root/eb-public/huggingface-datasets/openai/gsm8k/main/
Python: /root/blockdata/kv_cache_env/bin/python
```

## 三类质量门禁

- Purity：框架必须有 2–6 个非空步骤，不能包含具体数字、数字词、已求值等式、最终答案或 `####`。
- Semantic execution：生成训练标签时，4B 必须按候选框架得到与参考答案严格一致的 `#### number`；失败候选会被拒绝并重试。
- Answer-aware stopping：训练与评测的 Student 都只监控新生成 token，在完整换行结束的答案行后停止。未生成合法答案时才继续到 EOS 或上限。

Semantic execution gate 只用于有参考答案的特权训练标签。正式评测的 `generated_framework` 始终只读取问题。`oracle_framework` 会读取测试参考答案，只能作为诊断上界，不能作为可部署系统结果或主论文结论。

## 历史 v3/v4 运行命令（归档，不是当前收尾实验）

### 1. 生成语义合格的 v3 框架标签

```bash
cd /root/blockdata/framework-guided-opd/framework-guided-opd
nohup ./run_framework_data.sh > framework_data_v3.log 2>&1 &
tail -f framework_data_v3.log
```

成功产物：

```text
data/gsm8k_frameworks_v3.jsonl
data/gsm8k_frameworks_v3.generation-audit.json
data/gsm8k_frameworks_v3.audit.json
```

generation audit 必须满足：`status == complete`、`requested_valid == valid == semantic_passes == 1000`。`semantic_checks` 可以大于 1000，因为失败候选会重试。

### 2. 训练 v3 Framework Teacher

```bash
nohup ./run_teacher_training.sh > teacher_training_v3.log 2>&1 &
tail -f teacher_training_v3.log
```

成功产物：`outputs/teacher-framework-adapter-v3/`。

### 3. 运行 v3 单步训练冒烟

```bash
./run_smoke.sh
```

它分别运行 Vanilla 与 Guided 一个 optimizer step，检查答案停止、真实 token ID、框架重试/fallback、温度路由和蒸馏 loss。冒烟不代表准确率。

### 4. 重新训练两组 v3 Student

```bash
nohup ./run_comparison_training.sh > comparison_training_v3.log 2>&1 &
tail -f comparison_training_v3.log
```

输出：`outputs/vanilla-opd-v3/` 和 `outputs/guided-opd-v3/`。

框架生成温度固定为 `0.0`，与正式评测一致；Student solution rollout 仍使用 `0.7` 进行 on-policy 采样。旧 v2 adapter 是在答案后重复到 512 token 的 rollout 上训练的，不能代替这次重训。

### 5. 运行 v4 受控消融评测

```bash
nohup ./run_evaluation.sh > comparison_evaluation_v4.log 2>&1 &
tail -f comparison_evaluation_v4.log
```

输出目录：`outputs/comparison-eval-v4-framework-ablation/`。

双 A800 80GB 的评测配置为 `eval_devices: ["cuda:0", "cuda:1"]`，
`student_batch_size: 16` 和 `framework_batch_size: 16`，均为**每卡**批量。
两卡各加载完整模型副本并分摊题目，框架缓存、oracle 缓存以及每个 Student 条件依次执行；
同一阶段同时处理最多 32 题，不进行跨卡张量切分。每条答案单独停止，padding 不计入 token 成本。
显存不足时同时把两个 batch 参数改成 8；模型、提示、评分与 2048 输出上限不变。

批量评测的协议版本为 6，记录的单题 latency 是**整批耗时除以批量**，不是交互式单请求延迟；
跨 GPU 求和代表设备时间成本，不代表整体墙钟耗时。批量矩阵运算可能产生浮点数值差异，
不能承诺与旧逐题评测逐 token 一致。batch 和设备配置也属于实验签名，不应中途更换后混合结果。
旧逐题协议或旧源码结果不能直接 `--resume`；保留旧目录，使用新的输出目录重跑。
同版本同配置中断时可使用：

```bash
PYTHONPATH=src /root/blockdata/kv_cache_env/bin/python evaluate_comparison.py \
  --config configs/evaluation.json --resume
```

可运行 `PYTHONPATH=src:. /root/blockdata/kv_cache_env/bin/python tests/smoke_eval_batching.py`
做隔离的双卡吞吐与十单元流程检查。该脚本的两组标签都使用 Vanilla 权重作为测试夹具，
结果保存在新建临时目录，**不能用作正式 Guided 对照结论**。

每个 adapter 都评估以下五种条件：

| 条件 | System prompt | Framework 文本 | 是否读取测试答案 |
|---|---|---|---|
| `no_framework` | Vanilla | 无 | 否 |
| `empty_framework` | Framework-conditioned | 空 | 否 |
| `fallback_framework` | Framework-conditioned | 固定通用 fallback | 否 |
| `generated_framework` | Framework-conditioned | v3 Teacher 答案盲生成 | 否 |
| `oracle_framework` | Framework-conditioned | 4B 根据参考解答生成 | **是，仅诊断** |

因此可以分别估计：

- `empty − no`：仅 system instruction 的影响。
- `fallback/generated/oracle − empty`：在相同 system prompt 下，不同框架文本的影响。
- `guided − vanilla`：每个推理条件下的 adapter 效应。
- `guided_generated − vanilla_no`：完整可部署 Guided 系统相对传统基线的差异。

`framework_strata.json/csv/png` 固定报告 generated framework 的 valid/fallback 两个分层，并使用各自相同题目上的 empty-framework 结果作为配对基线。

## 主要评测产物

```text
framework_cache.jsonl
oracle_framework_cache.jsonl
predictions.jsonl
accuracy.csv
summary.json
paired_comparisons.json
paired_outcomes.csv
framework_strata.json
framework_strata.csv
grouped_accuracy.png
paired_deltas.png
paired_outcomes.png
accuracy_vs_cost.png
diagnostics.png
framework_strata_accuracy.png
run_manifest.json
```

主指标只接受最后一个非空物理行严格匹配 `#### number`。最后一个数字仅作为 relaxed 诊断。Student 正式上限为 2048 且代码禁止更大值；正常样本应由 answer-line stopping 提前结束。`diagnostics.png` 报告答案格式率、answer-line stopping rate、最大 token 截断率和平均 Student 输出长度。

## 文件职责

- `src/framework_opd/answer_stopping.py`：训练与评测共享的严格答案行语法和停止器。
- `src/framework_opd/framework_semantics.py`：用严格答案匹配执行验证候选框架。
- `prepare_framework_data.py`：生成、purity 检查、语义执行筛选并原子发布 v3 标签。
- `train_teacher.py`：验证 v3 audit 后训练 Framework Teacher LoRA。
- `src/framework_opd/rollout.py`：框架生成和答案感知 Student on-policy rollout。
- `train_opd.py`：Vanilla/Guided OPD 训练、checkpoint 和安全恢复。
- `evaluate_comparison.py`：五条件消融、配对统计、分层报告和绘图。
- `src/framework_opd/eval_batching.py`：仅用于评测的双卡批处理、逐样本停止、框架失败重试和 token 计数；不影响训练代码哈希。
- `configs/*_v3.json`：新训练与冒烟配置；旧 v2 配置和产物仅用于历史追溯。

## 测试

```bash
PYTHONPATH=src /root/blockdata/kv_cache_env/bin/python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/framework-opd-pycache \
  /root/blockdata/kv_cache_env/bin/python -m compileall -q \
  src train_opd.py train_teacher.py prepare_framework_data.py audit_framework_data.py evaluate_comparison.py
bash -n run_framework_data.sh run_teacher_training.sh run_smoke.sh run_comparison_training.sh run_evaluation.sh
```

当前仍是 single-seed pilot。论文级结论应至少运行 3–5 个独立 seed，报告 seed 间均值与方差；同一 checkpoint 的 paired bootstrap 不能替代训练随机性。
