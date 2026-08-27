# Framework-Guided OPD

本项目在本地 Qwen3 模型与 GSM8K 上比较传统 OPD 和 Framework-Guided OPD。所有模型与数据均从服务器本地路径读取，不进行下载。

## 当前 v3/v4 实验链路

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

## 从头运行新实验

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
