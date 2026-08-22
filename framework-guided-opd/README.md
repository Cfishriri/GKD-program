# Framework-Guided OPD

本项目在本地模型和 GSM8K 数据上实现一组受控对照实验：传统 OPD 与“Teacher 先给抽象解题框架、Student 再按框架作答”的 OPD。程序只读取服务器已有资源，不下载模型或数据。

## 实验结构

1. 4B Teacher 在“题目 + 参考解答”的特权信息下生成抽象框架标签。
2. 用通过纯度检查的标签对 4B Teacher 做 LoRA SFT，使其只看题目也能输出框架。
3. 分别训练相同初始化和超参数的 Vanilla OPD、Framework-Guided OPD Student。
4. 在相同题目上进行 `adapter × inference framework` 的 2×2 配对评估。

OPD 的评分 Teacher 始终是冻结的 Qwen3-4B base。框架 Teacher 是另一份加载了框架 LoRA 的 Qwen3-4B，二者职责不同。Student 只在自己真实采样出的 solution token 上计算蒸馏损失；prompt 和框架 token 不参与 loss。

当前对照配置使用 `beta=1.0`，对应传统 reverse-KL OPD。`beta=0.0` 是 forward KL；中间值才是 generalized JSD。

## 服务器资源

```text
项目:    /root/blockdata/framework-guided-opd
Student: /root/eb-public/huggingface-models/Qwen/Qwen3-1.7B
Teacher: /root/eb-public/huggingface-models/Qwen/Qwen3-4B
GSM8K:  /root/eb-public/huggingface-datasets/openai/gsm8k/main/
Python: /root/blockdata/kv_cache_env/bin/python
```

脚本已经设置 `PYTHONPATH` 和 Hugging Face 离线模式，应从项目目录运行。

## 重要数据门禁

旧文件 `data/gsm8k_frameworks.jsonl` 含大量数值、计算结果和最终答案，不能用于框架 Teacher 训练。它被保留用于追溯，但新的安全链路只使用：

```text
data/gsm8k_frameworks_v2.jsonl
```

每条 v2 框架必须满足：2–6 个非空步骤；正文不含数字/数字词/Unicode 数字符号、`####`、已求值等式或规范化后的最终答案。描述“取一半、加总、按频率相乘”等必要关系仍然允许，但不能给出算得的值。生成时会有限重试，写盘后还会再次独立全量审计。Teacher SFT 在加载模型和占用显存之前也会执行同一门禁；任一违规记录都会让训练立即失败。

旧 `outputs/teacher-framework-adapter`、`outputs/vanilla-opd`、`outputs/guided-opd` 也不会复用。新链路全部写入带 `-v2` 的目录，并以 `run_config`、内容指纹和 `RUN_COMPLETE` 共同证明产物身份；仅有一个权重文件不算完成。

## 从当前进度继续运行

### 1. 重新生成安全框架标签

你之前生成的旧标签不能继续使用。下一步应先运行新的 v2 生成脚本：

```bash
cd /root/blockdata/framework-guided-opd
nohup ./run_framework_data.sh > framework_data_v2.log 2>&1 &
tail -f framework_data_v2.log
```

成功条件：

```bash
wc -l data/gsm8k_frameworks_v2.jsonl
/root/blockdata/kv_cache_env/bin/python -c \
  'import json; a=json.load(open("data/gsm8k_frameworks_v2.audit.json")); assert a["total"] == 1000 and a["invalid"] == 0; print(a)'
```

`generation-audit.json` 中的 rejected source 数量可以大于零，因为生成器会继续扫描其他题目直到收集 1000 条合格标签；真正的训练门禁是 generation audit 的 `status == "complete"`、`requested_valid == valid == 1000`，并且最终 purity audit 的 `total == valid == 1000`。若扫描完仍不足，程序只保留 `.partial` 与 `status="incomplete"` 的诊断，不会发布正式 v2 文件；归档这些失败证据后再重跑。

### 2. 训练框架 Teacher

```bash
nohup ./run_teacher_training.sh > teacher_training.log 2>&1 &
tail -f teacher_training.log
```

成功条件：

```bash
PYTHONPATH=src /root/blockdata/kv_cache_env/bin/python -c \
  'from train_teacher import verify_teacher_artifact; verify_teacher_artifact("outputs/teacher-framework-adapter-v2"); print("teacher-ready")'
```

### 3. 先做两组单步冒烟

```bash
./run_smoke.sh
```

它会分别跑 Vanilla 和 Guided 各一个 optimizer step，并记录真实生成 token 数、EOS、截断、框架重试和 fallback。冒烟只验证代码链路，不代表最终准确率。

### 4. 训练两组单 seed OPD pilot

```bash
nohup ./run_comparison_training.sh > comparison_training.log 2>&1 &
tail -f comparison_training.log
```

脚本先跑 Vanilla，后跑 Guided；只有 `RUN_COMPLETE`、最终 adapter、完成时权重哈希、manifest 和与当前配置一致的 `run_config` 同时通过时才跳过。训练每隔若干 rollout 在 optimizer boundary 保存 checkpoint，最后一个不足完整 accumulation window 的窗口也按实际 completion token 总数归一化。checkpoint、metrics 和 manifest 共享同一 `run_id`，并绑定数据、模型/adapter 与实现源码指纹，不能跨输出目录拼接恢复。若恰好在最终 adapter 发布后、完成标志写完前中断，使用最终 checkpoint 恢复会只重建并提交完成状态，不会重复已完成的 rollout。

若中断，在相应 JSON 配置中把 `resume_from_checkpoint` 改为最后一个完整 checkpoint 目录后重启。例如：

```json
"resume_from_checkpoint": "/root/blockdata/framework-guided-opd/outputs/vanilla-opd-v2/checkpoints/checkpoint-000128"
```

### 5. 运行 2×2 配对评估并生成图

```bash
nohup ./run_evaluation.sh > comparison_evaluation.log 2>&1 &
tail -f comparison_evaluation.log
```

若评估中断，完整 cell 和框架缓存会保留；在配置与数据/adapter 指纹未变化时可安全续跑：

```bash
PYTHONPATH=src TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /root/blockdata/kv_cache_env/bin/python evaluate_comparison.py \
  --config configs/evaluation.json --resume
```

四个核心 cell 对同一批题目评估，并且两个 `with_framework` cell 逐题复用完全相同的已验证 4B 框架：

| Cell | Student adapter | 推理时框架 |
|---|---|---|
| `vanilla_no_framework` | Vanilla | 无 |
| `guided_no_framework` | Guided | 无 |
| `vanilla_with_framework` | Vanilla | 同一份缓存框架 |
| `guided_with_framework` | Guided | 同一份缓存框架 |

这样可以分别观察：

- `guided_no_framework − vanilla_no_framework`：不加测试时 4B 辅助的训练方法差异。
- `guided_with_framework − vanilla_with_framework`：相同 scaffold 下的训练方法差异。
- 两组 `with − no`：框架在推理阶段带来的收益。
- `guided_with_framework − vanilla_no_framework`：完整系统收益，但必须连同额外 4B 成本解释。
- 2×2 interaction：Guided 训练是否让 Student 更会利用框架。

主指标只接受 completion 的最后一个非空物理行严格匹配 `#### number`；“最后一个数字”只作为 relaxed 诊断，不参与胜负判断。框架必须先出现完整 `</framework>` 且标签内通过纯度校验；若模型在闭合标签后继续解释直到 token 上限，只提取闭合标签内的步骤，同时仍记录该次 hit-max 成本，不会把标签外文字传给 Student。程序同时报告配对 accuracy delta、bootstrap 95% CI、discordant pairs、exact McNemar p 值、合法答案格式率、Student/Framework Teacher 的 EOS 与截断、框架 fallback，以及两阶段 prompt/output token 和耗时。`accuracy_vs_cost.png` 使用明确标注的 token proxy；它不会把 4B 调用说成免费。

实测未训练充分的 Student 在 256 token 内经常来不及给出合法 marker，因此正式训练 cap 已提高到 512，评估 cap 提高到 1024，并在每个 cell 强制报告截断率。若正式 pilot 仍有明显截断，不能解释准确率排名，应使用新的输出目录提高 cap 后重跑。

主要证据文件：

```text
outputs/comparison-eval-v2/framework_cache.jsonl
outputs/comparison-eval-v2/predictions.jsonl
outputs/comparison-eval-v2/accuracy.csv
outputs/comparison-eval-v2/summary.json
outputs/comparison-eval-v2/paired_comparisons.json
outputs/comparison-eval-v2/paired_outcomes.csv
outputs/comparison-eval-v2/grouped_accuracy.png
outputs/comparison-eval-v2/paired_deltas.png
outputs/comparison-eval-v2/paired_outcomes.png
outputs/comparison-eval-v2/accuracy_vs_cost.png
outputs/comparison-eval-v2/diagnostics.png
outputs/comparison-eval-v2/run_manifest.json
```

只有在框架 purity 为零泄漏、Student 与 Framework Teacher 的截断/fallback 可接受，并且预设比较的配对 95% CI 排除零时，才把单次评估称为“这一 checkpoint 更好”。当前脚本明确是 single-seed pilot；paired bootstrap 只反映同一 checkpoint 在题目上的不确定性，不能替代训练随机性。论文级结论应复制两份匹配配置，以至少 3–5 个 seed 和互不覆盖的 `*-v2-seedNN` 输出目录重复训练、逐 seed 评估，再报告 seed 间均值和方差。

## 文件职责

- `prepare_framework_data.py`：生成、重试并原子写入合格框架标签。
- `audit_framework_data.py`：独立全量审计 JSONL，违规时返回非零状态。
- `train_teacher.py`：对 4B 框架 Teacher 做 LoRA SFT。
- `train_opd.py`：训练 Vanilla 或 Guided Student，支持 checkpoint/resume。
- `evaluate_comparison.py`：共享框架缓存、2×2 推理、配对统计和制图。
- `src/framework_opd/framework_validation.py`：训练前和在线 rollout 共用的框架纯度规则。
- `src/framework_opd/rollout.py`：框架生成与 Student on-policy rollout，保留真实 token ID。
- `src/framework_opd/loss.py`：completion-token 蒸馏损失及 token-global 归一化统计。
- `src/framework_opd/evaluation.py`：严格答案解析、Wilson 区间、paired bootstrap 与 McNemar。
- `configs/`：两组匹配训练配置、冒烟配置和评估配置。
- `run_*.sh`：按依赖顺序封装的服务器运行入口。

## 测试

```bash
cd /root/blockdata/framework-guided-opd
PYTHONPATH=src /root/blockdata/kv_cache_env/bin/python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/framework-opd-pycache \
  /root/blockdata/kv_cache_env/bin/python -m compileall -q \
  src train_opd.py train_teacher.py prepare_framework_data.py audit_framework_data.py evaluate_comparison.py
bash -n run_framework_data.sh run_teacher_training.sh run_smoke.sh run_comparison_training.sh run_evaluation.sh
test -x run_framework_data.sh -a -x run_teacher_training.sh -a -x run_smoke.sh -a -x run_comparison_training.sh -a -x run_evaluation.sh
```

Teacher 与 Student 必须共享 tokenizer 和 vocabulary；当前服务器上的两个 Qwen3 模型已核对一致。
