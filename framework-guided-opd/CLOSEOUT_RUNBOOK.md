# 人工确认框架的四组收尾实验

工作目录：`/root/blockdata/framework-guided-opd/framework-guided-opd`。

## 当前入口（2026-09-25，50题版本）

本文件下方的60训练/20测试说明属于已归档的旧轮次；当前`run_closeout.sh`指向`outputs/closeout-human36-test50-20260925-v1/`。100条人工标注中86条通过人工正确、结构及审核否决筛选，14条排除。当前为36条训练、50条测试（旧20条测试全部保留），两个student重新从原基座训练，GPU0普通OPD、GPU1框架OPD；每组50题分别有/无框架推理，共200条。已完成运行`outputs/closeout-full-FZgvRxMy/`，无需重复启动。

新增`report/prediction_tokens.csv`逐题记录prompt、实际生成、最终答案文本token数及正确性；`report/summary.json`按四组给出token总数、均值、最小值、中位数、P95、最大值；`report/REPORT.md`给出主要准确率与答案token摘要。答案token用student tokenizer对最终保存文本重新分词，可能小于实际生成数。生成上限训练1024、测试2048。17项定向单测和双GPU的2题真实容量测试通过；完整实验两worker/report退出0，其中guided无框架一题触及2048。

如确需重新运行（会从头训练，不是继续旧模型），确认两张A800空闲，并换用未存在的新日志名：

```bash
cd /root/blockdata/framework-guided-opd/framework-guided-opd
nvidia-smi
set -o noclobber
nohup bash /root/blockdata/framework-guided-opd/framework-guided-opd/run_closeout.sh full \
  > /root/blockdata/framework-guided-opd/framework-guided-opd/outputs/closeout_50_repeat_20260925.log 2>&1 &
echo $!
tail -f /root/blockdata/framework-guided-opd/framework-guided-opd/outputs/closeout_50_repeat_20260925.log
```

输入为上述冻结集，输出为日志第一行打印的新`outputs/closeout-full-XXXXXXXX/`；查看该目录`worker_exit_codes.json`、`report_exit_code.json`及`report/`。根据本次两路实际536.85/655.58秒、双GPU并行，预计约10～25分钟，单worker上限3小时。`Ctrl+C`只停止tail。已完成本轮无需再跑。

## 历史 60训练/20测试版本（仅供追溯）

- 原100条人工批注与2026-09-22补充裁决，绑定原题、答案、框架内容；原始材料不修改。
- 仅最终人工判断正确、结构检查通过且所纳入审核证据均未否决的框架入选。自动accepted不是人工确认的替代。
- 对旧审核证据采用保守否决并集，已知误拒也暂跳过，不据此声称审核器准确率。
- 100条中86条符合条件、14条排除。固定种子20260923选60条训练、20条测试、6条不使用。完整名单和来源hash在冻结目录。
- 这是已参与审核器开发的题目子集，不是独立未见测试集；框架来自参考答案辅助生成/人工筛选，属于oracle框架条件。训练/测试题目互不重叠，但不宣称对预训练数据去污染。
- 新建两组student LoRA，从原始Qwen3-1.7B开始；不载入旧checkpoint、旧student或框架teacher适配器。不重新训练框架生成器，直接复用已人工确认的固定框架。
- student自行rollout，冻结的原始Qwen3-4B提供同一completion的token分布监督；reverse KL只在生成位置计算。两组均lr1e-5、r8/alpha16、dropout0、累积4条、1轮60条（15次更新）。这是短程诊断，不保证训练收敛。
- 两组统一使用原生聊天模板，enable_thinking=false；要求输出计算出的数值，不使用字面量`#### number`占位示例。这是本轮独立入口的新提示协议，与旧实验raw提示不同，不作跨协议指标直接对比。
- guided有固定框架；vanilla无框架；两者训练题目、顺序、种子、预算相同。模型提示不含参考答案。
- 每个模型均做有/无框架推理，共四组，每组完全相同20题。两个有框架组使用相同框架，不在线生成/替换。
- 训练单次生成1024 token、temperature0.7；评测2048 token、贪心、batch4。完整`#### number`答案行提前停止。student错误、格式错误和达到上限都保留在准确率分母。
- `training_eligible=false`的原实验产物不修改；新数据以用户确认的人工批注作为准入依据，审核输出只提供额外否决证据。

## 文件

- `closeout_experiment.py`：prepare冻结材料、worker训练并评测一个模型、report核对四组后出报告。
- `run_closeout.sh`：GPU0普通OPD、GPU1框架OPD并行运行；每卡同时加载teacher和student。检查显存是否已占用，不自动终止其他任务。
- `test_closeout.py`：数据门禁、绑定、防篡改、四组完整性测试。
- `closeout_inputs_20260923/`：人工批注和补充意见的只读用途副本。
- `outputs/closeout-human60-test20-20260923-v3/`：当时的冻结集，包括train/test/skipped、绑定补充意见、原始输入副本、manifest。
- 无v3后缀的早期冻结版本仅保留溯源，不再用于启动；v2对应旧raw提示，v3是统一聊天模板。数据划分相同。
- 原`train_opd.py`、`evaluate_comparison.py`和`src`均未修改。

## 历史启动记录（不要直接复用以下旧日志名）

以下是旧60/20轮次曾使用的日志路径；当前脚本已改为36/50输入，复跑须使用文件顶部的当前命令及新的日志名。

```bash
cd /root/blockdata/framework-guided-opd/framework-guided-opd
nvidia-smi
set -o noclobber
nohup bash /root/blockdata/framework-guided-opd/framework-guided-opd/run_closeout.sh full \
  > /root/blockdata/framework-guided-opd/framework-guided-opd/outputs/closeout_full_20260925.log 2>&1 &
echo $!
tail -f /root/blockdata/framework-guided-opd/framework-guided-opd/outputs/closeout_full_20260925.log
```

`Ctrl+C`只退出tail，不会停止后台任务。主日志开头打印本次绝对`RUN_DIR`，无需依赖旧SSH会话变量。每次运行新建`outputs/closeout-full-XXXXXXXX/`，但主日志受noclobber保护；已有同名日志时请换新日志名，不覆盖。脚本每个worker硬超时3小时，失败保留证据、不自动重跑。

旧60/20实验原预计15～40分钟，两张GPU并行；该估算不适用于当前36/50脚本。本轮实测及复跑估时见文件顶部。

## 查看结果

主日志会打印真实结果目录。该目录内：

- `vanilla.log`、`guided.log`：训练及评测进度；加载模型时可能暂时无step输出。
- `worker_exit_codes.json`：两路退出码，均0才继续汇总。
- `vanilla/`、`guided/`：各自adapter、逐步training.jsonl、predictions.jsonl和run.json（包括实际耗时、显存峰值、指纹）。
- `report_exit_code.json`：报告退出码，0才代表报告完成。
- `report/REPORT.md`：四组正确数/总数、准确率、95%区间、截断数。
- `report/summary.json`：机器可读结果、筛选覆盖、跳过原因及配置。
- `report/paired_predictions.jsonl`：每题题目、正确答案、框架和四组完整输出。
- `report/accuracy.svg`：四组准确率对比图。

全部四组预测齐全且题目/框架/模型/配置指纹一致后才出完整报告；不把缺失输出变成“跳过的框架”。排除原因可能重叠，不应把原因计数直接相加作为排除题数。

## 功能测试

```bash
cd /root/blockdata/framework-guided-opd/framework-guided-opd
/root/blockdata/kv_cache_env/bin/python -B -m unittest test_closeout -v
bash /root/blockdata/framework-guided-opd/framework-guided-opd/run_closeout.sh smoke
```

单测CPU约1秒；smoke每路仅2训练/2测试、128 token，预计2～8分钟、每路20分钟硬超时。该结果只验证功能，不用于比较训练方法优劣。

`bash run_closeout.sh capacity`同样仅2训练/2测试，但采用完整1024/2048预算、batch4，预计3～8分钟，20分钟硬超时。报告仍标为smoke，不与完整实验混淆。

## 验证记录

- 首次`outputs/closeout-smoke-VsESDyja/`失败在CUDA初始化前的显存统计，未进入训练；修复后保留失败记录。
- `outputs/closeout-smoke-2vnDNYrL/`两路及报告退出0，约149秒，128token导致8个测试输出均截断，仅证明流程。
- `outputs/closeout-capacity-ciAjt1yl/`采用1024/2048预算，两路及报告退出0，vanilla180秒/guided215秒；峰值allocated12.32/16.87GiB。2题无框架组均2/2、有框架分别1/2和0/2，其中3个有框架输出达到2048，原始响应反复复制`#### number`，不能作为最终方法效果。
- 因上述提示问题，v3仅统一两组聊天模板/数值答案指令；不改变框架、划分、训练预算或评分；旧结果保留。最终v3验证另行记录。
- 2026-09-25确认`outputs/closeout-capacity-5b2xPm34/`完成：最终v3双worker和报告退出0，四组各2/2、无测试截断；耗时47.59/50.08秒，峰值allocated11.49/11.78GiB。不是效果结论，且这两道题已经用于提示调试。
