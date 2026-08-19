# Framework-Guided OPD

This project implements the requested three-stage experiment without downloading models or datasets.

## Algorithm

1. Use each GSM8K question and reference answer as privileged context to create a numbered framework label.
2. LoRA-train Qwen3-4B to predict that framework from the question alone.
3. During OPD, the frozen teacher emits a framework, Qwen3-1.7B generates a solution under it, and generalized JSD is minimized only on student-generated solution tokens.

The question and framework are context, not on-policy targets. The implementation uses the same generalized JSD definition as the local TRL GKD trainer. `beta=0` is forward KL, `beta=1` is reverse KL, and `beta=0.5` is symmetric generalized JSD.

## Local resources

```text
Student: /root/eb-public/huggingface-models/Qwen/Qwen3-1.7B
Teacher: /root/eb-public/huggingface-models/Qwen/Qwen3-4B
GSM8K:  /root/eb-public/huggingface-datasets/openai/gsm8k/main/train-00000-of-00001.parquet
Python: /root/blockdata/kv_cache_env/bin/python
```

Every command should set offline mode:

```bash
cd /root/blockdata/framework-guided-opd
export PYTHONPATH=$PWD/src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

## 1. Build framework labels

This optional bootstrap uses the answer only while creating labels. A manually curated JSONL with the same schema is preferable for a formal experiment.

```bash
/root/blockdata/kv_cache_env/bin/python prepare_framework_data.py \
  --model /root/eb-public/huggingface-models/Qwen/Qwen3-4B \
  --dataset /root/eb-public/huggingface-datasets/openai/gsm8k/main/train-00000-of-00001.parquet \
  --output data/gsm8k_frameworks.jsonl \
  --limit 1000
```

Each output record contains `question`, `answer`, and `framework: list[str]`.

## 2. Train the framework teacher

```bash
/root/blockdata/kv_cache_env/bin/python train_teacher.py \
  --model /root/eb-public/huggingface-models/Qwen/Qwen3-4B \
  --data data/gsm8k_frameworks.jsonl \
  --output outputs/teacher-framework-adapter \
  --steps 1000
```

The checked-in `configs/guided_opd.json` already points to this adapter as
`framework_teacher_adapter`. The framework teacher is separate from the frozen base Qwen3-4B used to score OPD tokens,
so both experiment arms receive logits from the same scoring teacher.

## 3. Run OPD

The checked-in smoke configuration performs one optimizer step:

```bash
./run_smoke.sh
```

Increase `max_steps` and `gradient_accumulation_steps` in a copied JSON config for a real run. Metrics and the student LoRA adapter are written below the configured `output_dir`.

## Verification

```bash
PYTHONPATH=src /root/blockdata/kv_cache_env/bin/python -m unittest discover -s tests -v
/root/blockdata/kv_cache_env/bin/python -m compileall -q src train_opd.py train_teacher.py prepare_framework_data.py
```

## Current limitations

- Teacher and student must share the Qwen3 tokenizer and vocabulary.
- The reference trainer uses batch size one and transfers full teacher logits between GPUs.
- Framework quality depends on curated or answer-privileged framework labels.
- The smoke run validates mechanics, not downstream accuracy.

## Controlled comparison: what to run next

Framework-label generation is complete when this prints `1000`:

```bash
wc -l data/gsm8k_frameworks.jsonl
```

Run the following commands from `/root/blockdata/framework-guided-opd`. Long commands are shown with `nohup` so an SSH
disconnect does not stop them.

### A. Train the framework teacher

```bash
nohup ./run_teacher_training.sh > teacher_training.log 2>&1 &
tail -f teacher_training.log
```

Completion evidence:

```bash
test -f outputs/teacher-framework-adapter/adapter_model.safetensors && echo teacher-ready
```

### B. Train both matched OPD arms

The two configurations differ only in rollout mode, output path, and the guided-only framework generator. The Student,
scoring Teacher, data order, seed, steps, LoRA, generation budget, temperature, and GKD loss are identical.

```bash
nohup ./run_comparison_training.sh > comparison_training.log 2>&1 &
tail -f comparison_training.log
```

This runs traditional OPD first and framework-guided OPD second. Completion evidence:

```bash
test -f outputs/vanilla-opd/student_adapter/adapter_model.safetensors && echo vanilla-ready
test -f outputs/guided-opd/student_adapter/adapter_model.safetensors && echo guided-ready
```

If separate jobs are preferred, run:

```bash
PYTHONPATH=src TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /root/blockdata/kv_cache_env/bin/python train_opd.py --config configs/vanilla_opd.json

PYTHONPATH=src TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /root/blockdata/kv_cache_env/bin/python train_opd.py --config configs/guided_opd.json
```

Do not reuse a non-empty output directory: the trainer refuses to append new metrics to an old run.

### C. Evaluate and generate visual evidence

```bash
nohup ./run_evaluation.sh > comparison_evaluation.log 2>&1 &
tail -f comparison_evaluation.log
```

The fixed, seeded 500-example GSM8K evaluation creates:

```text
outputs/comparison-eval/predictions.jsonl        per-question predictions and correctness
outputs/comparison-eval/accuracy.csv             accuracy, correct/total, 95% Wilson interval
outputs/comparison-eval/summary.json             machine-readable aggregate results
outputs/comparison-eval/accuracy_comparison.png  annotated accuracy bar chart
```

The primary decision metric is exact-match accuracy. The chart also displays 95% Wilson confidence intervals; overlapping
intervals mean the observed ranking should not automatically be treated as conclusive. `answer_format_rate` is reported as
a diagnostic rather than the primary score.
