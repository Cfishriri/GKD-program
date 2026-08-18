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

Set `teacher_adapter` in the OPD JSON config to load this adapter.

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
