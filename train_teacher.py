import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework_opd.prompts import format_rollout_framework_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3 teacher to emit solution frameworks")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
    ).to(args.device)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    with open(args.data, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError("framework training data is empty")

    model.train()
    for step in range(args.steps):
        record = records[step % len(records)]
        prompt = format_rollout_framework_prompt(record["question"])
        target = "\n".join(f"{index}. {item}" for index, item in enumerate(record["framework"], 1))
        target = target + "\n</framework>"
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt_ids + target_ids], device=args.device)
        labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], device=args.device)
        loss = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"step": step + 1, "loss": loss.item()}))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()

