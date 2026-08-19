import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework_opd.data import load_records
from framework_opd.loss import generalized_jsd_loss
from framework_opd.masking import causal_completion_mask
from framework_opd.rollout import generate_opd_rollout, validate_mode


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-guided on-policy distillation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    mode = validate_mode(config.get("mode", "guided"))

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["student_model"], local_files_only=True, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        config["teacher_model"], local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
    ).to(config["teacher_device"])
    teacher.eval()
    teacher.requires_grad_(False)

    framework_teacher = teacher
    if mode == "guided" and config.get("framework_teacher_adapter"):
        framework_base = AutoModelForCausalLM.from_pretrained(
            config["teacher_model"], local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
        ).to(config["teacher_device"])
        framework_teacher = PeftModel.from_pretrained(
            framework_base, config["framework_teacher_adapter"], is_trainable=False
        )
        framework_teacher.eval()
        framework_teacher.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(
        config["student_model"], local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
    ).to(config["student_device"])
    student.config.use_cache = False
    student = get_peft_model(
        student,
        LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"])
    records = load_records(config["dataset"])
    random.shuffle(records)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to mix runs in existing metrics file: {metrics_path}")
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    optimizer.zero_grad(set_to_none=True)

    for step, record in enumerate(records[: config["max_steps"]], 1):
        student.eval()
        rollout = generate_opd_rollout(
            framework_teacher,
            student,
            tokenizer,
            record["question"],
            mode=mode,
            framework_max_new_tokens=config["framework_max_new_tokens"],
            solution_max_new_tokens=config["solution_max_new_tokens"],
            temperature=config["generation_temperature"],
        )
        student.train()
        student_ids = rollout.input_ids.to(config["student_device"])
        student_mask = rollout.attention_mask.to(config["student_device"])
        student_logits = student(input_ids=student_ids, attention_mask=student_mask).logits[:, :-1]

        with torch.no_grad():
            teacher_ids = rollout.input_ids.to(config["teacher_device"])
            teacher_mask = rollout.attention_mask.to(config["teacher_device"])
            teacher_logits = teacher(input_ids=teacher_ids, attention_mask=teacher_mask).logits[:, :-1]
            teacher_logits = teacher_logits.to(config["student_device"])

        completion_mask = causal_completion_mask(rollout.labels).to(config["student_device"])
        loss, loss_metrics = generalized_jsd_loss(
            student_logits,
            teacher_logits,
            completion_mask,
            beta=config["beta"],
            temperature=config["temperature"],
        )
        (loss / config["gradient_accumulation_steps"]).backward()
        if step % config["gradient_accumulation_steps"] == 0 or step == config["max_steps"]:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metrics = {
            "step": step,
            "mode": mode,
            "loss": loss_metrics["loss"].item(),
            "num_tokens": int(loss_metrics["num_tokens"].item()),
            "framework": rollout.framework,
            "completion": rollout.completion,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(json.dumps({key: value for key, value in metrics.items() if key != "completion"}, ensure_ascii=False))
        del student_logits, teacher_logits, loss

    student.save_pretrained(output_dir / "student_adapter")
    tokenizer.save_pretrained(output_dir / "student_adapter")


if __name__ == "__main__":
    main()
