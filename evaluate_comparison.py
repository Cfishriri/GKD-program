import argparse
import gc
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework_opd.data import load_records
from framework_opd.evaluation import score_prediction, summarize
from framework_opd.prompts import format_student_prompt, format_vanilla_student_prompt
from framework_opd.rollout import generate_framework


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0, encoded.input_ids.shape[1] :]
    return tokenizer.decode(output, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base, vanilla OPD, and framework-guided OPD")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["student_model"], local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = load_records(config["dataset"])
    random.Random(config["seed"]).shuffle(records)
    records = records[: config["limit"]]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    framework_teacher = None
    predictions = []
    summaries = []
    for variant in config["variants"]:
        if variant["guided"] and framework_teacher is None:
            framework_base = AutoModelForCausalLM.from_pretrained(
                config["teacher_model"], local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
            ).to(config["teacher_device"])
            framework_teacher = PeftModel.from_pretrained(
                framework_base, config["framework_teacher_adapter"], is_trainable=False
            )
            framework_teacher.eval()

        model = AutoModelForCausalLM.from_pretrained(
            config["student_model"], local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
        ).to(config["student_device"])
        if variant["adapter"]:
            model = PeftModel.from_pretrained(model, variant["adapter"], is_trainable=False)
        model.eval()
        variant_rows = []
        for index, record in enumerate(records):
            framework = []
            if variant["guided"]:
                framework = generate_framework(
                    framework_teacher,
                    tokenizer,
                    record["question"],
                    max_new_tokens=config["framework_max_new_tokens"],
                    temperature=0.0,
                )
                prompt = format_student_prompt(record["question"], framework)
            else:
                prompt = format_vanilla_student_prompt(record["question"])
            prediction = generate(model, tokenizer, prompt, config["max_new_tokens"])
            row = {
                "variant": variant["name"],
                "example_id": index,
                "question": record["question"],
                "reference": record["answer"],
                "framework": framework,
                "prediction": prediction,
                **score_prediction(prediction, record["answer"]),
            }
            variant_rows.append(row)
            predictions.append(row)
        summaries.append({"variant": variant["name"], **summarize(variant_rows)})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(summaries).to_csv(output_dir / "accuracy.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))

    names = [row["variant"] for row in summaries]
    accuracies = [row["accuracy"] for row in summaries]
    lower = [row["accuracy"] - row["accuracy_ci95_low"] for row in summaries]
    upper = [row["accuracy_ci95_high"] - row["accuracy"] for row in summaries]
    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, accuracies, yerr=[lower, upper], capsize=6, color=["#94a3b8", "#3b82f6", "#10b981"])
    axis.set_ylabel("GSM8K exact-match accuracy")
    axis.set_ylim(0, 1.12)
    axis.set_title(f"Controlled OPD comparison (n={len(records)}, 95% Wilson CI)")
    axis.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, summaries):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{row['accuracy']:.1%}\n{row['correct']}/{row['total']}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=180)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
