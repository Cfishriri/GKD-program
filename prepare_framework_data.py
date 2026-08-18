import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework_opd.data import load_records
from framework_opd.prompts import extract_framework, format_framework_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Create answer-privileged framework labels")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as stream:
        for record in load_records(args.dataset, args.limit):
            prompt = format_framework_prompt(record["question"], record["answer"])
            encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0, encoded.input_ids.shape[1] :]
            text = "<framework>\n" + tokenizer.decode(generated, skip_special_tokens=True)
            try:
                framework = extract_framework(text)
            except ValueError:
                continue
            stream.write(json.dumps({**record, "framework": framework}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

