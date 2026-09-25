import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from framework_opd.data import load_records
from framework_opd.rollout import FrameworkGenerationResult, generate_framework_result
from train_teacher import verify_teacher_artifact


DEFAULT_STUDENT_MODEL = "/root/eb-public/huggingface-models/Qwen/Qwen3-1.7B"
DEFAULT_TEACHER_MODEL = "/root/eb-public/huggingface-models/Qwen/Qwen3-4B"
DEFAULT_TEACHER_ADAPTER = "outputs/teacher-framework-adapter-v3"
DEFAULT_DATASET = (
    "/root/eb-public/huggingface-datasets/openai/gsm8k/main/"
    "test-00000-of-00001.parquet"
)
DEFAULT_OUTPUT = "outputs/framework-generator-comparison-v3.jsonl"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_identity(path: str | None) -> dict | None:
    if path is None:
        return None
    root = Path(path).resolve()
    weights = root / "adapter_model.safetensors"
    config = root / "adapter_config.json"
    if not weights.is_file():
        raise FileNotFoundError(f"adapter weights are missing: {weights}")
    if not config.is_file():
        raise FileNotFoundError(f"adapter config is missing: {config}")
    return {
        "path": str(root),
        "adapter_model_sha256": _sha256_file(weights),
        "adapter_config_sha256": _sha256_file(config),
    }


def _result_payload(result: FrameworkGenerationResult) -> dict:
    return {
        "framework": list(result.steps),
        "used_fallback": result.used_fallback,
        "attempts": result.attempts,
        "validation_errors": list(result.validation_errors),
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "hit_max_attempts": result.hit_max_attempts,
        "last_ended_with_eos": result.last_ended_with_eos,
        "closed_tag": result.closed_tag,
    }


def build_comparison_record(
    *,
    example_id: int,
    source_record: dict[str, str],
    student_result: FrameworkGenerationResult,
    teacher_result: FrameworkGenerationResult,
    run_config: dict,
) -> dict:
    return {
        "schema_version": 1,
        "example_id": example_id,
        "question": source_record["question"],
        "reference_answer": source_record["answer"],
        "answer_blind_generation": True,
        "run_config": run_config,
        "student": _result_payload(student_result),
        "teacher": _result_payload(teacher_result),
    }


def load_resume_prefix(
    output: Path, records: list[dict[str, str]], run_config: dict
) -> list[dict]:
    if not output.exists():
        return []
    loaded: list[dict] = []
    with output.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on output line {line_number}: {error.msg}"
                ) from error
            index = len(loaded)
            if index >= len(records):
                raise ValueError("resume output contains more records than the selected dataset")
            expected = records[index]
            if (
                row.get("example_id") != index
                or row.get("question") != expected["question"]
                or row.get("reference_answer") != expected["answer"]
            ):
                raise ValueError("resume output does not match the selected dataset prefix")
            if row.get("run_config") != run_config:
                raise ValueError("resume output uses a different run configuration")
            loaded.append(row)
    return loaded


def _seed_generation(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_base_model_and_tokenizer(path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answer-blind frameworks from a Student and an adapter-backed "
            "Framework Teacher for aligned comparison."
        )
    )
    parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--student-adapter")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-adapter", default=DEFAULT_TEACHER_ADAPTER)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    teacher_artifact = verify_teacher_artifact(args.teacher_adapter)
    configured_base = teacher_artifact["run_config"].get("base_model")
    if Path(str(configured_base)).resolve() != Path(args.teacher_model).resolve():
        raise ValueError("Teacher adapter base model does not match --teacher-model")

    run_config = {
        "student_model": str(Path(args.student_model).resolve()),
        "student_adapter": _adapter_identity(args.student_adapter),
        "teacher_model": str(Path(args.teacher_model).resolve()),
        "teacher_adapter": {
            "path": str(Path(args.teacher_adapter).resolve()),
            "run_id": teacher_artifact["run_config"]["run_id"],
            "adapter_artifact_sha256": teacher_artifact["adapter_artifact"]["sha256"],
        },
        "dataset": str(Path(args.dataset).resolve()),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "max_attempts": args.max_attempts,
        "temperature": args.temperature,
        "seed": args.seed,
        "prompt_type": "answer_blind_rollout_framework",
    }
    records = load_records(args.dataset, limit=args.limit)
    if len(records) != args.limit:
        raise ValueError(
            f"dataset provided {len(records)} records, fewer than requested {args.limit}"
        )
    completed = load_resume_prefix(output, records, run_config) if args.resume else []

    student, student_tokenizer = _load_base_model_and_tokenizer(
        args.student_model, args.student_device
    )
    if args.student_adapter:
        from peft import PeftModel

        student = PeftModel.from_pretrained(
            student, args.student_adapter, is_trainable=False
        )
        student.eval()
    teacher_base, teacher_tokenizer = _load_base_model_and_tokenizer(
        args.teacher_model, args.teacher_device
    )
    from peft import PeftModel

    teacher = PeftModel.from_pretrained(
        teacher_base, args.teacher_adapter, is_trainable=False
    )
    teacher.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for example_id in range(len(completed), len(records)):
            record = records[example_id]
            _seed_generation(args.seed + example_id)
            student_result = generate_framework_result(
                student,
                student_tokenizer,
                record["question"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_attempts=args.max_attempts,
            )
            _seed_generation(args.seed + example_id)
            teacher_result = generate_framework_result(
                teacher,
                teacher_tokenizer,
                record["question"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_attempts=args.max_attempts,
            )
            comparison = build_comparison_record(
                example_id=example_id,
                source_record=record,
                student_result=student_result,
                teacher_result=teacher_result,
                run_config=run_config,
            )
            stream.write(json.dumps(comparison, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                json.dumps(
                    {
                        "event": "framework_comparison_progress",
                        "completed": example_id + 1,
                        "requested": len(records),
                        "student_fallback": student_result.used_fallback,
                        "teacher_fallback": teacher_result.used_fallback,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(f"Comparison written to {output}", flush=True)


if __name__ == "__main__":
    main()
