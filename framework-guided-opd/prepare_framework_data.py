import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from framework_opd.data import load_records
from framework_opd.framework_validation import LEAKAGE_REASONS, validate_framework
from framework_opd.framework_semantics import verify_framework_semantics
from framework_opd.prompts import extract_framework, format_framework_prompt, has_complete_framework


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = stream.name
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_generation_result(
    *,
    output: Path,
    audit_output: Path,
    accepted_records: list[dict],
    audit: dict,
    complete: bool,
) -> dict:
    """Publish a complete dataset atomically, or retain only a fixed partial artifact."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing formal output: {output}")
    if audit.get("valid") != len(accepted_records):
        raise ValueError("audit valid count does not match accepted records")
    expected_complete = audit.get("requested_valid") == len(accepted_records)
    if complete != expected_complete:
        raise ValueError("completion status does not match requested and accepted record counts")

    partial_output = Path(str(output) + ".partial")
    _atomic_write_jsonl(partial_output, accepted_records)
    data_sha256 = _sha256_file(partial_output)
    final_audit = dict(audit)
    final_audit["status"] = "complete" if complete else "incomplete"
    final_audit["data_sha256"] = data_sha256

    if complete:
        os.replace(partial_output, output)
        final_audit["output_sha256"] = data_sha256
        final_audit["partial_output"] = None
        final_audit["partial_sha256"] = None
    else:
        final_audit["output_sha256"] = None
        final_audit["partial_output"] = str(partial_output)
        final_audit["partial_sha256"] = data_sha256

    _atomic_write_json(audit_output, final_audit)
    return final_audit


def _is_eos(token_id: int, eos_token_id: int | list[int] | tuple[int, ...] | None) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, int):
        return token_id == eos_token_id
    return token_id in eos_token_id


def _print_progress(*, source_examined: int, valid: int, requested_valid: int, attempts: int) -> None:
    print(
        json.dumps(
            {
                "event": "framework_generation_progress",
                "source_examined": source_examined,
                "valid": valid,
                "requested_valid": requested_valid,
                "attempts_used": attempts,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create answer-privileged framework labels")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--semantic-max-new-tokens", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--audit-output",
        help="Audit JSON path (default: <output stem>.audit.json)",
    )
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if not 1 <= args.semantic_max_new_tokens <= 2048:
        parser.error("--semantic-max-new-tokens must be between 1 and 2048")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")

    output = Path(args.output)
    audit_output = Path(args.audit_output) if args.audit_output else output.with_name(output.stem + ".audit.json")
    if output.exists():
        parser.error(f"formal output already exists; refusing to overwrite: {output}")

    source_records = load_records(args.dataset)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    accepted_records: list[dict] = []
    invalid_examples: list[dict] = []
    reason_counts: Counter[str] = Counter()
    attempt_reason_counts: Counter[str] = Counter()
    leakage_records = 0
    attempts_used = 0
    prompt_tokens = 0
    generated_tokens = 0
    hit_max_attempts = 0
    closed_tag_attempts = 0
    ended_with_eos_attempts = 0
    semantic_checks = 0
    semantic_passes = 0
    semantic_failures = 0
    semantic_prompt_tokens = 0
    semantic_output_tokens = 0
    semantic_hit_max_tokens = 0
    semantic_answer_stops = 0

    source_examined = 0
    for index, record in enumerate(source_records):
        if len(accepted_records) >= args.limit:
            break
        source_examined += 1
        retry_reasons: tuple[str, ...] = ()
        observed_reasons: set[str] = set()
        framework: list[str] | None = None
        semantic_verification: dict | None = None
        for _ in range(args.max_attempts):
            attempts_used += 1
            prompt = format_framework_prompt(
                record["question"],
                record["answer"],
                retry_reasons=retry_reasons,
            )
            encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
            prompt_tokens += int(encoded.input_ids.shape[1])
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                generation_kwargs.update(temperature=args.temperature, top_p=0.9)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    **generation_kwargs,
                )[0, encoded.input_ids.shape[1] :]
            token_ids = [int(token_id) for token_id in generated.detach().cpu().tolist()]
            generated_tokens += len(token_ids)
            ended_with_eos = bool(token_ids and _is_eos(token_ids[-1], tokenizer.eos_token_id))
            ended_with_eos_attempts += int(ended_with_eos)
            hit_max_tokens = len(token_ids) >= args.max_new_tokens and not ended_with_eos
            hit_max_attempts += int(hit_max_tokens)
            text = "<framework>\n" + tokenizer.decode(token_ids, skip_special_tokens=True)
            closed_tag = has_complete_framework(text)
            closed_tag_attempts += int(closed_tag)
            generation_reasons: list[str] = []
            if not closed_tag:
                if hit_max_tokens:
                    generation_reasons.append("hit_max_tokens")
                generation_reasons.append("framework_not_closed")
            if generation_reasons:
                retry_reasons = tuple(generation_reasons)
                observed_reasons.update(retry_reasons)
                attempt_reason_counts.update(retry_reasons)
                continue
            try:
                candidate = extract_framework(text, require_closed=True)
            except ValueError:
                retry_reasons = ("parse_error",)
                observed_reasons.update(retry_reasons)
                attempt_reason_counts.update(retry_reasons)
                continue
            result = validate_framework(candidate, record["answer"])
            if result.valid:
                verification = verify_framework_semantics(
                    model,
                    tokenizer,
                    record["question"],
                    list(result.steps),
                    record["answer"],
                    args.semantic_max_new_tokens,
                )
                semantic_checks += 1
                semantic_prompt_tokens += verification.prompt_tokens
                semantic_output_tokens += verification.generated_tokens
                semantic_hit_max_tokens += int(verification.hit_max_tokens)
                semantic_answer_stops += int(verification.stopped_on_answer)
                if verification.correct:
                    semantic_passes += 1
                    framework = list(result.steps)
                    semantic_verification = {
                        "correct": True,
                        "predicted_answer": verification.predicted_answer,
                        "reference_answer": verification.reference_answer,
                        "generated_tokens": verification.generated_tokens,
                        "stopped_on_answer": verification.stopped_on_answer,
                        "hit_max_tokens": verification.hit_max_tokens,
                    }
                    break
                semantic_failures += 1
                retry_reasons = ("semantic_solution_mismatch",)
                observed_reasons.update(retry_reasons)
                attempt_reason_counts.update(retry_reasons)
                continue
            retry_reasons = result.reasons
            observed_reasons.update(retry_reasons)
            attempt_reason_counts.update(retry_reasons)

        if framework is not None:
            accepted_records.append(
                {
                    **record,
                    "framework": framework,
                    "semantic_verification": semantic_verification,
                }
            )
        else:
            final_reasons = sorted(observed_reasons or {"generation_failed"})
            reason_counts.update(final_reasons)
            if LEAKAGE_REASONS.intersection(final_reasons):
                leakage_records += 1
            if len(invalid_examples) < 20:
                invalid_examples.append({"index": index, "reasons": final_reasons})

        if source_examined % args.progress_every == 0:
            _print_progress(
                source_examined=source_examined,
                valid=len(accepted_records),
                requested_valid=args.limit,
                attempts=attempts_used,
            )

    if source_examined == 0 or source_examined % args.progress_every:
        _print_progress(
            source_examined=source_examined,
            valid=len(accepted_records),
            requested_valid=args.limit,
            attempts=attempts_used,
        )

    total = source_examined
    invalid = total - len(accepted_records)
    audit = {
        "schema_version": 3,
        "requested_valid": args.limit,
        "source_examined": source_examined,
        "total": total,
        "valid": len(accepted_records),
        "invalid": invalid,
        "reasons": dict(sorted(reason_counts.items())),
        "leakage_records": leakage_records,
        "leakage_rate": leakage_records / total if total else 0.0,
        "attempt_reasons": dict(sorted(attempt_reason_counts.items())),
        "attempts_used": attempts_used,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "hit_max_attempts": hit_max_attempts,
        "closed_tag_attempts": closed_tag_attempts,
        "ended_with_eos_attempts": ended_with_eos_attempts,
        "semantic_checks": semantic_checks,
        "semantic_passes": semantic_passes,
        "semantic_failures": semantic_failures,
        "semantic_prompt_tokens": semantic_prompt_tokens,
        "semantic_output_tokens": semantic_output_tokens,
        "semantic_hit_max_tokens": semantic_hit_max_tokens,
        "semantic_answer_stops": semantic_answer_stops,
        "semantic_max_new_tokens": args.semantic_max_new_tokens,
        "max_attempts": args.max_attempts,
        "seed": args.seed,
        "temperature": args.temperature,
        "invalid_examples": invalid_examples,
        "output": str(output),
    }
    complete = len(accepted_records) == args.limit
    audit = _publish_generation_result(
        output=output,
        audit_output=audit_output,
        accepted_records=accepted_records,
        audit=audit,
        complete=complete,
    )
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if not complete:
        raise RuntimeError(
            f"generated {len(accepted_records)} valid labels, fewer than requested {args.limit}; "
            "inspect the audit JSON"
        )


if __name__ == "__main__":
    main()
