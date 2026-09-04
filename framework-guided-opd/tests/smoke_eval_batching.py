"""Real-GPU functional/throughput check, never a scientific adapter comparison.

Run from project root with PYTHONPATH=src:. python tests/smoke_eval_batching.py.
All outputs use a new temporary directory; no formal outputs are modified.
"""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch

import evaluate_comparison as evaluation
from framework_opd.eval_batching import generate_batch, generate_framework_batch, iter_parallel_batches
from framework_opd.evaluation import artifact_fingerprint, score_prediction, file_sha256
from framework_opd.prompts import format_vanilla_student_prompt
from framework_opd.rollout import generate_framework_result


def main():
    output = Path(tempfile.mkdtemp(prefix="framework-eval-batching-"))
    print(f"OUTPUT={output}", flush=True)
    config = evaluation.resolve_project_paths(json.loads(Path("configs/evaluation.json").read_text()))
    records = evaluation.select_records(config["dataset"], 32, config["seed"])
    report = {"purpose": "functional and throughput smoke only", "max_new_tokens": 2048, "stages": {}}

    def measure(name, action):
        for device in (0, 1):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = action()
        for device in (0, 1):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        stats = {"seconds": elapsed, "samples": len(result), "samples_per_second": len(result) / elapsed,
                 "peak_allocated_gib": [torch.cuda.max_memory_allocated(d) / 2**30 for d in (0, 1)]}
        report["stages"][name] = stats
        evaluation.write_json(output / "benchmark.json", report)
        print(json.dumps({"stage": name, **stats}), flush=True)
        return result

    replicas = evaluation.load_eval_replicas(config, config["student_model"], config["adapters"]["vanilla"], "cuda:0")
    def student(model, tokenizer, batch):
        return generate_batch(model, tokenizer, [format_vanilla_student_prompt(r["question"]) for r in batch], 2048)
    list(iter_parallel_batches(records[:2], replicas, 1,
        lambda m, t, b: generate_batch(m, t, ["Hello" for _ in b], 8)))
    serial = measure("student_serial_8", lambda: [evaluation.generate_completion(
        *replicas[0], format_vanilla_student_prompt(r["question"]), 2048) for r in records[:8]])
    parallel = measure("student_dual_batch4_8", lambda: [result for _, result, _ in
        iter_parallel_batches(records[:8], replicas, 4, student)])
    stress = measure("student_dual_batch16_32", lambda: [result for _, result, _ in
        iter_parallel_batches(records, replicas, 16, student)])
    report["student_comparison"] = [dict(example_id=r["example_id"],
        serial=score_prediction(a.text, r["answer"]), batch=score_prediction(b.text, r["answer"]),
        identical_tokens=a.token_ids == b.token_ids, serial_tokens=len(a.token_ids), batch_tokens=len(b.token_ids))
        for r, a, b in zip(records, serial, parallel)]
    report["stress_answer_stops"] = sum(r.stopped_on_answer for r in stress)
    report["stress_hit_max"] = sum(r.hit_max_tokens for r in stress)
    replicas.clear()
    evaluation.clear_cuda_cache()

    replicas = evaluation.load_eval_replicas(config, config["teacher_model"], config["framework_teacher_adapter"], "cuda:1")
    serial_frameworks = measure("framework_serial_4", lambda: [generate_framework_result(
        *replicas[0], r["question"], max_new_tokens=128, temperature=0.0, max_attempts=3) for r in records[:4]])
    batched_frameworks = measure("framework_dual_batch16_32", lambda: [result for _, result, _ in
        iter_parallel_batches(records, replicas, 16,
            lambda m, t, b: generate_framework_batch(m, t, b, max_new_tokens=128, temperature=0.0, max_attempts=3))])
    report["framework_comparison"] = [dict(example_id=r["example_id"],
        same_steps=a.steps == b.steps, serial_attempts=a.attempts, batch_attempts=b.attempts,
        serial_output_tokens=a.generated_tokens, batch_output_tokens=b.generated_tokens)
        for r, a, b in zip(records, serial_frameworks, batched_frameworks)]
    replicas.clear()
    evaluation.clear_cuda_cache()

    # The real Guided v3 artifact is absent. Both fixture labels intentionally
    # use Vanilla v3 to exercise ten-cell plumbing, NOT model quality comparison.
    smoke = dict(config, limit=4, bootstrap_samples=20, progress_every=1,
        student_batch_size=2, framework_batch_size=2,
        adapters={"vanilla": config["adapters"]["vanilla"], "guided": config["adapters"]["vanilla"]})
    smoke_dir = output / "integration-fixture-not-scientific"
    smoke_dir.mkdir()
    smoke["output_dir"] = str(smoke_dir)
    manifest = {"schema_version": evaluation.SCHEMA_VERSION, "status": "running", "completed_cells": [],
        "fixture_warning": "Both adapter labels use Vanilla v3; not experimental results",
        "provenance": {"training_artifacts": {"framework_teacher": {"adapter":
            artifact_fingerprint(config["framework_teacher_adapter"])}}, "models": {"teacher": {"sha256": "smoke-base-4b"}}}}
    evaluation.run_evaluation(smoke, smoke_dir, manifest, resume=False)
    evaluation.verify_completed_output_integrity(smoke_dir, manifest)
    before = file_sha256(smoke_dir / "predictions.jsonl")
    with patch.object(evaluation, "load_eval_replicas", side_effect=AssertionError("completed resume must not load models")):
        evaluation.run_evaluation(smoke, smoke_dir, manifest, resume=True)
    assert file_sha256(smoke_dir / "predictions.jsonl") == before
    report["integration"] = {"cells": len(manifest["completed_cells"]), "samples_per_cell": 4,
        "completed_resume_reused_predictions": True, "fixture_warning": manifest["fixture_warning"]}
    evaluation.write_json(output / "benchmark.json", report)
    print(f"PASS: {output / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
