"""Evaluation-only batching; training rollout behavior and fingerprints stay unchanged."""

import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .answer_stopping import truncate_after_first_complete_answer
from .framework_validation import require_valid_framework
from .prompts import (
    extract_framework, format_framework_prompt, format_rollout_framework_prompt,
    has_complete_framework,
)
from .rollout import FALLBACK_FRAMEWORK, FrameworkGenerationResult, GenerationResult


class BatchCompletionStop(StoppingCriteria):
    """Remember each row's first stop length before generate adds batch padding."""

    def __init__(self, tokenizer, width, count, stop_on_answer):
        self.tokenizer = tokenizer
        self.width = width
        self.lengths = [None] * count
        self.stop_on_answer = stop_on_answer
        eos = tokenizer.eos_token_id
        self.eos_ids = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)

    def __call__(self, input_ids, scores, **kwargs):
        rows = input_ids[:, self.width:].detach().cpu().tolist()
        texts = self.tokenizer.batch_decode(rows, skip_special_tokens=True) if self.stop_on_answer else [""] * len(rows)
        for i, (row, text) in enumerate(zip(rows, texts)):
            if self.lengths[i] is None and (
                (row and row[-1] in self.eos_ids)
                or (self.stop_on_answer and truncate_after_first_complete_answer(text)[1])
            ):
                self.lengths[i] = len(row)
        return torch.tensor([n is not None for n in self.lengths], dtype=torch.bool, device=input_ids.device)


def generate_batch(model, tokenizer, prompts, max_new_tokens, *, temperature=0.0, stop_on_answer=True):
    if not prompts:
        return []
    if tokenizer.padding_side != "left":
        raise ValueError("decoder-only evaluation batches require left padding")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    width = encoded["input_ids"].shape[1]
    prompt_lengths = encoded["attention_mask"].sum(dim=1).cpu().tolist()
    stopper = BatchCompletionStop(tokenizer, width, len(prompts), stop_on_answer)
    kwargs = {"temperature": temperature, "top_p": 0.9} if temperature > 0 else {}
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True, num_beams=1, num_return_sequences=1,
            return_dict_in_generate=False,
            stopping_criteria=StoppingCriteriaList([stopper]), **kwargs,
        )
    results = []
    for i, padded in enumerate(output[:, width:].detach().cpu().tolist()):
        length = stopper.lengths[i]
        if length is None:
            # Also support EOS-only/model-provided stopping without a criteria callback.
            length = next((j + 1 for j, token in enumerate(padded) if token in stopper.eos_ids), len(padded))
        ids = tuple(int(token) for token in padded[:length])
        text = tokenizer.decode(ids, skip_special_tokens=True)
        # Match rollout._generate_text: framework generation does not stop early,
        # but its final text is still cropped so answer leakage cannot hide before
        # a closing framework tag that the step extractor would otherwise accept.
        text, answered = truncate_after_first_complete_answer(text)
        eos = bool(ids and ids[-1] in stopper.eos_ids)
        results.append(GenerationResult(ids, text, eos, len(ids) >= max_new_tokens and not eos and not answered,
            prompt_tokens=int(prompt_lengths[i]), stopped_on_answer=answered))
    return results


def generate_framework_batch(model, tokenizer, records, *, max_new_tokens, temperature, max_attempts, oracle=False):
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    states = [dict(errors=[], retry=None, prompt=0, output=0, hit=0) for _ in records]
    results = [None] * len(records)
    pending = list(range(len(records)))
    for attempt in range(1, max_attempts + 1):
        prompts = [
            format_framework_prompt(records[i]["question"], records[i]["answer"], retry_reasons=states[i]["retry"])
            if oracle else format_rollout_framework_prompt(records[i]["question"], retry_reasons=states[i]["retry"])
            for i in pending
        ]
        generations = generate_batch(model, tokenizer, prompts, max_new_tokens,
            temperature=temperature, stop_on_answer=False)
        retry = []
        for i, generation in zip(pending, generations):
            state = states[i]
            state["prompt"] += generation.prompt_tokens
            state["output"] += len(generation.token_ids)
            state["hit"] += int(generation.hit_max_tokens)
            text = "<framework>\n" + generation.text
            closed = has_complete_framework(text)
            reasons = []
            steps = None
            if not closed:
                if generation.hit_max_tokens:
                    reasons.append("hit_max_tokens")
                reasons.append("framework_not_closed")
            else:
                try:
                    steps = require_valid_framework(extract_framework(text, require_closed=True),
                        records[i]["answer"] if oracle else None)
                except ValueError as error:
                    reasons = list(getattr(error, "reasons", ("parse_error",)))
            state["errors"].extend(reasons)
            state["retry"] = tuple(reasons)
            if steps is not None or attempt == max_attempts:
                results[i] = FrameworkGenerationResult(
                    tuple(steps) if steps is not None else FALLBACK_FRAMEWORK,
                    attempt, steps is None, tuple(state["errors"]), state["prompt"], state["output"],
                    state["hit"], generation.ended_with_eos, closed,
                )
            else:
                retry.append(i)
        pending = retry
        if not pending:
            break
    return results


def iter_parallel_batches(records, replicas, batch_size, generate):
    """Generate one batch per replica concurrently; main thread alone persists rows.

    Latency is batch wall time / row count, a throughput cost, not request latency.
    """
    if batch_size <= 0 or not replicas:
        raise ValueError("positive batch_size and at least one replica are required")

    def run(replica, batch):
        started = time.perf_counter()
        results = generate(*replica, batch)
        if len(results) != len(batch):
            raise RuntimeError("generation result count does not match batch")
        return results, (time.perf_counter() - started) / len(batch)

    with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
        stride = batch_size * len(replicas)
        for offset in range(0, len(records), stride):
            jobs = []
            for worker, replica in enumerate(replicas):
                start = offset + worker * batch_size
                batch = records[start:start + batch_size]
                if batch:
                    jobs.append((batch, executor.submit(run, replica, batch)))
            for batch, future in jobs:
                results, latency = future.result()
                yield from ((record, result, latency) for record, result in zip(batch, results))
