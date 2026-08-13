# -*- coding: utf-8 -*-
"""
Token-level recoverability analysis with parallel teacher verification.
Student generates on GPU0, teacher evaluates each prefix on GPU1.
"""

import argparse
import math
import queue
import random
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path: str, device: str):
    """
    Load tokenizer and model from the given path.

    Returns:
        tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


def top_p_sample(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """
    Sample a token from logits using temperature and top-p (nucleus) sampling.

    Args:
        logits: [1, vocab_size]
        temperature: sampling temperature (<=0 for argmax)
        top_p: cumulative probability threshold

    Returns:
        token (tensor), token_prob (float)
    """
    if temperature <= 0:
        token = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits.float(), dim=-1)
        token_prob = probs[0, token.item()].item()
        return token, token_prob

    logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probs = F.softmax(sorted_logits.float(), dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p
        # Shift the mask to keep the first token that exceeds the threshold
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove_mask,
            float("-inf"),
        )

        probs = F.softmax(sorted_logits.float(), dim=-1)
        sampled_sorted_idx = torch.multinomial(probs, num_samples=1)
        token = sorted_indices.gather(-1, sampled_sorted_idx)
        token_prob = probs.gather(-1, sampled_sorted_idx).item()

    else:
        probs = F.softmax(logits.float(), dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = probs.gather(-1, token).item()

    return token, token_prob


# ============================================================
# Teacher Verifier
# ============================================================

class PrivilegedTeacherVerifier:
    """
    Teacher verifier that evaluates a student's reasoning prefix.
    It compares the prefix against the correct answer and outputs
    a recoverability score (probability of "A" vs "B").
    """

    def __init__(
        self,
        tokenizer,
        model,
        device,
        question,
        correct_answer,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.question = question
        self.correct_answer = correct_answer

        # A/B are usually single tokens, but we support multi-token fallback.
        self.a_ids = tokenizer("A", add_special_tokens=False).input_ids
        self.b_ids = tokenizer("B", add_special_tokens=False).input_ids

    def build_prompt(self, student_prefix: str) -> str:
        SYSTEM_PROMPT = """
    You are an evaluator observing an unfinished response produced by another language model.

    You are given:
    1. the original problem,
    2. the ground-truth final answer,
    3. the student's response generated so far.

    Your only task is to estimate whether the student's CURRENT trajectory will
    eventually end with the correct final answer.

    Judge the trajectory as it currently stands.

    Do not require the student's reasoning style or intermediate method to match
    your own.

    A different but valid reasoning path should still be considered likely to
    produce the correct answer.

    Output one of the following labels (use the single-letter form expected by the verifier):

    A = the current trajectory is likely to eventually produce the correct final answer.
    B = the current trajectory is unlikely to eventually produce the correct final answer.

    Output only A or B.
    """.strip()

        user_prompt = f"""
QUESTION:
{self.question}

PRIVILEGED CORRECT FINAL ANSWER:
{self.correct_answer}

CURRENT STUDENT REASONING PREFIX:
{student_prefix}

Does this prefix still naturally point toward the correct answer?

A = recoverable / still compatible with the correct solution
B = harmful deviation / requires correction or backtracking
""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text

    @torch.inference_mode()
    def score(self, student_prefix: str) -> float:
        """
        Return P(A) / (P(A) + P(B)) using the teacher model.

        If A and B are single tokens, we use a fast path.
        Otherwise, we fall back to multi-token likelihood.
        """
        prompt_text = self.build_prompt(student_prefix)

        # Fast path: A and B are single tokens
        if len(self.a_ids) == 1 and len(self.b_ids) == 1:
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)

            outputs = self.model(**inputs, use_cache=False)
            next_logits = outputs.logits[0, -1].float()

            pair_logits = torch.stack(
                [
                    next_logits[self.a_ids[0]],
                    next_logits[self.b_ids[0]],
                ]
            )
            pair_probs = F.softmax(pair_logits, dim=0)
            return pair_probs[0].item()

        # Fallback: multi-token
        return self._multi_token_score(prompt_text)

    @torch.inference_mode()
    def _multi_token_score(self, prompt_text: str) -> float:
        """Calculate likelihood ratio for multi-token A/B labels."""
        base_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        ).input_ids

        candidates = [self.a_ids, self.b_ids]
        sequences = [base_ids + cand for cand in candidates]

        max_len = max(len(seq) for seq in sequences)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []

        for seq in sequences:
            pad_len = max_len - len(seq)
            input_ids.append(seq + [pad_id] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            attention_masks, dtype=torch.long, device=self.device
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits.float()

        candidate_scores = []
        base_len = len(base_ids)

        for batch_idx, candidate in enumerate(candidates):
            score = 0.0
            for j, token_id in enumerate(candidate):
                prediction_position = base_len + j - 1
                log_probs = F.log_softmax(
                    logits[batch_idx, prediction_position], dim=-1
                )
                score += log_probs[token_id].item()
            score /= len(candidate)   # length normalization
            candidate_scores.append(score)

        scores = torch.tensor(candidate_scores)
        probs = F.softmax(scores, dim=0)
        return probs[0].item()


# ============================================================
# Teacher Worker Thread
# ============================================================

def teacher_worker(verifier, task_queue, result_dict, error_list):
    """Background thread that consumes tasks and evaluates prefixes."""
    try:
        torch.cuda.set_device(torch.device(verifier.device))

        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break

            step = task["step"]
            prefix = task["prefix"]
            p_correct = verifier.score(prefix)
            result_dict[step] = p_correct
            task_queue.task_done()

    except Exception as e:
        error_list.append(e)


# ============================================================
# Student Generation with Parallel Teacher
# ============================================================

@torch.inference_mode()
def generate_student_with_parallel_teacher(
    student_tokenizer,
    student_model,
    student_device,
    verifier,
    question,
    max_new_tokens,
    temperature,
    top_p,
):
    """
    Generate tokens from the student model, and asynchronously evaluate
    each prefix using the teacher model on a separate thread/GPU.
    """
    messages = [
        {
            "role": "system",
            "content": "Solve the problem carefully. "
                       "Show your reasoning and give the final answer.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    encoded = student_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(student_device)
    input_ids = encoded["input_ids"]
    attention_mask = torch.ones_like(input_ids)

    # --------------------------------------------------------
    # Teacher asynchronous queue
    # --------------------------------------------------------
    task_queue = queue.Queue()
    teacher_results = {}
    teacher_errors = []

    worker = threading.Thread(
        target=teacher_worker,
        args=(verifier, task_queue, teacher_results, teacher_errors),
        daemon=True,
    )
    worker.start()

    # --------------------------------------------------------
    # Student first forward
    # --------------------------------------------------------
    outputs = student_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    generated_ids = []
    token_records = []

    eos_token_id = student_tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_ids = set(eos_token_id)
    else:
        eos_ids = {eos_token_id}

    # --------------------------------------------------------
    # Token-by-token generation
    # --------------------------------------------------------
    for step in range(1, max_new_tokens + 1):
        next_token, student_token_prob = top_p_sample(
            next_logits,
            temperature=temperature,
            top_p=top_p,
        )

        token_id = next_token.item()
        generated_ids.append(token_id)

        token_text = student_tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        prefix_text = student_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        token_records.append({
            "step": step,
            "token_id": token_id,
            "token_text": token_text,
            "student_token_prob": student_token_prob,
            "prefix": prefix_text,
        })

        # Submit the prefix for teacher evaluation (async)
        if step % 8 == 0:
            task_queue.put({
                "step": step,
                "prefix": prefix_text,
            })

        if token_id in eos_ids:
            # If generation ends between two 8-token checkpoints,
            # evaluate the final student prefix as well.
            if step % 8 != 0:
                task_queue.put({
                    "step": step,
                    "prefix": prefix_text,
                })
            break

        # Next token using KV cache
        outputs = student_model(
            input_ids=next_token.view(1, 1),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
    # If generation stopped because max_new_tokens was reached,
    # make sure the final prefix is evaluated.
    if token_records:
        final_step = token_records[-1]["step"]

    if final_step % 8 != 0:
        task_queue.put({
            "step": final_step,
            "prefix": token_records[-1]["prefix"],
        })
    # --------------------------------------------------------
    # Wait for teacher to finish
    # --------------------------------------------------------
    task_queue.put(None)
    task_queue.join()
    worker.join()

    if teacher_errors:
        raise RuntimeError(f"Teacher worker failed: {teacher_errors[0]}")

    return token_records, teacher_results


# ============================================================
# Visualization / Saving
# ============================================================

def save_results(token_records, teacher_results, output_dir):
    """
    Save token-level student information and teacher evaluation scores,
    and generate visualizations for recoverability analysis.

    This function corresponds to the 'Results and Analysis' phase of the
    experimental pipeline. It processes raw generation data, exports
    structured tables, and produces figures that illustrate the teacher's
    confidence dynamics throughout the student's decoding process.

    Args:
        token_records (list): List of dicts containing per-token student info.
        teacher_results (dict): Mapping from step index to teacher P(correct).
        output_dir (str or Path): Directory where all outputs will be saved.

    Returns:
        pd.DataFrame: Full token-level table (including NaN for non-checkpoint steps).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Data Collection & Preprocessing
    # ------------------------------------------------------------
    full_df = _build_full_token_table(token_records, teacher_results)
    checkpoint_df = _build_checkpoint_table(token_records, teacher_results)

    # ------------------------------------------------------------
    # 2. Export Structured Data (Tables)
    # ------------------------------------------------------------
    table_paths = _save_tables(full_df, checkpoint_df, output_dir)

    # ------------------------------------------------------------
    # 3. Generate Analytical Figures
    # ------------------------------------------------------------
    if not checkpoint_df.empty:
        fig_paths = _generate_figures(checkpoint_df, output_dir)
    else:
        fig_paths = {}
        print("WARNING: No teacher checkpoints found. Figures will not be generated.")

    # ------------------------------------------------------------
    # 4. Print Summary & Key Findings
    # ------------------------------------------------------------
    _print_summary(token_records, checkpoint_df, table_paths, fig_paths)

    return full_df


# ==================== Helper Functions ====================

def _build_full_token_table(token_records, teacher_results):
    """
    Build a table that aligns every student token with the teacher's score
    at that step (if available, otherwise NaN).
    """
    rows = []
    for record in token_records:
        step = record["step"]
        rows.append({
            **record,
            "p_correct": teacher_results.get(step, float("nan")),
        })
    return pd.DataFrame(rows)


def _build_checkpoint_table(token_records, teacher_results):
    """
    Build a table containing only steps where the teacher performed an evaluation.
    This sparse table is used for confidence curves and drop detection.
    """
    checkpoint_rows = []
    previous_p = None

    for step in sorted(teacher_results.keys()):
        p_correct = teacher_results[step]
        delta_p = 0.0 if previous_p is None else p_correct - previous_p

        # Find the corresponding student token record
        record = next(r for r in token_records if r["step"] == step)

        checkpoint_rows.append({
            "step": step,
            "token_id": record["token_id"],
            "token_text": record["token_text"],
            "student_token_prob": record["student_token_prob"],
            "p_correct": p_correct,
            "delta_p_correct": delta_p,
            "prefix": record["prefix"],
        })
        previous_p = p_correct

    return pd.DataFrame(checkpoint_rows)


def _save_tables(full_df, checkpoint_df, output_dir):
    """Export both the full token table and the sparse checkpoint table to CSV."""
    full_path = output_dir / "token_recoverability.csv"
    full_df.to_csv(full_path, index=False)

    ckpt_path = output_dir / "teacher_checkpoints.csv"
    checkpoint_df.to_csv(ckpt_path, index=False)

    return {"full_table": full_path, "checkpoint_table": ckpt_path}


def _generate_figures(checkpoint_df, output_dir):
    """
    Generate two key figures:
    1. Teacher P(correct) over checkpoint steps.
    2. Delta P(correct) between consecutive checkpoints.
    """
    max_step = int(checkpoint_df["step"].max())
    tick_interval = _determine_tick_interval(max_step)

    # --- Figure 1: Recoverability Score Curve ---
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(
        checkpoint_df["step"],
        checkpoint_df["p_correct"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Teacher P(correct)",
    )
    ax1.axhline(0.5, linestyle="--", linewidth=1, color="gray", label="Threshold (0.5)")
    ax1.set_xlabel("Student token index")
    ax1.set_ylabel("Teacher P(correct)")
    ax1.set_title("Token-level privileged teacher recoverability score")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(0, max_step + tick_interval, tick_interval))
    ax1.set_xlim(0, max_step + max(8, tick_interval // 4))
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend()
    plt.tight_layout()
    prob_path = output_dir / "p_correct_curve.png"
    plt.savefig(prob_path, dpi=200)
    plt.close()

    # --- Figure 2: Recoverability Change (Delta) ---
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(
        checkpoint_df["step"],
        checkpoint_df["delta_p_correct"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Δ P(correct)",
    )
    ax2.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax2.set_xlabel("Student token index")
    ax2.set_ylabel("Δ Teacher P(correct)")
    ax2.set_title("Change in recoverability between teacher checkpoints")
    ax2.set_xticks(range(0, max_step + tick_interval, tick_interval))
    ax2.set_xlim(0, max_step + max(8, tick_interval // 4))
    ax2.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    delta_path = output_dir / "delta_p_correct_curve.png"
    plt.savefig(delta_path, dpi=200)
    plt.close()

    return {"p_correct_curve": prob_path, "delta_curve": delta_path}


def _determine_tick_interval(max_step):
    """Choose a reasonable x‑axis tick interval based on generation length."""
    if max_step <= 256:
        return 16
    elif max_step <= 512:
        return 32
    elif max_step <= 1024:
        return 64
    else:
        return 128


def _print_summary(token_records, checkpoint_df, table_paths, fig_paths):
    """
    Print key statistics and file locations to the console,
    including the largest drops in recoverability.
    """
    print("\n" + "=" * 30)
    print(" Summary of Teacher Evaluation")
    print("=" * 30)

    if not checkpoint_df.empty:
        # Top 10 largest drops
        drops = checkpoint_df.nsmallest(
            min(10, len(checkpoint_df)),
            "delta_p_correct",
        )[["step", "token_text", "p_correct", "delta_p_correct"]]

        print("\nLargest probability drops:")
        print(drops.to_string(index=False))

        print("\n" + "-" * 30)
        print(f"Student generated tokens : {len(token_records)}")
        print(f"Teacher evaluations       : {len(checkpoint_df)}")
        print(f"Final evaluated token     : {checkpoint_df['step'].iloc[-1]}")
    else:
        print("No teacher checkpoints available.")

    print("\nSaved files:")
    for name, path in table_paths.items():
        print(f"  {name}: {path}")
    for name, path in fig_paths.items():
        print(f"  {name}: {path}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Token-level recoverability with parallel teacher."
    )

    parser.add_argument(
        "--student_model",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Student model name or path",
    )

    parser.add_argument(
        "--teacher_model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Teacher model name or path",
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Math problem to solve",
    )

    parser.add_argument(
        "--correct_answer",
        type=str,
        required=True,
        help="The correct final answer (for privileged verification)",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling threshold",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./recoverability_output",
        help="Directory to save outputs",
    )

    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires at least 2 CUDA GPUs.")

    set_seed(args.seed)

    student_device = "cuda:0"
    teacher_device = "cuda:1"

    print("================================")
    print("Loading student on GPU0")
    print("================================")
    student_tokenizer, student_model = load_model(
        args.student_model,
        student_device,
    )

    print("================================")
    print("Loading teacher on GPU1")
    print("================================")
    teacher_tokenizer, teacher_model = load_model(
        args.teacher_model,
        teacher_device,
    )

    verifier = PrivilegedTeacherVerifier(
        tokenizer=teacher_tokenizer,
        model=teacher_model,
        device=teacher_device,
        question=args.question,
        correct_answer=args.correct_answer,
    )

    print("\n================================")
    print("Parallel inference starts")
    print("================================")

    token_records, teacher_results = generate_student_with_parallel_teacher(
        student_tokenizer=student_tokenizer,
        student_model=student_model,
        student_device=student_device,
        verifier=verifier,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    student_answer = token_records[-1]["prefix"]
    print("\n================================")
    print("Student answer")
    print("================================")
    print(student_answer)

    df = save_results(
        token_records,
        teacher_results,
        args.output_dir,
    )


if __name__ == "__main__":
    main()