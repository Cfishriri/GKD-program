"""Human-authorized, frozen-framework development-set OPD closeout.

Experimental auditor acceptance never authorizes training here. Human annotations
are the authority; all supplied negative audit evidence acts as a conservative veto.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from framework_opd.framework_validation import validate_framework
from framework_opd.prompts import format_student_prompt, format_vanilla_student_prompt

STUDENT = '/root/eb-public/huggingface-models/Qwen/Qwen3-1.7B'
TEACHER = '/root/eb-public/huggingface-models/Qwen/Qwen3-4B'
MODES = ('guided', 'vanilla')
CONDITIONS = ('no_framework', 'with_framework')


def read_rows(path):
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{i}: invalid JSONL: {error}') from error
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(row):
    value = [row['question'], row['answer'], list(row['framework'])]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def code_hashes():
    files = [Path(__file__), ROOT / 'train_opd.py', *sorted((ROOT / 'src/framework_opd').glob('*.py'))]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def token_budget(value):
    value = int(value)
    if not 1 <= value <= 2048:
        raise ValueError('generation token budget must be in [1, 2048]')
    return value


def token_counts(generation, tokenizer, prompt):
    """Actual generated IDs and retokenized, displayed answer have different scopes."""
    return {
        'prompt_tokens': len(tokenizer(prompt, add_special_tokens=True)['input_ids']),
        'generated_tokens': len(generation.token_ids),
        'answer_tokens': len(tokenizer(generation.text, add_special_tokens=False)['input_ids']),
    }


def token_distribution(rows, field):
    values = sorted(int(row[field]) for row in rows)
    if not values:
        raise ValueError('token distribution requires nonempty rows')
    return dict(total=sum(values), mean=statistics.mean(values), min=values[0],
                median=statistics.median(values), p95=values[math.ceil(0.95 * len(values)) - 1],
                max=values[-1])


def question_key(row):
    return ' '.join(row['question'].split()).casefold()


def validate_splits(train, test):
    if not train or not test:
        raise ValueError('train and test must be nonempty')
    if ({r['sample_id'] for r in train} & {r['sample_id'] for r in test}
            or {question_key(r) for r in train} & {question_key(r) for r in test}):
        raise ValueError('train/test overlap')


def bind_corrections(annotations, corrections, reviewed_data):
    """Bind legacy ID-only human notes to their explicitly reviewed source rows."""
    source = {r['sample_id']: r for r in reviewed_data}
    current = {r['sample_id']: r for r in annotations}
    if len(source) != len(reviewed_data):
        raise ValueError('duplicate reviewed source ID')
    bound = []
    for row in corrections:
        key = row['sample_id']
        if key not in source or key not in current or content_hash(source[key]) != content_hash(current[key]):
            raise ValueError(f'human correction source content mismatch: {key}')
        digest = content_hash(source[key])
        if row.get('content_sha256', digest) != digest:
            raise ValueError(f'human correction content hash mismatch: {key}')
        bound.append({**row, 'content_sha256': digest,
                      'binding_method': 'exact question/answer/framework match against explicit reviewed source'})
    return bound


def select_data(annotations, corrections, audit_rows, train_count, test_count, seed):
    if min(train_count, test_count) < 1:
        raise ValueError('positive sample counts required')
    by_id = {}
    questions = set()
    for row in annotations:
        key = row['sample_id']
        if key in by_id or question_key(row) in questions:
            raise ValueError('duplicate annotation ID/question')
        by_id[key] = row
        questions.add(question_key(row))
    changes = {}
    for row in corrections:
        key = row['sample_id']
        if key not in by_id or key in changes or row.get('human_confirmed') is not True:
            raise ValueError('unknown, duplicate or unconfirmed human correction')
        if row.get('content_sha256') != content_hash(by_id[key]):
            raise ValueError('human correction content hash mismatch')
        changes[key] = row
    evidence = {key: [] for key in by_id}
    for row in audit_rows:
        key = row['sample_id']
        if key not in by_id:
            continue  # e.g. a separately repaired control, never a replacement.
        audit = row['audit']
        if audit['content_sha256'] != content_hash(by_id[key]):
            raise ValueError(f'audit content hash mismatch: {key}')
        if audit.get('status') == 'accepted' and audit.get('accepted') is not True:
            raise ValueError(f'contradictory audit status: {key}')
        evidence[key].append(audit)
    eligible, skipped = [], []
    for key, row in by_id.items():
        correction = changes.get(key)
        label = correction['review_label'] if correction else row.get('reference_label')
        reasons = []
        if row.get('human_confirmed') is not True:
            reasons.append('human_unconfirmed')
        if label != 'correct':
            reasons.append('human_' + str(label))
        if not evidence[key]:
            reasons.append('audit_missing')
        reasons.extend('audit_' + a.get('status', 'missing') for a in evidence[key]
                       if a.get('status') != 'accepted')
        valid = validate_framework(row.get('framework'), row.get('answer'))
        reasons.extend('structure_' + r for r in valid.reasons)
        if not row.get('question', '').strip() or not row.get('answer', '').strip():
            reasons.append('empty_question_or_answer')
        if reasons:
            skipped.append(dict(sample_id=key, reasons=sorted(set(reasons))))
            continue
        eligible.append({**row, 'human_final_label': label,
                         'human_correction': correction, 'content_sha256': content_hash(row),
                         'admission_authority': 'user_confirmed_annotation',
                         'audit_statuses': [a['status'] for a in evidence[key]]})
    if len(eligible) < train_count + test_count:
        raise ValueError(f'insufficient retained rows: {len(eligible)}; need {train_count + test_count}')
    random.Random(seed).shuffle(eligible)
    train, test = eligible[:train_count], eligible[train_count:train_count + test_count]
    validate_splits(train, test)
    return train, test, skipped


def prepare(args):
    annotations = read_rows(args.annotations)
    corrections = bind_corrections(annotations, read_rows(args.corrections), read_rows(args.reviewed_data))
    audits = [row for path in args.audits for row in read_rows(path)]
    train, test, skipped = select_data(annotations, corrections, audits, args.train_count, args.test_count, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [args.annotations, args.corrections, args.reviewed_data, *args.audits]
    copies = []
    for i, path in enumerate(inputs):
        name = f'input-{i}.jsonl'
        shutil.copyfile(path, args.output / name)
        copies.append(dict(file=name, original=str(Path(path).resolve()), sha256=sha(args.output / name)))
    write_rows(args.output / 'train.jsonl', train)
    write_rows(args.output / 'test.jsonl', test)
    write_rows(args.output / 'skipped.jsonl', skipped)
    write_rows(args.output / 'bound_corrections.jsonl', corrections)
    accounted = {r['sample_id'] for r in train + test + skipped}
    unselected = [r['sample_id'] for r in annotations if r['sample_id'] not in accounted]
    statuses = Counter(r['audit'].get('status', 'missing') for r in audits)
    manifest = dict(schema=1, seed=args.seed, train_count=len(train), test_count=len(test),
                    source_count=len(annotations), skipped_count=len(skipped),
                    eligible_count=len(annotations) - len(skipped),
                    unselected_count=len(annotations) - len(skipped) - len(train) - len(test),
                    unselected_ids=unselected,
                    retained_coverage=(len(annotations) - len(skipped)) / len(annotations),
                    skip_reasons=dict(Counter(x for r in skipped for x in r['reasons'])),
                    audit_record_status_counts=dict(statuses), inputs=copies, code=code_hashes(),
                    authority='human-confirmed; experimental audit is veto-only; flags unchanged',
                    limitation='selected known development set; reference-assisted/oracle framework condition; not unseen generalization',
                    files={name: sha(args.output / name) for name in ('train.jsonl', 'test.jsonl', 'skipped.jsonl', 'bound_corrections.jsonl')})
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def load_bundle(path):
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['code'] != code_hashes():
        raise ValueError('code changed after freeze; create a new bundle')
    for name, digest in manifest['files'].items():
        if sha(path / name) != digest:
            raise ValueError('frozen dataset changed')
    for source in manifest['inputs']:
        if sha(path / source['file']) != source['sha256']:
            raise ValueError('frozen source evidence changed')
    source_rows = [read_rows(path / item['file']) for item in manifest['inputs']]
    corrections = bind_corrections(source_rows[0], source_rows[1], source_rows[2])
    if corrections != read_rows(path / 'bound_corrections.jsonl'):
        raise ValueError('bound human corrections changed')
    train, test, skipped = select_data(source_rows[0], corrections,
                                     [r for group in source_rows[3:] for r in group],
                                     manifest['train_count'], manifest['test_count'], manifest['seed'])
    if (train != read_rows(path / 'train.jsonl') or test != read_rows(path / 'test.jsonl')
            or skipped != read_rows(path / 'skipped.jsonl')):
        raise ValueError('frozen rows disagree with human evidence')
    return manifest, train, test


def make_prompt(row, guided, tokenizer):
    # Reference answer is intentionally not supplied to the model.
    raw = (format_student_prompt(row['question'], row['framework']) if guided
           else format_vanilla_student_prompt(row['question']))
    raw = raw.replace('End with the final answer in the form: #### number',
                      'End with a line starting with #### followed by your computed numeric result. '
                      'Do not output a placeholder or repeat the answer.')
    return tokenizer.apply_chat_template([{'role': 'user', 'content': raw}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def check_predictions(test, predictions):
    expected = {(r['sample_id'], mode, condition): content_hash(r)
                for r in test for mode in MODES for condition in CONDITIONS}
    actual = {}
    for row in predictions:
        key = row['sample_id'], row['mode'], row['condition']
        if key in actual or key not in expected or row['content_sha256'] != expected[key]:
            raise ValueError('duplicate, unexpected or mismatched prediction')
        if type(row['correct']) is not bool:
            raise ValueError('correct must be boolean')
        actual[key] = row
    if set(actual) != set(expected):
        raise ValueError('incomplete four-cell predictions; no partial accuracy report')


def worker(args):
    manifest, train, test = load_bundle(args.bundle)
    if args.smoke:
        train, test = train[:2], test[:2]
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from framework_opd.evaluation import score_prediction, artifact_fingerprint
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from framework_opd.eval_batching import generate_batch
    from framework_opd.rollout import _build_rollout
    from framework_opd.loss import generalized_jsd_loss
    from framework_opd.masking import causal_completion_mask
    from train_opd import _model_identity

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.cuda.init()
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    seed = manifest['seed']
    config = dict(seed=seed, train_tokens=args.train_tokens, eval_tokens=args.eval_tokens,
                  batch_size=args.batch_size, learning_rate=1e-5, beta=1.0, temperature=1.0,
                  rollout_temperature=0.7, accumulation=4, lora_r=8, lora_alpha=16,
                  prompt_protocol='native_chat_no_thinking_no_literal_number_placeholder_v1',
                  smoke=args.smoke, bundle_sha256=sha(args.bundle / 'manifest.json'),
                  train_ids=[r['sample_id'] for r in train], test_ids=[r['sample_id'] for r in test],
                  code=code_hashes(), models={p: _model_identity(p) for p in (STUDENT, TEACHER)})
    write_json(args.output / 'run.json', dict(config=config, mode=args.mode, device=args.device, complete=False))
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True, padding_side='left')
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError('teacher/student tokenizer vocabularies differ')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, local_files_only=True, dtype=torch.bfloat16).to(args.device)
    teacher.eval().requires_grad_(False)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = AutoModelForCausalLM.from_pretrained(STUDENT, local_files_only=True, dtype=torch.bfloat16).to(args.device)
    student = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
        bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    window_tokens, updates = 0, 0
    for i, row in enumerate(train, 1):
        tick = time.monotonic()
        prompt = make_prompt(row, args.mode == 'guided', tokenizer)
        student.eval()
        generation = generate_batch(student, tokenizer, [prompt], args.train_tokens, temperature=0.7)[0]
        rollout = _build_rollout(question=row['question'], framework=row['framework'] if args.mode == 'guided' else [],
                                 student_prompt=prompt, generation=generation, tokenizer=tokenizer)
        student.train()
        ids, mask = rollout.input_ids.to(args.device), rollout.attention_mask.to(args.device)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
        student_logits = student(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
        completion_mask = causal_completion_mask(rollout.labels).to(args.device)
        # Only completion positions participate; float32 divergence avoids bf16 cancellation.
        active = completion_mask.bool()
        loss, metrics = generalized_jsd_loss(student_logits[active].float().unsqueeze(0),
            teacher_logits[active].float().unsqueeze(0),
            torch.ones((1, int(active.sum())), device=args.device), beta=1.0, reduction='sum')
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite OPD loss')
        loss.backward()
        window_tokens += int(metrics['num_tokens'])
        if i % 4 == 0 or i == len(train):
            for parameter in student.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_tokens)
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_tokens = 0
            updates += 1
        result = dict(step=i, sample_id=row['sample_id'], content_sha256=content_hash(row),
                      loss=float(metrics['loss']), **token_counts(generation, tokenizer, prompt),
                      hit_max_tokens=generation.hit_max_tokens, optimizer_steps=updates,
                      seconds=time.monotonic() - tick, completion=generation.text)
        with (args.output / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'completion'}), flush=True)
        del loss, metrics, teacher_logits, student_logits, ids, mask, rollout
    student.save_pretrained(args.output / 'adapter')
    train_seconds = time.monotonic() - started
    del teacher, optimizer
    torch.cuda.empty_cache()
    student.eval()
    student.gradient_checkpointing_disable()
    predictions = []
    for condition in CONDITIONS:
        for offset in range(0, len(test), args.batch_size):
            batch = test[offset:offset + args.batch_size]
            prompts = [make_prompt(r, condition == 'with_framework', tokenizer) for r in batch]
            generations = generate_batch(student, tokenizer, prompts, args.eval_tokens)
            for row, prompt, generation in zip(batch, prompts, generations):
                result = dict(sample_id=row['sample_id'], mode=args.mode, condition=condition,
                              content_sha256=content_hash(row), completion=generation.text,
                              **token_counts(generation, tokenizer, prompt),
                              hit_max_tokens=generation.hit_max_tokens,
                              stopped_on_answer=generation.stopped_on_answer, ended_with_eos=generation.ended_with_eos,
                              **score_prediction(generation.text, row['answer']))
                predictions.append(result)
                with (args.output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps(dict(event='evaluation_progress', condition=condition,
                                  completed=min(offset + args.batch_size, len(test)), total=len(test))), flush=True)
    write_json(args.output / 'run.json', dict(config=config, mode=args.mode, device=args.device,
               complete=True, train_seconds=train_seconds, seconds=time.monotonic() - started,
               peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 1024**3,
               optimizer_steps=updates, adapter=artifact_fingerprint(args.output / 'adapter'),
               predictions_sha256=sha(args.output / 'predictions.jsonl')))


def report(args):
    from framework_opd.evaluation import score_prediction, summarize, artifact_fingerprint
    manifest, train, test = load_bundle(args.bundle)
    runs = [json.loads((path / 'run.json').read_text(encoding='utf-8')) for path in args.workers]
    if len(runs) != 2 or {r['mode'] for r in runs} != set(MODES) or not all(r['complete'] for r in runs):
        raise ValueError('both training/evaluation workers must complete')
    if runs[0]['config'] != runs[1]['config']:
        raise ValueError('unmatched training/evaluation budgets or inputs')
    cfg = runs[0]['config']
    if cfg['bundle_sha256'] != sha(args.bundle / 'manifest.json') or cfg['code'] != code_hashes():
        raise ValueError('worker bundle/code mismatch')
    if cfg['smoke']:
        train, test = train[:2], test[:2]
    if cfg['train_ids'] != [r['sample_id'] for r in train] or cfg['test_ids'] != [r['sample_id'] for r in test]:
        raise ValueError('worker sample set mismatch')
    predictions = []
    for path, run in zip(args.workers, runs):
        if sha(path / 'predictions.jsonl') != run['predictions_sha256']:
            raise ValueError('predictions changed after completion')
        if artifact_fingerprint(path / 'adapter') != run['adapter']:
            raise ValueError('adapter changed after completion')
        rows = read_rows(path / 'predictions.jsonl')
        for row in rows:
            if row['mode'] != run['mode']:
                raise ValueError('worker prediction mode mismatch')
        predictions.extend(rows)
    check_predictions(test, predictions)
    references = {r['sample_id']: r for r in test}
    for row in predictions:
        scored = score_prediction(row['completion'], references[row['sample_id']]['answer'])
        if any(row[k] != value for k, value in scored.items()):
            raise ValueError('prediction scoring mismatch')
    cells = []
    for mode in MODES:
        for condition in CONDITIONS:
            rows = [r for r in predictions if r['mode'] == mode and r['condition'] == condition]
            measured = summarize(rows)
            keys = ('correct', 'total', 'accuracy', 'accuracy_ci95_low', 'accuracy_ci95_high',
                    'relaxed_correct', 'relaxed_accuracy', 'answer_format_rate', 'answer_stop_rate',
                    'eos_rate', 'truncation_rate', 'average_generated_tokens')
            cells.append(dict(mode=mode, condition=condition, **{k: measured[k] for k in keys},
                              prompt_tokens=token_distribution(rows, 'prompt_tokens'),
                              generated_tokens=token_distribution(rows, 'generated_tokens'),
                              answer_tokens=token_distribution(rows, 'answer_tokens')))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'summary.json', dict(complete=True, smoke=cfg['smoke'], cells=cells,
        selection=manifest, config=cfg, worker_seconds={r['mode']: r['seconds'] for r in runs}))
    paired = [dict(**r, predictions=[p for p in predictions if p['sample_id'] == r['sample_id']]) for r in test]
    write_rows(args.output / 'paired_predictions.jsonl', paired)
    with (args.output / 'prediction_tokens.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        names = ('sample_id', 'mode', 'condition', 'correct', 'relaxed_correct',
                 'predicted_answer', 'reference_answer', 'prompt_tokens',
                 'generated_tokens', 'answer_tokens', 'hit_max_tokens', 'stopped_on_answer')
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows({name: row[name] for name in names} for row in predictions)
    labels = {'guided': '框架OPD', 'vanilla': '普通OPD', 'with_framework': '有框架测试', 'no_framework': '无框架测试'}
    lines = ['# 四组收尾实验', '', '仅为人工筛选的已见开发集、参考答案辅助框架对照；不是未见题泛化结论。',
             f"功能冒烟测试：{cfg['smoke']}；训练{len(train)}题，测试{len(test)}题。",
             f"候选{manifest['source_count']}题，排除{manifest['skipped_count']}题；保留比例{manifest['retained_coverage']:.1%}。",
             '', '| 训练 | 测试 | 严格正确 | 宽松正确 | 答案token均值/中位数/P95 | 实际生成token均值 | 截断数 |',
             '|---|---|---|---|---|---|---|']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="370" viewBox="0 0 900 370">',
           '<rect width="900" height="370" fill="white"/>',
           '<g font-family="sans-serif" font-size="17" fill="#172338">',
           '<text x="25" y="30">Human-confirmed frameworks: selected development subset</text>']
    for i, cell in enumerate(cells):
        label = f"{labels[cell['mode']]} + {labels[cell['condition']]}"
        lengths = cell['answer_tokens']
        lines.append(f"| {labels[cell['mode']]} | {labels[cell['condition']]} | {cell['correct']}/{cell['total']} ({cell['accuracy']:.1%}) | {cell['relaxed_correct']}/{cell['total']} ({cell['relaxed_accuracy']:.1%}) | {lengths['mean']:.1f}/{lengths['median']:.1f}/{lengths['p95']} | {cell['generated_tokens']['mean']:.1f} | {round(cell['truncation_rate'] * cell['total'])} |")
        y = 70 + i * 65
        color = '#247ba0' if cell['mode'] == 'guided' else '#b15d20'
        svg.extend([f'<text x="25" y="{y + 22}">{label}</text>',
            f'<rect x="340" y="{y}" width="400" height="30" fill="#edf1f5"/>',
            f'<rect x="340" y="{y}" width="{400 * cell["accuracy"]}" height="30" fill="{color}"/>',
            f'<text x="755" y="{y + 22}">{cell["correct"]}/{cell["total"]} ({cell["accuracy"]:.0%})</text>'])
    svg.extend(['<text x="25" y="355">Same questions in all 4 cells; skipped frameworks excluded before inference.</text>', '</g></svg>'])
    (args.output / 'accuracy.svg').write_text('\n'.join(svg), encoding='utf-8')
    lines.extend(['', '答案token是最终保存文本用student tokenizer重新分词的长度；实际生成token是模型产生的token数。',
                  '各组token总数、最小值、中位数、P95、最大值见summary.json；逐题见prediction_tokens.csv。'])
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--corrections', type=Path, required=True)
    p.add_argument('--reviewed-data', type=Path, required=True)
    p.add_argument('--audits', type=Path, nargs='+', required=True)
    p.add_argument('--train-count', type=int, default=60)
    p.add_argument('--test-count', type=int, default=20)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser('worker')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--device', choices=('cuda:0', 'cuda:1'), required=True)
    p.add_argument('--train-tokens', type=token_budget, default=1024)
    p.add_argument('--eval-tokens', type=token_budget, default=2048)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.set_defaults(func=worker)
    p = sub.add_parser('report')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--workers', type=Path, nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=report)
    args = parser.parse_args()
    if getattr(args, 'batch_size', 1) < 1:
        parser.error('batch-size must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
