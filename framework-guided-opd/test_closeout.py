import copy
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from closeout_experiment import (content_hash, select_data, validate_splits, check_predictions,
                                 token_budget, bind_corrections, prepare, load_bundle, write_rows, make_prompt,
                                 token_counts, token_distribution)


def fixture(n=6):
    rows = [dict(sample_id=f's{i}', question=f'Question {i}?', answer='#### 7',
                 framework=['Identify the given quantities.', 'Add the given quantities.'],
                 human_confirmed=True, reference_label='correct') for i in range(n)]
    audits = [dict(sample_id=r['sample_id'], audit=dict(content_sha256=content_hash(r),
              status='accepted', accepted=True, training_eligible=False)) for r in rows]
    return rows, audits


class CloseoutTests(unittest.TestCase):
    def test_generated_and_retained_answer_tokens_are_distinct(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': [i for i, _ in enumerate(text.split())]}
        generation = SimpleNamespace(token_ids=(1, 2, 3, 4, 5), text='final 7')
        self.assertEqual(token_counts(generation, Tokenizer(), 'question prompt'),
                         {'prompt_tokens': 2, 'generated_tokens': 5, 'answer_tokens': 2})

    def test_token_distribution(self):
        self.assertEqual(token_distribution([{'answer_tokens': n} for n in (1, 2, 3, 4)],
                                            'answer_tokens'),
                         {'total': 10, 'mean': 2.5, 'min': 1, 'median': 2.5, 'p95': 4, 'max': 4})

    def test_fifty_test_items_from_eighty_six_retained(self):
        rows, audits = fixture(86)
        train, test, skipped = select_data(rows, [], audits, 36, 50, 20260923)
        self.assertEqual((len(train), len(test), len(skipped)), (36, 50, 0))
        validate_splits(train, test)

    def test_chat_prompt_no_literal_placeholder_or_reference(self):
        rows, _ = fixture()
        rows[0]['answer'] = 'SECRET_REFERENCE #### 7'
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return '<chat>' + messages[0]['content']
        for guided in (True, False):
            tokenizer = Tokenizer()
            prompt = make_prompt(rows[0], guided, tokenizer)
            self.assertNotIn('#### number', prompt)
            self.assertNotIn('SECRET_REFERENCE', prompt)
            self.assertEqual(tokenizer.kwargs, dict(tokenize=False, add_generation_prompt=True, enable_thinking=False))
            self.assertEqual('<framework>' in prompt, guided)

    def test_disjoint_and_deterministic(self):
        rows, audits = fixture()
        result = select_data(rows, [], audits, 2, 2, 42)
        self.assertEqual(result, select_data(rows, [], audits, 2, 2, 42))
        validate_splits(*result[:2])

    def test_machine_acceptance_is_not_human_approval(self):
        rows, audits = fixture()
        rows[0]['human_confirmed'] = False
        train, test, skipped = select_data(rows, [], audits, 2, 2, 42)
        self.assertNotIn('s0', [r['sample_id'] for r in train + test])
        self.assertIn('human_unconfirmed', skipped[0]['reasons'])

    def test_correction_and_audit_veto(self):
        rows, audits = fixture(8)
        amendments = [dict(sample_id='s0', human_confirmed=True, review_label='incorrect',
                           content_sha256=content_hash(rows[0]))]
        audits[1]['audit'].update(status='unknown', accepted=False)
        train, test, skipped = select_data(rows, amendments, audits, 2, 2, 42)
        self.assertTrue({'s0', 's1'}.isdisjoint(r['sample_id'] for r in train + test))
        self.assertEqual(len(skipped), 2)

    def test_conflicting_history_is_not_overridden(self):
        rows, audits = fixture()
        extra = copy.deepcopy(audits[0])
        extra['audit'].update(status='rejected', accepted=False)
        train, test, skipped = select_data(rows, [], audits + [extra], 2, 2, 42)
        self.assertIn('audit_rejected', skipped[0]['reasons'])

    def test_hash_mismatch(self):
        rows, audits = fixture()
        rows[0]['framework'][0] = 'Something else.'
        with self.assertRaisesRegex(ValueError, 'hash'):
            select_data(rows, [], audits, 2, 2, 42)

    def test_duplicate_question(self):
        rows, audits = fixture()
        rows[1]['question'] = rows[0]['question'].upper() + ' '
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            select_data(rows, [], audits, 2, 2, 42)

    def test_insufficient_does_not_silently_shrink(self):
        rows, audits = fixture()
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            select_data(rows, [], audits, 5, 5, 42)

    def test_overlap(self):
        rows, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_splits(rows[:2], rows[1:3])

    def test_token_cap(self):
        for value in [0, -1, 2049]:
            with self.assertRaises(ValueError):
                token_budget(value)
        self.assertEqual(token_budget(2048), 2048)

    def test_correction_wrong_hash(self):
        rows, audits = fixture()
        amendments = [dict(sample_id='s0', human_confirmed=True, review_label='correct', content_sha256='wrong')]
        with self.assertRaisesRegex(ValueError, 'hash'):
            select_data(rows, amendments, audits, 2, 2, 42)
        with self.assertRaisesRegex(ValueError, 'hash'):
            bind_corrections(rows, amendments, rows)

    def test_correction_source_mismatch(self):
        rows, _ = fixture()
        source = copy.deepcopy(rows)
        source[0]['framework'][0] = 'Changed.'
        with self.assertRaisesRegex(ValueError, 'source content'):
            bind_corrections(rows, [dict(sample_id='s0')], source)

    def test_freeze_roundtrip_and_tamper(self):
        rows, audits = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in [('annotations', rows), ('corrections', []), ('reviewed', rows), ('audit', audits)]:
                write_rows(root / name, data)
            args = SimpleNamespace(annotations=root / 'annotations', corrections=root / 'corrections',
                    reviewed_data=root / 'reviewed', audits=[root / 'audit'], train_count=2, test_count=2,
                    seed=42, output=root / 'bundle')
            prepare(args)
            manifest, train, test = load_bundle(args.output)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 2)
            self.assertEqual(len(manifest['unselected_ids']), 2)
            with (args.output / 'train.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_bundle(args.output)

    def test_predictions_require_all_four_cells(self):
        rows, _ = fixture(2)
        predictions = [dict(sample_id=r['sample_id'], mode=m, condition=c,
                            content_sha256=content_hash(r), correct=False)
                       for r in rows for m in ('guided', 'vanilla') for c in ('no_framework', 'with_framework')]
        check_predictions(rows, predictions)
        with self.assertRaises(ValueError):
            check_predictions(rows, predictions[:-1])
        with self.assertRaises(ValueError):
            check_predictions(rows, predictions + predictions[:1])
        predictions[0]['content_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            check_predictions(rows, predictions)


if __name__ == '__main__':
    unittest.main()
