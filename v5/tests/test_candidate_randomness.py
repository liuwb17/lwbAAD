import json
from pathlib import Path
import tempfile
import unittest

from aad.io import digest, read_json, save_json
from aad.problem import Case
from aad.parallel_engine import V51Engine, V51Provider
from aad.candidate_evaluation import from_rows, parse_candidate, seeded_source


CODE = 'int main(){return 0;}\n// AAD_V5_COMPLETE\n'


class Provider:
    def __init__(self, root, flags):
        self.root, self.flags, self.max_calls, self.ledger = Path(root), flags, len(flags), []

    @property
    def requests(self): return len(self.ledger)

    def request(self, kind, task):
        call = self.requests
        code = CODE.replace('return 0', f'return {call}')
        content = json.dumps({'code': code, 'hypothesis': 'test design', 'is_randomized': self.flags[call]})
        save_json(self.root / 'llm' / f'{call:05d}.response.json',
                  {'choices': [{'message': {'content': content}}]})
        self.ledger.append({'status': 'ok', 'usage': {'total_tokens': 1}})
        return {'call_id': call, 'content': content}


class RandomnessTests(unittest.TestCase):
    def test_runtime_ratio_averages_cases_and_seeds_without_recounting_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            engine, _ = self.engine(root, [False, True])
            evaluate = engine.evaluate

            def timed(code, subset, seconds, replica, *, seed):
                result = evaluate(code, subset, seconds, replica, seed=seed)
                for row in result['cases']:
                    row['seconds'] = 0.1 * (int(row['name']) + 1) + 0.2 * seed
                return result

            engine.evaluate = timed
            engine.run()
            records = list(engine.s['records'].values())
            self.assertAlmostEqual(records[0]['runtime_ratio'], 0.45 / 2)
            self.assertAlmostEqual(records[1]['runtime_ratio'], 0.65 / 2)
            saved = read_json(Path(root) / 'state.json')['records']
            for ident, record in engine.s['records'].items():
                self.assertEqual(saved[ident]['runtime_ratio'], record['runtime_ratio'])
                self.assertEqual(record['runtime_ratio_basis'], 'total_process_time')
                self.assertEqual(engine.feedback(ident)['timing']['runtime_ratio'], record['runtime_ratio'])

    def test_failed_precheck_also_records_observed_runtime_ratio(self):
        with tempfile.TemporaryDirectory() as root:
            engine, _ = self.engine(root, [False], lambda call, seed: (0, 'WA'))
            engine.run()
            record = next(iter(engine.s['records'].values()))
            self.assertAlmostEqual(record['runtime_ratio'], 0.75)
            self.assertNotIn('full', record)

    def engine(self, root, flags, behavior=None):
        cases = [Case(str(i), f'input {i}') for i in range(8)]
        seen = []
        def evaluate(code, subset, seconds, replica, *, seed):
            call = int(code.split('return ')[1].split(';')[0])
            seen.extend((call, c.name, seed) for c in subset)
            score, status = behavior(call, seed) if behavior else (10 + seed, 'AC')
            return {**from_rows([{'name': c.name, 'score': score, 'raw_score': score,
                                'status': status, 'stdout': f'output-seed-{seed}',
                                'message': '', 'seed': seed, 'seconds': 1.5} for c in subset]), 'seed': seed}
        provider = Provider(root, flags)
        engine = V51Engine(root, provider, evaluate, cases,
                          {'direction': 'maximize', 'time_limit': 2, 'smoke_count': 2})
        return engine, seen

    def test_each_case_once_or_three_times_and_no_two_win_gate(self):
        def behavior(call, seed):
            return ((10 if call == 0 else [15, 9, 9][seed] if call == 1 else 0), 'AC')
        with tempfile.TemporaryDirectory() as root:
            engine, seen = self.engine(root, [False, True, True], behavior)
            engine.run()
            self.assertEqual(len(seen), 8 + 24 + 24)
            for call, expected_seeds in ((0, {0}), (1, {0, 1, 2}), (2, {0, 1, 2})):
                for case in engine.train:
                    seeds = [s for c, name, s in seen if c == call and name == case.name]
                    self.assertEqual(len(seeds), len(expected_seeds))
                    self.assertEqual(set(seeds), expected_seeds)
            champion = engine.s['incumbent']
            self.assertIn('return 1', engine.source(champion))
            self.assertEqual(engine.mean(champion), 11)
            feedback = engine.feedback(champion)
            self.assertEqual(feedback['train_mean'], 11)
            self.assertEqual(feedback['examples'][0]['score'], 9)
            self.assertEqual(feedback['examples'][0]['mean_score'], 11)
            self.assertNotIn('confirmation', engine.s['events'][1])
            engine.finalize()
            self.assertEqual(read_json(Path(root) / 'report.json')['evaluation_count'], 3)

    def test_missing_or_nonboolean_flag_and_macro_override_are_rejected(self):
        for flag in (None, 'true', 'false', 0, 1):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'boolean'):
                parse_candidate('program', None, json.dumps({'code': CODE, 'hypothesis': 'x',
                                                            'is_randomized': flag}))
        for directive in ('#define AAD_RANDOM_SEED 7', '#undef AAD_RANDOM_SEED'):
            with self.assertRaisesRegex(ValueError, 'override'):
                parse_candidate('program', None, json.dumps({'code': directive + '\n' + CODE,
                                              'hypothesis': 'x', 'is_randomized': True}))

    def test_local_edit_can_change_classification_in_either_direction(self):
        for flag in (False, True):
            payload = {'hypothesis': 'change algorithm', 'parent_sha256': digest(CODE),
                       'is_randomized': flag, 'edits': [{'old': 'return 0', 'new': 'return 1'}]}
            code, _, metadata = parse_candidate('edit', CODE, json.dumps(payload))
            self.assertIn('return 1', code)
            self.assertIs(metadata['is_randomized'], flag)
        del payload['is_randomized']
        with self.assertRaisesRegex(ValueError, 'boolean'):
            parse_candidate('edit', CODE, json.dumps(payload))

    def test_randomized_resume_reuses_first_seed_without_extra_model_request(self):
        fail = [True]
        def behavior(call, seed):
            if seed == 1 and fail[0]:
                fail[0] = False
                raise OSError('infrastructure interruption')
            return 10 + seed, 'AC'
        with tempfile.TemporaryDirectory() as root:
            engine, seen = self.engine(root, [True], behavior)
            with self.assertRaises(OSError):
                engine.run()
            resumed = V51Engine(root, engine.provider, engine.evaluate, engine.train, engine.config)
            resumed.run()
            self.assertEqual(engine.provider.requests, 1)
            self.assertEqual(len([r for r in seen if r[2] == 0]), 8)
            self.assertEqual(len([r for r in seen if r[2] == 2]), 8)
            self.assertEqual(resumed.mean(resumed.s['incumbent']), 11)

    def test_seed_specific_invalidity_disqualifies_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            engine, seen = self.engine(root, [True], lambda call, seed: (10, 'WA' if seed == 1 else 'AC'))
            engine.run()
            self.assertIsNone(engine.s['incumbent'])
            self.assertEqual(engine.s['events'][0]['status'], 'invalid_candidate')
            self.assertEqual(set(s for _, _, s in seen), {0, 1})  # Fail fast once a seed is invalid.
            self.assertEqual(engine.task()['arm'], 'repair')

    def test_same_code_conflicting_flag_cannot_change_evaluation_count(self):
        with tempfile.TemporaryDirectory() as root:
            engine, seen = self.engine(root, [False, True])
            original = engine.provider.request
            def request(kind, task):
                reply = original(kind, task)
                payload = json.loads(reply['content'])
                payload['code'] = CODE
                reply['content'] = json.dumps(payload)
                return reply
            engine.provider.request = request
            engine.run()
            self.assertEqual(len(seen), 8)
            self.assertEqual(engine.s['events'][1]['status'], 'failed')
            self.assertIn('Conflicting metadata', engine.s['events'][1]['error'])

    def test_provider_requests_boolean_for_all_solver_operations(self):
        with tempfile.TemporaryDirectory() as root:
            provider = V51Provider(root, 'mock', 1, 0, key='test', problem='test')
            for kind in ('program', 'edit'):
                body = provider.request_body(kind, {'problem': 'test'})
                self.assertEqual(body['response_format'], {'type': 'json_object'})
                self.assertIn('is_randomized', body['messages'][0]['content'])
                self.assertIn('AAD_RANDOM_SEED', body['messages'][0]['content'])

    def test_seed_prefix_preserves_candidate_source(self):
        for seed in (0, 1, 2):
            executable = seeded_source(CODE, seed)
            self.assertTrue(executable.startswith(f'#define AAD_RANDOM_SEED {seed}ULL\n'))
            self.assertTrue(executable.endswith(CODE))
        with self.assertRaises(ValueError):
            seeded_source(CODE, 3)


if __name__ == '__main__':
    unittest.main()
