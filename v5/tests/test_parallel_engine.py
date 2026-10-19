import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from aad.io import digest, read_json, save_json
from aad.problem import Case
from aad.candidate_policy import pool
from aad.ale_batch import AleBatchEvaluator, load_task, prepare, private_test, statement_for, summarize
from aad.parallel_engine import V51Engine, V51Provider
from aad.candidate_evaluation import aggregate_evaluations


CODE = 'int main(){return 0;}\n// AAD_V5_COMPLETE\n'


def result(values, cases, direction='maximize', status='ACCEPTED'):
    items = [SimpleNamespace(judge_result=SimpleNamespace(value=status), absolute_score=value,
                             execution_time=1.5, output_str='answer', error_str='', message='')
             for value in values]
    return summarize(items, cases, direction)


class FakeProvider:
    def __init__(self, root, calls=6):
        self.root, self.max_calls, self.ledger, self.tasks = Path(root), calls, [], []

    @property
    def requests(self):
        return len(self.ledger)

    def request(self, kind, task):
        self.tasks.append(task)
        call = self.requests
        code = CODE.replace('return 0', f'return {call}')
        if kind == 'edit':
            old = task['parent_source']
            content = json.dumps({'hypothesis': 'improve', 'is_randomized': False, 'parent_sha256': task['parent_sha256'],
                                  'edits': [{'old': old, 'new': code}]})
        else:
            content = json.dumps({'code': code, 'hypothesis': 'improve', 'is_randomized': False})
        save_json(self.root / 'llm' / f'{call:05d}.response.json',
                  {'choices': [{'message': {'content': content}}]})
        self.ledger.append({'status': 'ok', 'usage': {'total_tokens': 10}})
        return {'call_id': call, 'content': content}


class V51Tests(unittest.TestCase):
    def cases(self, count=5):
        return [Case(f'p{i}', f'{i}\n') for i in range(count)]

    def config(self, direction='maximize'):
        return {'direction': direction, 'time_limit': 2.0}

    def add(self, engine, ident, values, direction='maximize', parent=None):
        code = CODE.replace('return 0', f'return {ident}')
        key = digest(code)
        folder = engine.root / 'candidates' / key
        folder.mkdir(parents=True)
        (folder / 'solution.cpp').write_text(code, encoding='utf-8')
        r = result(values, engine.train, direction)
        engine.s['records'][key] = {'lineage': key, 'parent': parent,
                                    'smoke': r, 'full': r, 'evaluations': [r], 'is_randomized': False}
        return key

    def test_full_workflow_maximize_minimize_and_negative_scores(self):
        for direction in ('maximize', 'minimize'):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as root:
                cases, seen = self.cases(), []
                sign = 1 if direction == 'maximize' else -1
                def evaluate(code, subset, seconds, replica, *, seed=0):
                    self.assertEqual(seconds, 2)
                    seen.extend(c.name for c in subset)
                    call = int(code.split('return ')[1].split(';')[0])
                    return result([-100 + sign * call] * len(subset), subset, direction)
                provider = FakeProvider(root)
                engine = V51Engine(root, provider, evaluate, cases, self.config(direction))
                engine.run()
                self.assertEqual(provider.requests, 6)
                self.assertEqual(set(seen), {c.name for c in cases})
                self.assertIn('return 5', engine.source(engine.s['incumbent']))
                count = len(seen)
                champion = engine.finalize()
                self.assertEqual(len(seen), count)  # No holdout evaluation at freeze.
                self.assertEqual(engine.finalize(), champion)
                report = read_json(Path(root) / 'report.json')
                self.assertEqual(report['train_mean'], -100 + sign * 5)
                self.assertNotIn('validation_mean', report)
                resumed = V51Engine(root, provider, evaluate, cases, self.config(direction))
                with self.assertRaisesRegex(ValueError, 'closed'):
                    resumed.run()
                for task in provider.tasks:
                    self.assertNotIn('.992', task['instruction'])
                    if 'feedback' in task:
                        self.assertEqual(task['feedback']['direction'], direction)

    def test_minimization_archive_donor_and_worst_examples(self):
        with tempfile.TemporaryDirectory() as root:
            cases = self.cases(2)
            engine = V51Engine(root, None, None, cases,
                              {**self.config('minimize'), 'initial_valid_roots': 2})
            parent = self.add(engine, 1, [10, 30], 'minimize')
            donor = self.add(engine, 2, [5, 50], 'minimize')
            self.assertEqual(pool(engine.s['records'])[0], parent)
            engine.s['schedule_slot'] = 4
            task = engine.task()
            self.assertEqual((task['arm'], task['parent'], task['donor']), ('crossover', parent, donor))
            feedback = engine.feedback(parent)
            self.assertEqual(feedback['examples'][0]['score'], 30)
            self.assertEqual(feedback['train_mean'], 20)

    def test_mean_only_gate_and_margin(self):
        for gains, margin, promoted in (([1, 1, -1], 0, True), ([5, -1, -1], 0, True),
                                         ([0, 0, 0], 0, False), ([1, 1, 1], 1, False)):
            with self.subTest(gains=gains, margin=margin), tempfile.TemporaryDirectory() as root:
                cases = self.cases()
                engine = V51Engine(root, None, None, cases,
                                  {**self.config(), 'promotion_margin': margin})
                old = self.add(engine, 1, [10] * len(cases))
                new = self.add(engine, 2, [11] * len(cases))
                engine.s['incumbent'] = old
                r = engine.s['records'][new]
                r['is_randomized'] = True
                r['evaluations'] = [result([10 + g] * len(cases), cases) for g in gains]
                r['full'] = aggregate_evaluations(r['evaluations'])
                pending = {'parent': old, 'arm': 'local'}
                engine.qualify(new, pending)
                self.assertEqual(engine.s['incumbent'] == new, promoted)

    def test_seed_failure_repairs_and_excludes_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            cases = self.cases()
            engine = V51Engine(root, None, None, cases, self.config())
            key = self.add(engine, 1, [10] * len(cases))
            r = engine.s['records'][key]
            r['is_randomized'] = True
            r['evaluations'].append(result([0] * len(cases), cases, status='WRONG_ANSWER'))
            r['full'] = aggregate_evaluations(r['evaluations'])
            pending = {'parent': None, 'arm': 'independent'}
            engine.qualify(key, pending)
            self.assertEqual(pending['status'], 'invalid_candidate')
            self.assertEqual(pool(engine.s['records']), [])
            self.assertTrue(engine.feedback(key)['failed_evaluation'])
            self.assertEqual(engine.task()['arm'], 'repair')

    def test_protocol_and_training_validation(self):
        with tempfile.TemporaryDirectory() as root:
            V51Engine(root, None, None, self.cases(), self.config())
            with self.assertRaisesRegex(ValueError, 'Protocol changed'):
                V51Engine(root, None, None, self.cases(), self.config('minimize'))
            with self.assertRaises(ValueError):
                V51Engine(root, None, None, [], self.config())
        for field, value in (('time_limit', 0), ('promotion_margin', -1), ('repeats', 1)):
            with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
                V51Engine(root, None, None, self.cases(), {**self.config(), field: value})

    def test_pending_candidate_resume_does_not_spend_another_call(self):
        with tempfile.TemporaryDirectory() as root:
            cases, attempts = self.cases(), []
            provider = FakeProvider(root, calls=1)
            def evaluate(code, subset, seconds, replica, *, seed=0):
                attempts.append(replica)
                if len(attempts) == 1:
                    raise OSError('temporary infrastructure outage')
                return result([10] * len(subset), subset)
            engine = V51Engine(root, provider, evaluate, cases, self.config())
            with self.assertRaises(OSError):
                engine.run()
            self.assertIsNotNone(engine.s['pending']['candidate'])
            resumed = V51Engine(root, provider, evaluate, cases, self.config())
            resumed.run()
            self.assertEqual(provider.requests, 1)
            self.assertEqual(len(resumed.s['records']), 1)
            self.assertIsNone(resumed.s['pending'])
            self.assertEqual(attempts, ['smoke', 'smoke'])

    def test_provider_injects_only_actual_task(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, *args):
                return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': CODE}}],
                                   'usage': {'total_tokens': 1}}).encode()
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                        return_value=Response()) as http:
            provider = V51Provider(root, 'mock', 1, 0, key='test', problem='Minimize routing cost.')
            provider.request('program', {'problem': 'unwanted override'})
            body = json.loads(http.call_args.args[0].data)
            self.assertEqual(json.loads(body['messages'][1]['content'])['problem'], 'Minimize routing cost.')
            self.assertNotIn('rectangles', json.dumps(body))

    def test_nonfinite_and_infrastructure_results_stop(self):
        with self.assertRaises(ValueError):
            result([float('nan')], self.cases(1))
        with self.assertRaises(OSError):
            result([0], self.cases(1), status='INTERNAL_ERROR')


class AleBundleTests(unittest.TestCase):
    def archive(self, root, kind='batch'):
        path = Path(root) / 'toy.zip'
        data = {'metadata': {'problem_id': 'toy', 'problem_type': kind, 'score_type': 'minimize'},
                'constraints': {'time_limit': 2, 'memory_limit': 1024},
                'seeds': {'public': list(range(50)), 'private': [101, 102]}}
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('toy/data.json', json.dumps(data))
            z.writestr('toy/statement_en.md', 'Find a short route.')
            z.writestr('toy/tools/Cargo.toml', '# fixture')
            z.writestr('toy/standings_scores.csv', 'MUST NOT ENTER TRAINING')
        return path

    def prepare(self, root):
        archive = self.archive(root)
        bundle = Path(root) / 'task'
        with patch('aad.ale_batch.build_tools'), patch('aad.ale_batch.generate',
                side_effect=lambda seeds, _: [f'input {s}' for s in seeds]) as gen:
            prepare(archive, 'toy', bundle)
            self.assertEqual(gen.call_args.args[0], list(range(50)))
        return archive, bundle

    def test_public_default_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as root:
            _, bundle = self.prepare(root)
            task, cases = load_task(bundle)
            self.assertEqual(len(cases), 50)
            self.assertNotIn('private', json.dumps(task))
            self.assertFalse((bundle / 'standings_scores.csv').exists())
            self.assertNotIn('argv[1]', statement_for(task))
            (bundle / 'public/0.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                load_task(bundle)

    def test_reactive_and_nonpublic_seeds_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.ale_batch.generate') as gen:
            archive = self.archive(root, kind='reactive')
            with self.assertRaisesRegex(ValueError, 'non-interactive'):
                prepare(archive, 'toy', Path(root) / 'task')
            archive = self.archive(root)
            with self.assertRaisesRegex(ValueError, 'public seeds'):
                prepare(archive, 'toy', Path(root) / 'task', [101])
            gen.assert_not_called()

    def test_public_subset_and_tool_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            archive = self.archive(root)
            bundle = Path(root) / 'task'
            with patch('aad.ale_batch.build_tools'), patch('aad.ale_batch.generate',
                    return_value=['one', 'two']) as gen:
                prepare(archive, 'toy', bundle, [2, 4])
                self.assertEqual(gen.call_args.args[0], [2, 4])
            self.assertEqual(len(load_task(bundle)[1]), 2)
            (bundle / 'tools/Cargo.toml').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'tool source changed'):
                load_task(bundle)

    def test_sdk_call_contract_and_separate_repeat_caches(self):
        with tempfile.TemporaryDirectory() as root:
            _, bundle = self.prepare(root)
            _, cases = load_task(bundle)
            items = [SimpleNamespace(judge_result=SimpleNamespace(value='ACCEPTED'),
                                     absolute_score=12, execution_time=.1, output_str='out',
                                     error_str='', message='') for _ in cases]
            # Match the pinned official run_cases keyword-only usage and enum contract.
            calls = []
            def run_cases(*, inputs, code, code_language, judge_version, time_limit,
                          memory_limit, problem_id, problem_type, tool_dir, return_details,
                          skip_local_visualization):
                calls.append(locals())
                self.assertEqual((code_language, judge_version, problem_type), ('cpp17', '202301', 'batch'))
                self.assertEqual((time_limit, memory_limit, problem_id), (2, 1024, 'toy'))
                return items
            modules = {
                'ale_bench': SimpleNamespace(),
                'ale_bench.code_language': SimpleNamespace(
                    CodeLanguage=SimpleNamespace(CPP17='cpp17'), JudgeVersion=SimpleNamespace(V202301='202301')),
                'ale_bench.data': SimpleNamespace(ProblemType=SimpleNamespace(BATCH='batch')),
                'ale_bench.tool_wrappers': SimpleNamespace(run_cases=run_cases)}
            with patch.dict(sys.modules, modules), patch('aad.ale_batch.runtime_identity',
                    return_value={'cpu_policy': {}}), patch('aad.cpu_pool.make_runner', return_value=run_cases):
                evaluator = AleBatchEvaluator(bundle, Path(root) / 'cache')
                first = evaluator(CODE, cases, 2, 'train-r0')
                self.assertEqual(evaluator(CODE, cases, 2, 'train-r0'), first)
                evaluator(CODE, cases, 2, 'train-r1')
                evaluator(CODE, cases, 2, 'train-r1', seed=1)
                self.assertEqual(len(calls), 3)
                self.assertTrue(calls[0]['code'].startswith('#define AAD_RANDOM_SEED 0ULL'))
                self.assertTrue(calls[2]['code'].startswith('#define AAD_RANDOM_SEED 1ULL'))
                self.assertEqual((first['raw_mean'], first['mean']), (12, -12))
                with self.assertRaisesRegex(ValueError, 'official task time limit'):
                    evaluator(CODE, cases, .6, 'smoke')

    def test_private_freeze_gate_and_terminal_test(self):
        with tempfile.TemporaryDirectory() as root:
            archive, bundle = self.prepare(root)
            task, cases = load_task(bundle)
            run = Path(root) / 'run'
            config = {'direction': 'minimize', 'time_limit': 2,
                      'task_sha256': digest(json.dumps(task, sort_keys=True)), 'runtime': {}, 'workers': 1}
            engine = V51Engine(run, None, None, cases, config)
            with patch('aad.ale_batch.generate') as gen, self.assertRaisesRegex(ValueError, 'Freeze'):
                private_test(run, bundle, archive)
            gen.assert_not_called()
            key = V51Tests().add(engine, 1, [20] * len(cases), 'minimize')
            engine.s['incumbent'] = key
            engine.finalize()
            before = copy.deepcopy(engine.s)
            class Evaluator:
                runtime = {}
                def __call__(self, code, subset, seconds, replica, *, seed=0):
                    return result([30] * len(subset), subset, 'minimize')
            with patch('aad.ale_batch.AleBatchEvaluator', return_value=Evaluator()), patch(
                    'aad.ale_batch.generate', return_value=['private a', 'private b']) as gen:
                report = private_test(run, bundle, archive)
                self.assertEqual(gen.call_args.args[0], [101, 102])
                self.assertEqual(report['private_mean'], 30)
                self.assertEqual(read_json(run / 'state.json'), before)
                with self.assertRaisesRegex(ValueError, 'already started'):
                    private_test(run, bundle, archive)
            self.assertTrue(read_json(run / 'report.json')['private_evaluated'])

    def test_private_tampered_source_rejected_without_generator(self):
        with tempfile.TemporaryDirectory() as root:
            archive, bundle = self.prepare(root)
            task, cases = load_task(bundle)
            run = Path(root) / 'run'
            engine = V51Engine(run, None, None, cases, {'direction': 'minimize', 'time_limit': 2,
                'task_sha256': digest(json.dumps(task, sort_keys=True)), 'runtime': {}, 'workers': 1})
            key = V51Tests().add(engine, 1, [20] * len(cases), 'minimize')
            engine.s['incumbent'] = key
            engine.finalize()
            (run / 'best.cpp').write_text('changed')
            with patch('aad.ale_batch.generate') as gen, self.assertRaisesRegex(ValueError, 'source changed'):
                private_test(run, bundle, archive)
            gen.assert_not_called()

    def test_randomized_private_uses_three_seeds_and_mean(self):
        with tempfile.TemporaryDirectory() as root:
            archive, bundle = self.prepare(root)
            task, cases = load_task(bundle)
            run = Path(root) / 'run'
            engine = V51Engine(run, None, None, cases, {'direction': 'minimize', 'time_limit': 2,
                'task_sha256': digest(json.dumps(task, sort_keys=True)), 'runtime': {}, 'workers': 1})
            key = V51Tests().add(engine, 1, [20] * len(cases), 'minimize')
            record = engine.s['records'][key]
            record['is_randomized'] = True
            record['evaluations'] = [copy.deepcopy(record['full']) for _ in range(3)]
            record['full'] = aggregate_evaluations(record['evaluations'])
            engine.s['incumbent'] = key
            engine.finalize()
            seen = []
            class Evaluator:
                runtime = {}
                def __call__(self, code, subset, seconds, replica, *, seed=0):
                    seen.append((replica, seed))
                    return result([30 + seed] * len(subset), subset, 'minimize')
            with patch('aad.ale_batch.AleBatchEvaluator', return_value=Evaluator()), patch(
                    'aad.ale_batch.generate', return_value=['private a', 'private b']):
                report = private_test(run, bundle, archive)
            self.assertEqual(seen, [('private-r0', 0), ('private-r1', 1), ('private-r2', 2)])
            self.assertEqual(report['private_mean'], 31)
            self.assertEqual(report['evaluation_count'], 3)
            self.assertEqual(report['algorithm_seeds'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
