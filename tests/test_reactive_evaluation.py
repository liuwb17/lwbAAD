"""Reactive protocol regressions; no Docker, model calls or private benchmark data."""
import ast
from contextlib import contextmanager
import math
from pathlib import Path
import re
import shlex
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from aad import ale_batch as legacy
from aad import ale_evaluation as ale
from aad.cpu_pool import isolated_runner
from aad.container_execution import GuardedContainer, scoped_batch_command
from aad.fixed_initials import WORKSPACE
import test_parallel_engine as fixtures
import test_cpu_pool as cpu_fixtures
import test_ale_evaluation as shared_tests


class ReactiveEvaluationTests(shared_tests.EvaluationTests):
    # Run the existing public/private/freeze/CPU-cache/CLI regressions again,
    # using a reactive task bundle instead of batch.
    def prepare_bundle(self, root):
        archive = fixtures.AleBundleTests().archive(root, kind='reactive')
        bundle = Path(root) / 'task'
        with patch('aad.ale_batch.build_tools'), patch('aad.ale_batch.generate',
                side_effect=lambda seeds, _: [f'input {s}' for s in seeds]):
            ale.prepare(archive, 'toy', bundle)
        return archive, bundle

    def test_default_is_official_public_zero_through_nineteen(self):
        with tempfile.TemporaryDirectory() as root:
            _, bundle = self.prepare_bundle(root)
            task, cases = ale.load_task(bundle)
            self.assertEqual(task['problem_type'], 'reactive')
            self.assertEqual([c['seed'] for c in task['cases']], list(range(20)))
            self.assertEqual(len(cases), 20)
            self.assertNotIn('private', str(task))
            self.assertFalse((bundle / 'standings_scores.csv').exists())
            with self.assertRaises(ValueError):
                legacy.load_task(bundle)
            statement = ale.statement_for(task)
            self.assertIn('flush', statement)
            self.assertIn('private to the tester, not your stdin', statement)

    def test_private_uses_full_official_seed_list_after_freeze_and_same_evaluator(self):
        with patch.object(shared_tests, 'load_task', ale.load_task):
            super().test_private_uses_full_official_seed_list_after_freeze_and_same_evaluator()

    def test_dispatch_seed_cache_and_official_failures(self):
        with tempfile.TemporaryDirectory() as root:
            _, bundle = self.prepare_bundle(root)
            _, cases = ale.load_task(bundle)
            runtime = {'cpu_policy': cpu_fixtures.topology_policy(cpu_fixtures.topology())}
            problem_types = SimpleNamespace(BATCH='batch-enum', REACTIVE='reactive-enum')
            modules = {
                'ale_bench.code_language': SimpleNamespace(
                    CodeLanguage=SimpleNamespace(CPP17='cpp17'), JudgeVersion=SimpleNamespace(V202301='202301')),
                'ale_bench.data': SimpleNamespace(ProblemType=problem_types)}
            with patch('aad.ale_batch.runtime_identity', return_value=runtime), \
                 patch('aad.ale_evaluation.make_runner') as factory, patch.dict('sys.modules', modules):
                evaluator = ale.AleEvaluator(bundle, Path(root) / 'cache')
                self.assertEqual(evaluator.runtime['execution_protocol'], ale.REACTIVE_PROTOCOL)
                runner = factory.return_value
                def item(status):
                    official_status = next(key for key, value in legacy.STATUS.items() if value == status)
                    return SimpleNamespace(judge_result=SimpleNamespace(value=official_status), absolute_score=17,
                                           execution_time=.3, output_str='transcript', error_str='', message='')
                runner.return_value = [item('AC')]
                result = evaluator('int main(){}', cases[:1], 2, 0, seed=0)
                self.assertEqual((result['raw_mean'], result['mean']), (17, -17))
                kwargs = runner.call_args.kwargs
                self.assertEqual(kwargs['problem_type'], 'reactive-enum')
                self.assertEqual(kwargs['inputs'], [cases[0].text])
                self.assertIn('AAD_RANDOM_SEED', kwargs['code'])
                evaluator('int main(){}', cases[:1], 2, 0, seed=0)
                self.assertEqual(runner.call_count, 1)
                evaluator('int main(){}', cases[:1], 2, 1, seed=0)
                self.assertEqual(runner.call_count, 2)
                for index, status in enumerate(('WA', 'RE', 'TLE', 'MLE', 'CE')):
                    runner.return_value = [item(status)]
                    result = evaluator('int main(){}', cases[:1], 2, index + 10)
                    self.assertFalse(result['valid'])
                    self.assertEqual(result['cases'][0]['status'], status)
                runner.return_value = [item('IE')]
                with self.assertRaises(OSError):
                    evaluator('int main(){}', cases[:1], 2, 99)


class ReactiveRunnerTests(unittest.TestCase):
    def test_official_reactive_scoring_and_failure_classification(self):
        source = WORKSPACE / 'ALE-Bench/src/ale_bench/tool_wrappers/case_runner.py'
        if not source.is_file():
            self.skipTest('External ALE-Bench source checkout is not installed at ALE-Bench/')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'run_reactive_judge_container')
        outcomes = [('Score = 42', .3, None), ('Invalid move', .3, 'WA'),
                    ('', .3, 'RE'), ('', 3.2, 'TLE')]
        for stderr, elapsed, expected in outcomes:
            with self.subTest(expected=expected):
                removed, started = [], []
                container = SimpleNamespace(
                    wait=lambda: None, logs=lambda **kw: stderr.encode(),
                    attrs={'State': {'ExitCode': 0}}, remove=lambda **kw: removed.append(kw))
                @contextmanager
                def client():
                    def run(**kwargs):
                        started.append(kwargs)
                        return container
                    yield SimpleNamespace(containers=SimpleNamespace(run=run))
                clock = iter([0, elapsed])
                scope = {
                    'docker_client': client, 'CaseResult': SimpleNamespace, 're': re,
                    'JudgeResult': SimpleNamespace(TIME_LIMIT_EXCEEDED='TLE', RUNTIME_ERROR='RE', WRONG_ANSWER='WA'),
                    'get_docker_image_name': lambda *args: 'official-image',
                    'time': SimpleNamespace(perf_counter=lambda: next(clock)),
                    'os': SimpleNamespace(getuid=lambda: 1000, getgid=lambda: 1000),
                    'ale_bench': SimpleNamespace(constants=SimpleNamespace(
                        MAX_MEMORY_LIMIT='2g', WORK_DIR='/work', REJECTED_ABSOLUTE_SCORE=0))}
                exec(compile('from __future__ import annotations\n' + ast.unparse(function),
                             str(source), 'exec'), scope)
                result = scope['run_reactive_judge_container'](None, None, 2, {}, 'tester solver', 'hidden', None)
                if expected is None:
                    self.assertEqual(result, (.3, 42, 'Score = 42'))
                else:
                    self.assertEqual(result.judge_result, expected)
                self.assertEqual(removed, [{'force': True}])
                self.assertTrue(started[0]['network_disabled'])
                self.assertEqual(started[0]['cpu_quota'], 100000)

    def test_real_official_builder_keeps_tester_solver_order_and_limits(self):
        source = WORKSPACE / 'ALE-Bench/src/ale_bench/tool_wrappers/case_runner.py'
        if not source.is_file():
            self.skipTest('External ALE-Bench source checkout is not installed at ALE-Bench/')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'build_reactive_judge_command')
        constants = SimpleNamespace(INPUT_FILE='/input', OUTPUT_FILE='/output', PROFILES_FILE='/profiles',
                                    TESTER_BIN='/tester', TIME_OUTPUT_FORMAT='%e')
        scope = {'ale_bench': SimpleNamespace(constants=constants), 'math': math, 'shlex': shlex,
                 'get_run_command': lambda *args: '/solver'}
        code = 'from __future__ import annotations\n' + ast.unparse(function)
        exec(compile(code, str(source), 'exec'), scope)
        builder = scope['build_reactive_judge_command']
        original = builder(None, None, 2)
        wrapped = scoped_batch_command(builder)(None, None, 2)
        self.assertEqual(wrapped, original[:-len('; sync')] + '; sync -- /output /profiles')
        self.assertIn('/tester /usr/bin/time', wrapped)
        self.assertIn('/solver < /input > /output', wrapped)
        self.assertIn('timeout 3.2 prlimit --cpu=3.1', wrapped)

    def test_reactive_lease_precedes_clock_guards_wait_and_releases_on_failure(self):
        module = ModuleType('fake_reactive_runner')
        module.clock, module.seen = [0], []
        exec('''
def docker_client():
    raise AssertionError('Original client used')
def build_batch_run_command(output_file='/output', profiles_file='/profiles'):
    return 'solver; sync'
def build_reactive_judge_command(output_file='/output', profiles_file='/profiles'):
    return 'tester solver; sync'
def run_batch_run_container():
    raise AssertionError('Batch selected for reactive')
def run_reactive_judge_container(time_limit):
    start = clock[0]
    with docker_client() as client:
        container = client.containers.run('tester+solver', network_disabled=True)
        seen.append(container)
        container.wait()
    clock[0] += 1
    return clock[0] - start
def run_cases(**kwargs):
    return run_reactive_judge_container(kwargs['time_limit'])
''', vars(module))
        released, calls = [], []
        failure = [False]
        class Container:
            id = 'mock-reactive-container'
            def wait(self, **kwargs):
                calls.append(kwargs)
                if failure[0]:
                    raise OSError('daemon unavailable')
                return {'StatusCode': 0}
            def remove(self, **kwargs): pass
        class Pool:
            capacity = 14
            policy = {'service_cpus': [0, 1]}
            @contextmanager
            def lease(self):
                module.clock[0] += 100
                try:
                    yield {'cpu': 4, 'token': 'reactive'}
                finally:
                    released.append(True)
            @contextmanager
            def client_factory(self):
                def run(*args, **kwargs):
                    calls.append(kwargs)
                    return Container()
                yield SimpleNamespace(containers=SimpleNamespace(run=run))
        with tempfile.TemporaryDirectory() as root:
            pool = Pool()
            pool.audit_root = Path(root)
            original = module.run_reactive_judge_container
            runner = isolated_runner(module, pool, supported_types=('batch', 'reactive'))
            kwargs = dict(inputs=['x'], problem_type=SimpleNamespace(value='reactive'), time_limit=2)
            self.assertEqual(runner(**kwargs), 1)
            self.assertEqual(module.clock[0], 101)
            self.assertEqual(calls[0]['cpuset_cpus'], '4')
            self.assertTrue(calls[0]['network_disabled'])
            self.assertIsInstance(module.seen[0], GuardedContainer)
            self.assertGreater(calls[1]['timeout'], 0)
            self.assertIs(module.run_reactive_judge_container, original)
            failure[0] = True
            with self.assertRaises(OSError):
                runner(**kwargs)
            self.assertEqual(len(released), 2)
            self.assertTrue((Path(root) / 'reactive.failure.json').exists())
            with self.assertRaises(ValueError):
                runner(**kwargs, reuse_containers=True)


if __name__ == '__main__':
    unittest.main()
