"""Official-seed defaults and shared public/private evaluator, offline only."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

from aad.io import digest, read_json
from aad.parallel_engine import V51Engine
from aad.ale_batch import load_task
from aad.ale_evaluation import prepare, private_test, AleBatchEvaluator
from aad.idle_cpu_pool import CPU_PROTOCOL
import scripts.rsi_search as cli
import test_parallel_engine as fixtures
import test_cpu_pool as cpu_fixtures


class EvaluationTests(unittest.TestCase):
    def prepare_bundle(self, root):
        archive = fixtures.AleBundleTests().archive(root)
        bundle = Path(root)/'task'
        with patch('aad.ale_batch.build_tools'), patch('aad.ale_batch.generate',
                side_effect=lambda seeds, _: [f'input {s}' for s in seeds]) as generate:
            prepare(archive, 'toy', bundle)
            self.assertEqual(list(generate.call_args.args[0]), list(range(20)))
        return archive, bundle

    def test_default_is_official_public_zero_through_nineteen(self):
        with tempfile.TemporaryDirectory() as root:
            _, bundle = self.prepare_bundle(root)
            task, cases = load_task(bundle)
            self.assertEqual(len(cases), 20)
            self.assertEqual([c['seed'] for c in task['cases']], list(range(20)))
            self.assertNotIn('private', json.dumps(task))

    def test_private_uses_full_official_seed_list_after_freeze_and_same_evaluator(self):
        with tempfile.TemporaryDirectory() as root:
            archive, bundle = self.prepare_bundle(root)
            task, cases = load_task(bundle)
            run = Path(root)/'run'
            runtime = {'cpu_policy': {'protocol': CPU_PROTOCOL}}
            engine = V51Engine(run, None, None, cases, {
                'direction': 'minimize', 'time_limit': 2, 'runtime': runtime,
                'task_sha256': digest(json.dumps(task, sort_keys=True))})
            with patch('aad.ale_batch.generate') as generator:
                with self.assertRaisesRegex(ValueError, 'Freeze'):
                    private_test(run, bundle, archive)
                generator.assert_not_called()
            key = fixtures.V51Tests().add(engine, 1, [20]*20, 'minimize')
            engine.s['incumbent'] = key
            engine.finalize()
            before = read_json(run/'state.json')

            class Evaluator:
                def __init__(self, *args): self.runtime = runtime
                def __call__(self, code, subset, seconds, replica, *, seed=0):
                    return fixtures.result([30]*len(subset), subset, 'minimize')

            with patch('aad.ale_evaluation.AleBatchEvaluator', side_effect=Evaluator) as evaluator, \
                 patch('aad.ale_batch.generate', return_value=['private a', 'private b']) as generator:
                report = private_test(run, bundle, archive)
                self.assertEqual(generator.call_args.args[0], [101, 102])
                self.assertEqual(report['private_count'], 2)
                evaluator.assert_called_once_with(bundle, run/'private_measurements')
                self.assertEqual(read_json(run/'state.json'), before)

    def test_realtime_cpu_policy_enters_evaluation_cache_identity(self):
        with tempfile.TemporaryDirectory() as root:
            _, bundle = self.prepare_bundle(root)
            original = {'cpu_policy': cpu_fixtures.topology_policy(cpu_fixtures.topology())}
            with patch('aad.ale_batch.runtime_identity', return_value=original), \
                 patch('aad.ale_evaluation.make_runner') as runner:
                evaluator = AleBatchEvaluator(bundle, Path(root)/'cache')
                self.assertEqual(evaluator.runtime['cpu_policy']['protocol'], CPU_PROTOCOL)
                self.assertEqual(runner.call_args.args[0], evaluator.runtime['cpu_policy'])
                legacy_runtime = {**original, 'cpu_policy': cpu_fixtures.topology_policy(cpu_fixtures.topology())}
                old_signature = digest(json.dumps({'task': evaluator.task, 'runtime': legacy_runtime}, sort_keys=True))
                self.assertNotEqual(evaluator.signature, old_signature)

    def test_cli_prepare_defaults_and_explicit_subset(self):
        for expected, extra in ((list(range(20)), []), ([1, 3], ['--seeds', '1', '3'])):
            with patch.object(cli, 'prepare', return_value={'problem_id': 'toy', 'cases': []}) as prepare_fn, \
                 redirect_stdout(io.StringIO()):
                cli.main(['prepare', '--problem', 'toy', '--output', 'unused', '--archive', 'fixture.zip', *extra])
                self.assertEqual(prepare_fn.call_args.args[3], expected)

    def test_cli_meta_settings_are_separate_from_solver_and_saved_in_protocol_config(self):
        for method, provider_name in (('5.2.r.0.online', 'OnlineProvider'),):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                _, bundle = self.prepare_bundle(root)
                engine = MagicMock()
                with patch.dict(cli.ENGINES, {method: engine}), \
                     patch.object(cli, 'AleBatchEvaluator', return_value=SimpleNamespace(runtime={})), \
                     patch.object(cli, provider_name) as provider, redirect_stdout(io.StringIO()):
                    cli.main(['run', '--task', str(bundle), '--run-dir', str(Path(root)/'run'),
                              '--method', method, '--calls', '1', '--model', 'solver-model',
                              '--meta-model', 'meta-model', '--no-private-after-run'])
                config = engine.call_args.args[4]
                self.assertEqual(config['model'], 'solver-model')
                self.assertEqual(config['meta_model'], 'meta-model')
                self.assertEqual(config['meta_reasoning_effort'], 'max')
                self.assertEqual(provider.call_args.args[1], 'solver-model')
                self.assertEqual(provider.call_args.kwargs['meta_model'], 'meta-model')
                self.assertEqual(provider.call_args.kwargs['meta_reasoning_effort'], 'max')

    def test_cli_rejects_old_fifty_case_bundle_before_evaluator_or_model(self):
        with tempfile.TemporaryDirectory() as root:
            archive, bundle = fixtures.AleBundleTests().prepare(root)
            with patch.object(cli, 'AleBatchEvaluator') as evaluator, patch.object(cli, 'V51Provider') as provider:
                with self.assertRaisesRegex(ValueError, 'public seeds differ'):
                    cli.main(['run', '--task', str(bundle), '--run-dir', str(Path(root)/'run'), '--calls', '1'])
                evaluator.assert_not_called()
                provider.assert_not_called()

    def test_cli_defaults_to_freeze_then_private_but_stop_marker_only_pauses(self):
        for stopped in (False, True):
            with self.subTest(stopped=stopped), tempfile.TemporaryDirectory() as root:
                archive, bundle = self.prepare_bundle(root)
                run = Path(root)/'run'
                order = []
                class Engine:
                    def __init__(self, *args, **kwargs): pass
                    def initialize(self): order.append('initials')
                    def run(self):
                        order.append('evolution')
                        if stopped:
                            (run/'STOP_AFTER_CURRENT').touch()
                    def finalize(self): order.append('freeze')
                def private(*args):
                    order.append('private')
                    return {'private_count': 2}
                with patch.dict(cli.ENGINES, {'5.2.r.0': Engine}), \
                     patch.object(cli, 'AleBatchEvaluator', return_value=SimpleNamespace(runtime={})), \
                     patch.object(cli, 'V51Provider'), patch.object(cli, 'private_test', side_effect=private), \
                     redirect_stdout(io.StringIO()):
                    cli.main(['run', '--task', str(bundle), '--run-dir', str(run), '--calls', '1', '--archive', str(archive)])
                self.assertEqual(order, ['initials', 'evolution'] if stopped else ['initials', 'evolution', 'freeze', 'private'])


if __name__ == '__main__':
    unittest.main()
