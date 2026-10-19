"""Offline differential tests against unmodified v5.2, using the common importer."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aad.io import digest, read_json
from aad.problem import Case
from aad.candidate_policy import pool
from aad.ale_batch import summarize
from aad.parallel_engine import V51Provider
from aad.parallel_search import round_tasks, start_batch
from aad.fixed_initials import FixedInitialEngine, read_bank
from aad.search_policy.engine import R0Engine
from aad.search_policy.components import parallelism, parents, operators, operator_selection, memory, population


def fixed_bank():
    initials = []
    for index in range(4):
        code = f'int main(){{return {index + 10};}}\n// AAD_V5_COMPLETE\n'
        if index == 0:
            code = code.replace('\n', '\r\n')  # Import must not normalize canonical bytes.
        initials.append({'code': code, 'source_bytes': code.encode(), 'source_sha256': digest(code),
                         'initial_direction': index, 'direction': f'direction {index}',
                         'hypothesis': f'initial {index}', 'is_randomized': index % 2 == 1,
                         'bank_relative_path': f'tasks/mock/initial_{index}',
                         'bank_manifest_sha256': 'fixture'})
    return initials, {'directions': [f'direction {i}' for i in range(4)]}


def deterministic_choice(options):
    """Couple decisions by their choices, not reordered global RNG draw positions."""
    return options[-1]


class Response:
    def __init__(self, content):
        self.payload = {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
                        'usage': {'total_tokens': 10}}

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, *args): return json.dumps(self.payload).encode()


class R0Tests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('aad.fixed_initials.read_bank', side_effect=lambda *a: fixed_bank()))
        self.stack.enter_context(patch('random.choice', side_effect=deterministic_choice))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.http_count = 0
        self.stack.enter_context(patch('aad.parallel_search.urllib.request.urlopen', side_effect=self.respond))

    def respond(self, request, **kwargs):
        self.http_count += 1
        body = json.loads(request.data)
        prompt = json.loads(body['messages'][1]['content'])
        old_value = int(re.search(r'return (\d+)', prompt['parent_source'])[1])
        # Some improvements, some regressions, each fully determined by the prompt.
        value = old_value + (1 if prompt['operation'] in ('local', 'repair') else -1)
        code = (f'int main(){{return {value};}}\n// proposal:{prompt["operation"]} '
                f'{digest(json.dumps(prompt, sort_keys=True))}\n// AAD_V5_COMPLETE\n')
        result = {'hypothesis': 'measured change', 'is_randomized': prompt['operation'] != 'local'}
        if prompt['operation'] == 'local':
            result.update(parent_sha256=prompt['parent_sha256'],
                          edits=[{'old': prompt['parent_source'], 'new': code}])
        else:
            result['code'] = code
        return Response(json.dumps(result))

    def make(self, cls, root, calls=9, tokens=0, direction='maximize', fail_arm=None, memory_enabled=True):
        provider = V51Provider(root, 'mock', calls, tokens, key='test', problem='Public task')
        cases = [Case(str(i), f'input {i}') for i in range(3)]

        def evaluate(code, subset, seconds, replica, *, seed=0):
            score = int(re.search(r'return (\d+)', code)[1]) + seed
            status = 'COMPILATION_ERROR' if fail_arm and f'proposal:{fail_arm}' in code else 'ACCEPTED'
            items = [SimpleNamespace(judge_result=SimpleNamespace(value=status), absolute_score=score,
                                     execution_time=1.5, output_str='ok', message='', error_str='') for _ in subset]
            return summarize(items, subset, direction)

        config = {'problem_id': 'mock', 'direction': direction, 'time_limit': 2,
                  'calls': calls, 'token_budget': tokens, 'failure_experience': memory_enabled,
                  'smoke_count': 1}
        return cls(root, provider, evaluate, cases, config)

    def compare_runs(self, *, direction='maximize', fail_arm=None, memory_enabled=True):
        snapshots = []
        for cls in (FixedInitialEngine, R0Engine):
            with tempfile.TemporaryDirectory() as root:
                engine = self.make(cls, root, direction=direction, fail_arm=fail_arm,
                                   memory_enabled=memory_enabled)
                engine.run()
                winner = engine.finalize()
                self.assertEqual(engine.finalize(), winner)
                report = read_json(Path(root) / 'report.json')
                self.assertEqual(report['harness_version'], cls.harness_version)
                self.assertEqual(report['initialization']['evaluated_case_runs'], 12)
                events = [{k: v for k, v in e.items() if k not in ('repair', 'kind')}
                          for e in engine.s['events']]
                bodies = [read_json(engine.provider.folder / f'{i:05d}.request.json')
                          for i in range(engine.provider.requests)]
                snapshots.append((engine.s['records'], events, winner, bodies,
                                  engine.provider.requests, engine.provider.charged_tokens))
                with self.assertRaisesRegex(ValueError, 'closed'):
                    engine.run()
        self.assertEqual(snapshots[0], snapshots[1])

    def test_complete_rounds_prompts_results_memory_and_cost_match_v52(self):
        for direction in ('maximize', 'minimize'):
            for enabled in (False, True):
                with self.subTest(direction=direction, memory=enabled):
                    self.compare_runs(direction=direction, memory_enabled=enabled)

    def test_one_shot_repairs_and_budget_overrun_match_v52(self):
        self.compare_runs(fail_arm='redesign')

    def test_round_planning_matches_runtime_complementarity_and_parity(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            old, new = self.make(FixedInitialEngine, a), self.make(R0Engine, b)
            old.initialize()
            new.initialize()
            new.update_population()
            for ratio in (0, .30, .300001, 1):
                for complementary in (False, True):
                    for number in range(2):
                        for engine in (old, new):
                            ranked = pool(engine.s['records'])
                            engine.s['records'][ranked[0]]['runtime_ratio'] = ratio
                            engine.s['records'][ranked[2]]['full']['cases'][0]['score'] = 100 if complementary else 1
                            engine.s['parallel_round'] = number
                            engine.s.update(requested_calls=9, effective_calls=9)
                        _, expected = round_tasks(old)
                        actual = new.round_tasks()
                        self.assertEqual(expected, [{k: v for k, v in t.items() if k not in ('kind', 'repair')}
                                                    for t in actual])
                        start_batch(old, 'evolution', expected)
                        new.start_batch('evolution', actual)
                        self.assertEqual([t['prompt'] for t in expected], [t['prompt'] for t in actual])

    def test_original_bytes_order_and_resume_without_evaluation_or_plan(self):
        for cls in (FixedInitialEngine, R0Engine):
            with self.subTest(cls=cls), tempfile.TemporaryDirectory() as root:
                engine = self.make(cls, root)
                engine.initialize()
                self.assertEqual(self.http_count, 0)
                for index, item in enumerate(fixed_bank()[0]):
                    ident = item['source_sha256']
                    self.assertEqual((Path(root) / 'candidates' / ident / 'solution.cpp').read_bytes(), item['source_bytes'])
                    self.assertEqual(engine.s['records'][ident]['initial_slot'], index)
                resumed = self.make(cls, root)
                resumed.evaluate = lambda *a, **k: self.fail('Completed originals must not be evaluated again')
                resumed.initialize()
                self.assertEqual(len(resumed.s['events']), 4)
                self.assertFalse((Path(root) / 'planning').exists())

    def test_invalid_originals_preserve_four_slots_and_never_generate(self):
        for cls in (FixedInitialEngine, R0Engine):
            with tempfile.TemporaryDirectory() as root:
                engine = self.make(cls, root)
                original = engine.evaluate

                def invalid(code, cases, *args, **kwargs):
                    result = original(code, cases, *args, **kwargs)
                    result['valid'] = False
                    for row in result['cases']:
                        row['status'] = 'CE'
                    return result

                engine.evaluate = invalid
                with self.assertRaisesRegex(ValueError, 'Invalid fixed originals'):
                    engine.run()
                self.assertEqual(len(engine.s['records']), 4)
                self.assertEqual(len(engine.s['initialization']['completed']), 4)
                self.assertEqual(engine.provider.requests, 0)
                with self.assertRaisesRegex(ValueError, 'initialization incomplete'):
                    engine.finalize()
        self.assertEqual(self.http_count, 0)

    def test_explicit_acceptance_preserves_invalid_slots_and_evolves_valid_parents(self):
        for cls in (FixedInitialEngine, R0Engine):
            with tempfile.TemporaryDirectory() as root:
                engine = self.make(cls, root, calls=1)
                engine.config['allow_invalid_initials'] = True
                original = engine.evaluate
                def evaluate(code, cases, *args, **kwargs):
                    result = original(code, cases, *args, **kwargs)
                    if 'return 10;' in code or 'return 11;' in code:
                        result['valid'] = False
                        for row in result['cases']:
                            row['status'] = 'CE'
                    return result
                engine.evaluate = evaluate
                engine.run()
                state = engine.s['initialization']
                self.assertEqual(state['status'], 'complete')
                self.assertEqual(len(state['invalid']), 2)
                self.assertTrue(state['invalid_originals_accepted'])
                self.assertEqual(len(state['completed']), 4)
                self.assertFalse(any(e['arm'] == 'independent' for e in engine.s['events']))
                self.assertGreater(engine.provider.requests, 0)
                for item in fixed_bank()[0]:
                    self.assertEqual(engine.source(item['source_sha256']), item['code'])
                engine.finalize()

    def test_resume_saved_batch_does_not_resend_or_reselect(self):
        with tempfile.TemporaryDirectory() as root:
            engine = self.make(R0Engine, root, calls=1)
            engine.initialize()
            original = engine.evaluate
            engine.evaluate = lambda *a, **k: (_ for _ in ()).throw(OSError('interrupted'))
            with self.assertRaisesRegex(OSError, 'interrupted'):
                engine.run()
            self.assertEqual(self.http_count, 4)
            resumed = self.make(R0Engine, root, calls=1)
            resumed.evaluate = original
            with patch.object(parents, 'select_parents', side_effect=AssertionError('No reselection')):
                resumed.run()
            self.assertEqual(self.http_count, 4)
            self.assertEqual(len(resumed.s['events']), 8)
            resumed.finalize()

    def test_token_budget_reserves_whole_batch_and_allows_freeze(self):
        with tempfile.TemporaryDirectory() as root:
            engine = self.make(R0Engine, root, tokens=1)
            engine.run()
            self.assertEqual(engine.provider.requests, 0)
            self.assertTrue(engine.s['parallel_batch']['budget_exhausted'])
            engine.finalize()

    def test_framework_directly_calls_all_six_components_and_variable_para(self):
        with tempfile.TemporaryDirectory() as root:
            engine = self.make(R0Engine, root, calls=1)
            with patch.object(parallelism, 'select_para', return_value=2) as width, \
                 patch.object(parents, 'select_parents', wraps=parents.select_parents) as parent_fn, \
                 patch.object(operator_selection, 'select_operators', wraps=operator_selection.select_operators) as select_fn, \
                 patch.object(operators, 'define_operator', wraps=operators.define_operator) as define_fn, \
                 patch.object(memory, 'build_context', wraps=memory.build_context) as context_fn, \
                 patch.object(memory, 'update_memory', wraps=memory.update_memory) as memory_fn, \
                 patch.object(population, 'update_population', wraps=population.update_population) as pop_fn:
                engine.run()
                for mocked in (width, parent_fn, select_fn, define_fn, context_fn, memory_fn, pop_fn):
                    self.assertTrue(mocked.called)
                self.assertEqual(parent_fn.call_count, 1)  # Joint, not per-slot selection.
                self.assertEqual(select_fn.call_count, 1)
            self.assertEqual(engine.provider.requests, 2)
            self.assertEqual(len(engine.initials), 4)
            engine.finalize()

    def test_batch_memory_is_deferred_and_resume_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            engine = self.make(R0Engine, root, calls=1)
            def interrupt(batch):
                self.assertFalse(any(r.get('failure_experience') for r in engine.s['records'].values()))
                raise OSError('before memory commit')

            engine.finish_batch = interrupt
            with self.assertRaisesRegex(OSError, 'memory commit'):
                engine.run()
            resumed = self.make(R0Engine, root, calls=1)
            resumed.run()
            self.assertEqual(self.http_count, 4)
            histories = [r.get('failure_experience', []) for r in resumed.s['records'].values()]
            self.assertTrue(any(histories))
            for history in histories:
                self.assertEqual(len(history), len({e['call_id'] for e in history}))

    def test_population_matches_baseline_and_snapshots_cannot_mutate_records(self):
        with tempfile.TemporaryDirectory() as root:
            engine = self.make(R0Engine, root)
            engine.run()
            self.assertEqual(engine.s['population'], pool(engine.s['records']))
            view = engine.view()
            view['records'].clear()
            self.assertTrue(engine.s['records'])

    def test_changed_source_or_component_identity_rejects_resume(self):
        with tempfile.TemporaryDirectory() as root:
            engine = self.make(R0Engine, root)
            engine.initialize()
            key = engine.initials[0]['source_sha256']
            (Path(root) / 'candidates' / key / 'solution.cpp').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                self.make(R0Engine, root).initialize()
            with self.assertRaisesRegex(ValueError, 'Protocol changed'):
                self.make(FixedInitialEngine, root)

    def test_unknown_reservation_is_counted_and_never_resent(self):
        from aad.io import save_json
        with tempfile.TemporaryDirectory() as root:
            engine = self.make(R0Engine, root, calls=1)
            engine.initialize()
            engine.update_population()
            engine.s.update(requested_calls=1, effective_calls=1)
            engine.start_batch('evolution', engine.round_tasks())
            save_json(engine.provider.path, [{'id': 0, 'kind': 'edit', 'status': 'started',
                                             'reserved_tokens': 100}])
            resumed = self.make(R0Engine, root, calls=1)
            resumed.run()
            self.assertEqual(self.http_count, 3)
            self.assertEqual(resumed.provider.requests, 4)
            self.assertEqual(resumed.provider.charged_tokens, 130)
            self.assertEqual(resumed.s['events'][4]['status'], 'failed')
            resumed.finalize()

    def test_invalid_proposals_keep_same_parent_memory_as_baseline(self):
        respond = self.respond

        def invalid(request, **kwargs):
            prompt = json.loads(json.loads(request.data)['messages'][1]['content'])
            if prompt['operation'] == 'redesign':
                return Response(json.dumps({'hypothesis': 'missing code', 'is_randomized': False}))
            return respond(request, **kwargs)

        with patch('aad.parallel_search.urllib.request.urlopen', side_effect=invalid):
            self.compare_runs()

    def test_stop_marker_prevents_initial_evaluation_and_requests(self):
        for cls in (FixedInitialEngine, R0Engine):
            with tempfile.TemporaryDirectory() as root:
                engine = self.make(cls, root)
                (Path(root) / 'STOP_AFTER_CURRENT').touch()
                engine.evaluate = lambda *a, **k: self.fail('No evaluation after stop')
                engine.run()
                self.assertEqual(engine.provider.requests, 0)
                self.assertEqual(engine.s['records'], {})

    def test_new_operator_definition_and_selection_need_no_engine_edit(self):
        with tempfile.TemporaryDirectory() as root:
            engine = self.make(R0Engine, root, calls=1)
            extra = {'kind': 'program', 'repair': None, 'instruction': 'Try a new representation.'}
            with patch.dict(operators.OPERATORS, {'new_operator': extra}), \
                 patch.object(operator_selection, 'select_operators', return_value=['new_operator'] * 4):
                engine.run()
            self.assertEqual([e['arm'] for e in engine.s['events'][4:]], ['new_operator'] * 4)
            self.assertEqual(engine.provider.requests, 4)


class CanonicalBankTests(unittest.TestCase):
    def test_real_bank_reader_supports_batch_and_reactive(self):
        from tools.fixed_initials import DEFAULT_BANK
        if not (DEFAULT_BANK / 'resolved_manifest.json').is_file():
            self.skipTest('External canonical initial bank is not included in the source release')
        initials, plan = read_bank('ahc001')
        self.assertEqual([i['initial_direction'] for i in initials], list(range(4)))
        self.assertEqual(len(plan['directions']), 4)
        for item in initials:
            self.assertEqual(digest(item['code']), item['source_sha256'])
        with self.assertRaises(ValueError):
            read_bank('missing_problem')
        reactive, _ = read_bank('ahc008')
        self.assertEqual(len(reactive), 4)
        self.assertEqual([r['initial_direction'] for r in reactive], list(range(4)))

    def test_bank_hash_mismatch_has_no_generation_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'resolved_manifest.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                read_bank('ahc001', Path(root))


if __name__ == '__main__':
    unittest.main()
