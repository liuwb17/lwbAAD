import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aad.io import read_json, save_json
from aad.problem import Case
from aad.ale_batch import summarize
from aad.parallel_engine import V51Engine, V51Provider
from aad.search_plan import PLAN_RULE, ensure_plan, parse_directions


DIRECTIONS = ['Use greedy construction with local exchanges.',
              'Use a dynamic programming representation.',
              'Use beam search over partial solutions.',
              'Use randomized construction with large neighborhood search.']


class Response:
    def __init__(self, content, finish='stop'):
        self.payload = {'choices': [{'finish_reason': finish, 'message': {'content': content}}],
                        'usage': {'total_tokens': 10}}

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, *args): return json.dumps(self.payload).encode()


class PlanTests(unittest.TestCase):
    def make(self, root, enabled=True, calls=5):
        cases = [Case(f'public_{i}', f'input {i}') for i in range(3)]
        provider = V51Provider(root, 'mock', calls, 0, key='test', problem='A non-geometric test problem.')
        def evaluate(code, subset, seconds, replica, *, seed=0):
            items = [SimpleNamespace(judge_result=SimpleNamespace(value='ACCEPTED'),
                                     absolute_score=10, execution_time=1.5, output_str='ok',
                                     message='', error_str='') for _ in subset]
            return summarize(items, subset, 'maximize')
        return V51Engine(root, provider, evaluate, cases,
                         {'plan': enabled, 'direction': 'maximize', 'time_limit': 2})

    def transport(self, bodies):
        def respond(request, **kwargs):
            body = json.loads(request.data)
            bodies.append(body)
            if body['messages'][0]['content'] == PLAN_RULE:
                return Response(json.dumps({'directions': DIRECTIONS}))
            task = json.loads(body['messages'][1]['content'])
            code = f'int main(){{return {len(bodies)};}}\n// AAD_V5_COMPLETE\n'
            if task['operation'] == 'local':
                code = json.dumps({'hypothesis': 'try change', 'is_randomized': False, 'parent_sha256': task['parent_sha256'],
                                   'edits': [{'old': task['parent_source'], 'new': code}]})
            else:
                code = json.dumps({'hypothesis': 'try design', 'is_randomized': False, 'code': code})
            return Response(code)
        return respond

    def test_four_hints_one_plan_no_shared_sources_and_no_later_hint(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root, calls=5)
            engine.run()
            self.assertEqual([r['kind'] for r in engine.provider.ledger],
                             ['program', 'program', 'program', 'program', 'edit'])
            self.assertEqual(engine.provider.requests, 5)
            self.assertEqual(engine.provider.charged_tokens, 50)
            self.assertEqual(engine.provider.planner.requests, 1)
            self.assertEqual(engine.provider.planner.charged_tokens, 10)
            self.assertEqual(json.loads(bodies[0]['messages'][1]['content']),
                             {'problem': 'A non-geometric test problem.'})
            for i, body in enumerate(bodies[1:5]):
                task = json.loads(body['messages'][1]['content'])
                self.assertEqual(set(task), {'problem', 'operation', 'instruction'})
                self.assertTrue(task['instruction'].endswith(DIRECTIONS[i]))
                self.assertNotIn('parent_source', task)
                self.assertNotIn('feedback', task)
                for j, direction in enumerate(DIRECTIONS):
                    if j != i:
                        self.assertNotIn(direction, task['instruction'])
            self.assertEqual([e['initial_direction'] for e in engine.s['events'][:4]], list(range(4)))
            engine.s['schedule_slot'] = 7
            restart = engine.task()
            self.assertEqual(restart['arm'], 'independent')
            self.assertNotIn('Suggested design direction', restart['instruction'])
            self.assertEqual(read_json(Path(root) / 'plan.json')['directions'], DIRECTIONS)
            self.assertEqual(read_json(Path(root) / 'protocol.json')['config']['plan_protocol'],
                             'four-cold-start-directions-2-separate-budget')

    def test_disabled_is_four_independent_calls(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root, enabled=False, calls=4)
            engine.run()
            self.assertEqual(len(bodies), 4)
            self.assertTrue(all(r['kind'] == 'program' for r in engine.provider.ledger))
            self.assertNotIn('plan_call', engine.s)
            self.assertFalse((Path(root) / 'plan.json').exists())
            for body in bodies:
                task = json.loads(body['messages'][1]['content'])
                self.assertNotIn('Suggested design direction', task['instruction'])
                self.assertNotIn('parent_source', task)

    def test_resume_successful_plan_before_artifact_save_without_new_call(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root)
            engine.s['plan_call'] = 0
            engine.save()
            engine.provider.planner.request('plan', {})  # Crash before artifact/state write.
            resumed = self.make(root)
            resumed.run()
            self.assertEqual(len(bodies), 6)
            self.assertEqual(resumed.provider.planner.requests, 1)
            again = self.make(root)
            again.run()
            self.assertEqual(len(bodies), 6)

    def test_invalid_plan_stops_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                return_value=Response('{"directions":["only one"]}')) as http:
            engine = self.make(root)
            with self.assertRaisesRegex(ValueError, 'exactly four'):
                engine.run()
            resumed = self.make(root)
            with self.assertRaises(ValueError):
                resumed.run()
            self.assertEqual(http.call_count, 1)
            self.assertEqual(resumed.provider.requests, 0)
            self.assertEqual(resumed.provider.planner.requests, 1)
            self.assertEqual(resumed.s['records'], {})

    def test_failed_transport_stays_charged_and_no_retry(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=TimeoutError) as http:
            engine = self.make(root)
            with self.assertRaisesRegex(RuntimeError, 'Network failure'):
                engine.run()
            resumed = self.make(root)
            with self.assertRaisesRegex(ValueError, 'no automatic retry'):
                resumed.run()
            self.assertEqual(http.call_count, 1)
            self.assertGreater(resumed.provider.planner.charged_tokens, 0)
            self.assertEqual(resumed.provider.charged_tokens, 0)

    def test_stop_budget_and_frozen_gates_before_plan(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen') as http:
            engine = self.make(root)
            stop = Path(root) / 'STOP_AFTER_CURRENT'
            stop.touch()
            engine.run()
            self.assertNotIn('plan_call', engine.s)
            stop.unlink()
            engine.s['frozen'] = True
            with self.assertRaisesRegex(ValueError, 'closed'):
                engine.run()
            http.assert_not_called()

    def test_plan_is_outside_solve_call_budget(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root, calls=1)
            engine.run()
            self.assertEqual(len(bodies), 2)
            self.assertEqual(len(engine.s['records']), 1)
            self.assertEqual(engine.provider.requests, 1)
            self.assertEqual(engine.provider.planner.requests, 1)

    def test_plan_is_outside_solve_token_budget(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root)
            engine.provider.max_tokens_total = 1
            engine.run()
            self.assertEqual(len(bodies), 1)
            self.assertEqual(engine.provider.requests, 0)
            self.assertEqual(engine.provider.charged_tokens, 0)
            self.assertEqual(engine.provider.planner.charged_tokens, 10)
            self.assertEqual(engine.s['pending']['status'], 'budget_exhausted')

    def test_tampered_artifact_and_flag_change_rejected(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root)
            ensure_plan(engine)
            artifact = read_json(Path(root) / 'plan.json')
            artifact['directions'][0] = 'changed'
            save_json(Path(root) / 'plan.json', artifact)
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                self.make(root).run()
            with self.assertRaisesRegex(ValueError, 'Protocol changed'):
                self.make(root, enabled=False)
            self.assertEqual(len(bodies), 1)

    def test_failed_root_retry_only_missing_direction(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root)
            engine.run()
            # Simulate direction 1 becoming ineligible; another cold start should refill it.
            record = next(r for r in engine.s['records'].values() if r['initial_direction'] == 1)
            record['full']['valid'] = False
            task = engine.task()
            self.assertEqual(task['initial_direction'], 1)
            self.assertTrue(task['instruction'].endswith(DIRECTIONS[1]))
            self.assertIsNone(task['parent'])

    def test_plan_schema(self):
        for value in ({}, {'directions': DIRECTIONS[:3]}, {'directions': ['same'] * 4},
                      {'directions': [7, *DIRECTIONS[1:]]},
                      {'directions': ['a\nb', *DIRECTIONS[1:]]},
                      {'directions': ['x' * 501, *DIRECTIONS[1:]]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_directions(json.dumps(value))

    def test_initialization_continues_beyond_twenty_and_cannot_freeze_incomplete(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root, calls=3)
            engine.run()
            engine.s['root_attempts'] = 20
            task = engine.task()
            self.assertEqual(task['arm'], 'independent')
            self.assertEqual(task['initial_direction'], 3)
            self.assertEqual(engine.s['root_attempts'], 21)
            with self.assertRaisesRegex(ValueError, 'four valid initial directions'):
                engine.finalize()

    def test_report_separates_plan_from_solve_requests(self):
        bodies = []
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=self.transport(bodies)):
            engine = self.make(root, calls=4)
            engine.run()
            engine.finalize()
            report = read_json(Path(root) / 'report.json')
            self.assertEqual((report['plan_requests'], report['solve_requests'], report['requests']), (1, 4, 5))

    def test_twenty_one_failed_calls_then_four_valid_roots_within_solve_budget(self):
        bodies = []
        normal = self.transport(bodies)
        def respond(request, **kwargs):
            response = normal(request, **kwargs)
            if 2 <= len(bodies) <= 22:  # Plan is call 1; first 21 solver outputs are invalid.
                return Response('invalid source without completion marker')
            return response
        with tempfile.TemporaryDirectory() as root, patch('aad.evolution_provider.urllib.request.urlopen',
                                                         side_effect=respond):
            engine = self.make(root, calls=25)
            engine.run()
            self.assertEqual(engine.provider.requests, 25)
            self.assertEqual(engine.provider.planner.requests, 1)
            self.assertEqual(len(bodies), 26)
            self.assertEqual(engine.s['root_attempts'], 25)
            self.assertEqual({r['initial_direction'] for r in engine.s['records'].values()}, set(range(4)))
            self.assertEqual(sum(e['status'] == 'failed' for e in engine.s['events']), 21)
            self.assertEqual(engine.s['schedule_slot'], 0)
            engine.finalize()


if __name__ == '__main__':
    unittest.main()
