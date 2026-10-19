import json
from pathlib import Path
import re
import tempfile
import threading
import urllib.error
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aad.io import read_json
from aad.problem import Case
from aad.ale_batch import summarize
from aad.parallel_engine import V51Engine, V51Provider
from aad.parallel_search import round_tasks, start_batch
from aad.search_plan import PLAN_RULE


class Response:
    def __init__(self, content):
        self.payload = {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
                        'usage': {'total_tokens': 10}}

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, *args): return json.dumps(self.payload).encode()


class ParallelTests(unittest.TestCase):
    def test_runtime_override_uses_best_parent_inclusive_threshold_and_round_parity(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=self.transport()):
            engine = self.make(root, calls=4)
            engine.run()
            best = max(engine.s['records'], key=engine.mean)
            for ratio in (0.0, 0.30):
                engine.s['records'][best]['runtime_ratio'] = ratio
                _, tasks = round_tasks(engine)
                self.assertEqual(tasks[3]['arm'], 'runtime')
                self.assertEqual(tasks[3]['parent'], best)
                self.assertNotIn('donor', tasks[3])
                self.assertEqual([t['arm'] for t in tasks[:2]], ['local', 'redesign'])
            engine.s['records'][best]['runtime_ratio'] = 0.300001
            _, tasks = round_tasks(engine)
            self.assertEqual(tasks[3]['arm'], 'redesign')
            _, tasks = round_tasks(engine)
            self.assertEqual(tasks[3]['arm'], 'crossover')
            self.assertEqual(tasks[3]['donor'], best)

    def make(self, root, calls=9, plan=False, tokens=0):
        provider = V51Provider(root, 'mock', calls, tokens, key='test', problem='Test batch task')
        cases = [Case(str(i), f'input {i}') for i in range(3)]

        def evaluate(code, subset, seconds, replica, *, seed=0):
            score = int(re.search(r'return (\d+)', code)[1])
            items = [SimpleNamespace(judge_result=SimpleNamespace(value='ACCEPTED'),
                                     absolute_score=score, execution_time=1.5, output_str='ok',
                                     message='', error_str='') for _ in subset]
            return summarize(items, subset, 'maximize')

        return V51Engine(root, provider, evaluate, cases,
                         {'para': 4, 'plan': plan, 'direction': 'maximize', 'time_limit': 2,
                          'calls': calls, 'token_budget': tokens})

    def transport(self, barrier=None, invalid_calls=()):
        lock = threading.Lock()
        self.count = 0
        self.bodies = []

        def respond(request, **kwargs):
            body = json.loads(request.data)
            if body['messages'][0]['content'] == PLAN_RULE:
                return Response(json.dumps({'directions': [f'Use direction {i}.' for i in range(4)]}))
            with lock:
                self.count += 1
                number = self.count
                self.bodies.append(body)
            if barrier:
                barrier.wait(timeout=5)
            if number in invalid_calls:
                return Response('not JSON')
            prompt = json.loads(body['messages'][1]['content'])
            code = f'int main(){{return {number};}}\n// AAD_V5_COMPLETE\n'
            result = {'hypothesis': 'Try this design', 'is_randomized': False}
            if prompt['operation'] == 'local':
                result.update(parent_sha256=prompt['parent_sha256'],
                              edits=[{'old': prompt['parent_source'], 'new': code}])
            else:
                result['code'] = code
            return Response(json.dumps(result))
        return respond

    def test_concurrent_rounds_snapshot_parent_donor_alternation_and_budget(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=self.transport(threading.Barrier(4))):
            engine = self.make(root)
            engine.run()
            self.assertEqual(engine.provider.requests, 12)  # 9 rounds up to 12
            self.assertEqual(engine.provider.charged_tokens, 120)
            events = engine.s['events']
            self.assertEqual([e['implementation_call'] for e in events], list(range(12)))
            roots = [e['candidate'] for e in events[:4]]
            best = max(roots, key=engine.mean)
            self.assertEqual([e['parent'] for e in events[4:7]], [best] * 3)
            self.assertEqual([e['arm'] for e in events[4:6]], ['local', 'redesign'])
            self.assertIn(events[6]['arm'], ('local', 'redesign'))  # dominated donor fallback
            self.assertEqual(events[7]['arm'], 'redesign')
            self.assertNotEqual(engine.s['records'][events[7]['parent']]['lineage'], best)
            self.assertEqual(events[11]['arm'], 'crossover')
            self.assertEqual(events[11]['donor'], events[8]['parent'])
            self.assertNotEqual(engine.s['records'][events[11]['parent']]['lineage'],
                                engine.s['records'][events[11]['donor']]['lineage'])
            self.assertEqual(len({r['lineage'] for r in engine.s['records'].values()}), 4)
            engine.finalize()
            report = read_json(Path(root) / 'report.json')
            self.assertEqual((report['requested_calls'], report['effective_calls']), (9, 12))

    def test_plan_initial_hints_are_independent_and_missing_slot_retried(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=self.transport(invalid_calls=(2,))):
            engine = self.make(root, calls=6, plan=True)
            engine.run()
            events = engine.s['events']
            self.assertEqual([e['initial_slot'] for e in events[:4]], list(range(4)))
            failed_slot = next(e['initial_slot'] for e in events[:4] if e['status'] == 'failed')
            self.assertEqual(events[4]['initial_slot'], failed_slot)
            self.assertEqual(engine.provider.requests, 9)  # 4 + retry 1 + evolution 4
            self.assertEqual(engine.provider.planner.requests, 1)
            for event in events[:5]:
                body = read_json(engine.provider.folder / f"{event['implementation_call']:05d}.request.json")
                prompt = json.loads(body['messages'][1]['content'])
                self.assertNotIn('parent_source', prompt)
                self.assertTrue(prompt['instruction'].endswith(f"Use direction {event['initial_slot']}."))
            engine.finalize()

    def test_complementary_donor_and_other_lineage_representatives(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=self.transport()):
            engine = self.make(root, calls=4)
            engine.run()
            records = engine.s['records']
            ranked = sorted(records, key=engine.mean, reverse=True)
            # Give third place a uniquely strong per-case advantage.
            records[ranked[2]]['full']['cases'][0]['score'] = 100
            phase, tasks = round_tasks(engine)
            self.assertEqual(phase, 'evolution')
            self.assertEqual(tasks[2]['arm'], 'crossover')
            self.assertEqual(tasks[2]['donor'], ranked[2])
            self.assertIn(tasks[3]['parent'], ranked[1:])
            _, tasks = round_tasks(engine)
            self.assertEqual((tasks[3]['arm'], tasks[3]['donor']), ('crossover', ranked[0]))

    def test_saved_responses_resume_without_extra_http(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=self.transport()):
            engine = self.make(root, calls=4)
            original = engine.evaluate
            engine.evaluate = lambda *args, **kwargs: (_ for _ in ()).throw(OSError('evaluation interrupted'))
            with self.assertRaisesRegex(OSError, 'interrupted'):
                engine.run()
            self.assertEqual(self.count, 4)
            resumed = self.make(root, calls=4)
            resumed.evaluate = original
            resumed.run()
            self.assertEqual(self.count, 4)
            self.assertEqual(len(resumed.s['events']), 4)
            resumed.finalize()

    def test_unknown_reserved_call_not_retried_and_missing_slot_gets_new_call(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=self.transport()):
            engine = self.make(root, calls=5)
            engine.s.update(requested_calls=5, effective_calls=5)
            phase, tasks = round_tasks(engine)
            start_batch(engine, phase, tasks)
            # Simulate one HTTP reservation with unknown outcome before a crash.
            from aad.io import save_json
            save_json(engine.provider.path, [{'id': 0, 'kind': 'program', 'status': 'started',
                                             'reserved_tokens': 100}])
            resumed = self.make(root, calls=5)
            resumed.run()
            self.assertEqual(self.count, 4)  # three remaining initial calls + missing slot retry
            self.assertEqual(resumed.provider.requests, 5)
            self.assertEqual(resumed.s['events'][0]['status'], 'failed')
            resumed.finalize()

    def test_invalid_candidates_all_get_one_repair_and_budget_records_overrun(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=self.transport()):
            engine = self.make(root, calls=4)
            evaluate = engine.evaluate

            def invalid_initial(code, subset, seconds, replica, *, seed=0):
                result = evaluate(code, subset, seconds, replica, seed=seed)
                if int(re.search(r'return (\d+)', code)[1]) <= 4:
                    result['valid'] = False
                    for row in result['cases']:
                        row['status'] = 'WA'
                return result

            engine.evaluate = invalid_initial
            engine.run()
            self.assertEqual(engine.provider.requests, 8)
            self.assertEqual([e['arm'] for e in engine.s['events'][4:]], ['repair'] * 4)
            self.assertEqual(len({r['lineage'] for r in engine.s['records'].values()}), 4)
            engine.finalize()

    def test_token_limit_atomic_reservation_and_incomplete_freeze(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen') as http:
            engine = self.make(root, calls=1, tokens=100000)
            engine.run()
            http.assert_not_called()
            self.assertEqual(engine.provider.requests, 0)
            with self.assertRaises(ValueError):
                engine.finalize()

    def test_unsubmitted_token_limited_round_can_freeze_existing_incumbent(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=self.transport()):
            engine = self.make(root, calls=4)
            engine.run()
            engine.s['requested_calls'] = 5
            engine.provider.max_tokens_total = 1
            engine.run()
            self.assertTrue(engine.s['parallel_batch']['budget_exhausted'])
            self.assertEqual(engine.provider.requests, 4)
            engine.finalize()
            self.assertIsNone(engine.s['parallel_batch'])
            self.assertIn('unsubmitted_batch', engine.s)

    def test_failed_http_does_not_discard_other_responses_and_is_charged(self):
        respond = self.transport()
        lock = threading.Lock()
        attempts = 0

        def flaky(request, **kwargs):
            nonlocal attempts
            with lock:
                attempts += 1
                number = attempts
            if number == 1:
                raise urllib.error.URLError('offline')
            return respond(request, **kwargs)

        with tempfile.TemporaryDirectory() as root, patch('aad.parallel_search.urllib.request.urlopen',
                side_effect=flaky):
            engine = self.make(root, calls=5)
            engine.run()
            self.assertEqual(engine.provider.requests, 5)
            failed = [r for r in engine.provider.ledger if r['status'] == 'network_error']
            self.assertEqual(len(failed), 1)
            self.assertEqual(engine.provider.charged_tokens, failed[0]['reserved_tokens'] + 40)
            self.assertEqual(len(engine.s['records']), 4)
            engine.finalize()

    def test_stop_after_current_keeps_next_batch_unstarted(self):
        with tempfile.TemporaryDirectory() as root:
            respond = self.transport(threading.Barrier(4))

            def stopping(request, **kwargs):
                result = respond(request, **kwargs)
                (Path(root) / 'STOP_AFTER_CURRENT').touch()
                return result

            with patch('aad.parallel_search.urllib.request.urlopen', side_effect=stopping):
                engine = self.make(root, calls=8)
                engine.run()
            self.assertEqual(engine.provider.requests, 4)
            self.assertEqual(len(engine.s['events']), 4)
            self.assertIsNone(engine.s['parallel_batch'])

    def test_para_validation(self):
        with tempfile.TemporaryDirectory() as root:
            for value in (0, 1, 2, 3, 5, True, '4', 4.0):
                with self.assertRaisesRegex(ValueError, 'para'):
                    V51Engine(root, None, None, [], {'para': value})


if __name__ == '__main__':
    unittest.main()
