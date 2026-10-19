import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aad.io import read_json
from aad.problem import Case
from aad.parallel_engine import V51Engine, V51Provider
from aad.candidate_evaluation import from_rows
from aad.search_memory import augment_task, note_response, record_attempt


def record(mean, valid=True):
    return {'full': {'valid': valid, 'mean': mean,
                     'cases': [{'status': 'AC' if valid else 'TLE'}]}}


class ExperienceTests(unittest.TestCase):
    def engine(self, direction='maximize'):
        return SimpleNamespace(config={'failure_experience': True, 'direction': direction},
                               sign=1 if direction == 'maximize' else -1,
                               s={'records': {'parent': record(10), 'child': record(10),
                                              'donor': record(100)}})

    def attempt(self, call=0, **extra):
        return {'parent': 'parent', 'candidate': 'child', 'donor': 'donor',
                'arm': 'crossover', 'implementation_call': call, 'status': 'archived',
                'attempt_hypothesis': f'Try a larger neighborhood {call}.', **extra}

    def test_recent_eight_parent_only_and_snapshot_does_not_change(self):
        engine = self.engine()
        for call in range(10):
            record_attempt(engine, self.attempt(call))
        history = engine.s['records']['parent']['failure_experience']
        self.assertEqual([e['call_id'] for e in history], list(range(2, 10)))
        self.assertNotIn('failure_experience', engine.s['records']['child'])
        self.assertNotIn('failure_experience', engine.s['records']['donor'])
        prompt = augment_task(engine, self.attempt(), {'instruction': 'Improve it.'})
        record_attempt(engine, self.attempt(10))
        record_attempt(engine, self.attempt(10))  # resume must not duplicate the entry
        self.assertEqual(len(history), 8)
        self.assertEqual(prompt['parent_failure_experience'][-1]['call_id'], 9)
        self.assertIn('one concise sentence', prompt['instruction'])

    def test_compare_main_parent_signed_scores_not_global_incumbent_or_margin(self):
        for direction in ('maximize', 'minimize'):
            engine = self.engine(direction)
            engine.config['promotion_margin'] = 1000
            engine.s['records']['child'] = record(11)
            record_attempt(engine, self.attempt())
            self.assertNotIn('failure_experience', engine.s['records']['parent'])
            engine.s['records']['child'] = record(9)
            record_attempt(engine, self.attempt(1))
            entry = engine.s['records']['parent']['failure_experience'][0]
            self.assertEqual(entry['child_mean'], 9 * engine.sign)
            self.assertEqual(entry['parent_mean'], 10 * engine.sign)
            self.assertEqual(entry['outcome'], 'no_improvement')

    def test_invalid_repair_duplicate_and_missing_response(self):
        engine = self.engine()
        engine.s['records']['child'] = record(0, valid=False)
        record_attempt(engine, self.attempt(status='invalid_candidate'))
        self.assertEqual(engine.s['records']['parent']['failure_experience'][0]['evaluation_statuses'], ['TLE'])
        engine.s['records']['parent']['full']['valid'] = False
        engine.s['records']['child'] = record(1)
        record_attempt(engine, self.attempt(1, arm='repair'))
        self.assertEqual(len(engine.s['records']['parent']['failure_experience']), 1)
        pending = self.attempt(2, candidate='parent', status='duplicate')
        engine.s['records']['parent']['full']['valid'] = True
        note_response(engine, pending, json.dumps({'hypothesis': 'A different attempted change.'}))
        record_attempt(engine, pending)
        self.assertEqual(engine.s['records']['parent']['failure_experience'][-1]['hypothesis'],
                         'A different attempted change.')
        pending = self.attempt(3, status='failed', candidate=None)
        del pending['attempt_hypothesis']
        note_response(engine, pending, 'not JSON')
        record_attempt(engine, pending)
        self.assertEqual(len(engine.s['records']['parent']['failure_experience']), 2)
        note_response(engine, pending, json.dumps({'hypothesis': 'Try a new representation.', 'code': 'broken'}))
        record_attempt(engine, pending)
        self.assertEqual(engine.s['records']['parent']['failure_experience'][-1]['outcome'], 'invalid_proposal')

    def test_serial_and_parallel_prompts_enabled_disabled_and_persisted(self):
        for para in (None, 4):
            for enabled in (False, True):
                with self.subTest(para=para, enabled=enabled), tempfile.TemporaryDirectory() as root:
                    lock = threading.Lock()
                    count = 0

                    def transport(request, **kwargs):
                        nonlocal count
                        with lock:
                            count += 1
                            number = count
                        body = json.loads(request.data)
                        task = json.loads(body['messages'][1]['content'])
                        result = {'hypothesis': f'Try mechanism {number}.', 'is_randomized': False}
                        code = f'int main(){{return {number};}}\n// AAD_V5_COMPLETE\n'
                        if task['operation'] == 'local':
                            result.update(parent_sha256=task['parent_sha256'],
                                          edits=[{'old': task['parent_source'], 'new': code}])
                        else:
                            result['code'] = code
                        payload = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}],
                                   'usage': {'total_tokens': 1}}
                        class Response:
                            def __enter__(self): return self
                            def __exit__(self, *args): pass
                            def read(self, *args): return json.dumps(payload).encode()
                        return Response()

                    def evaluate(code, cases, seconds, replica, *, seed):
                        return from_rows([{'name': c.name, 'status': 'AC', 'score': 10, 'raw_score': 10,
                                           'seconds': 1.5, 'stdout': '', 'message': ''} for c in cases])

                    calls = 12 if para == 4 else 4
                    provider = V51Provider(root, 'mock', calls, 0, key='test', problem='Test problem')
                    engine = V51Engine(root, provider, evaluate, [Case('a', 'input')],
                                       {'para': para, 'failure_experience': enabled, 'direction': 'maximize',
                                        'time_limit': 2, 'initial_valid_roots': 4 if para else 1})
                    with patch('urllib.request.urlopen', side_effect=transport):
                        engine.run()
                    self.assertEqual(provider.requests, calls)
                    nonempty = []
                    for event in engine.s['events']:
                        call = event['implementation_call']
                        body = read_json(provider.folder / f'{call:05d}.request.json')
                        prompt = json.loads(body['messages'][1]['content'])
                        if not enabled or not event['parent']:
                            self.assertNotIn('parent_failure_experience', prompt)
                            continue
                        memories = prompt['parent_failure_experience']
                        if memories:
                            nonempty.append(memories)
                        for memory in memories:
                            previous = engine.s['events'][memory['call_id']]
                            self.assertEqual(previous['parent'], event['parent'])
                            self.assertLess(memory['call_id'], call // 4 * 4 if para else call)
                    self.assertEqual(bool(nonempty), enabled)
                    saved = read_json(Path(root) / 'state.json')
                    self.assertEqual(saved['records'], engine.s['records'])
                    if not enabled:
                        self.assertTrue(all('failure_experience' not in r for r in saved['records'].values()))

    def test_strict_boolean_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            for value in (0, 1, 'true', None):
                with self.assertRaisesRegex(ValueError, 'failure_experience'):
                    V51Engine(root, None, None, [], {'failure_experience': value})
