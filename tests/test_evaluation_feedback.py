from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from aad.parallel_engine import V51Engine
from aad.search_policy.engine import R0Engine
from aad.meta_online.engine import OnlineEngine
from aad.evaluation_feedback import ERROR_CHARS, score_result


class FeedbackTests(unittest.TestCase):
    def engine(self, status='AC'):
        rows = [{'name': str(i), 'raw_score': 10 + i, 'score': 10 + i, 'seconds': .2,
                 'seed': 0, 'status': status, 'stdout': 'RAW_OUTPUT_SENTINEL' * 50000,
                 'input': 'RAW_INPUT_SENTINEL',
                 'message': 'compiler error: line 17 ' + 'detail ' * 10000 + ' final diagnostic'}
                for i in range(3)]
        result = {'valid': status == 'AC', 'mean': 11, 'raw_mean': 11, 'cases': rows}
        engine = R0Engine.__new__(R0Engine)
        engine.config = {'direction': 'maximize', 'time_limit': 2}
        engine.provider = None
        engine.s = {'records': {'p': {'is_randomized': True, 'full': result, 'smoke': result,
                                     'evaluations': [result], 'runtime_ratio': .1}},
                    'events': [], 'population': ['p'], 'incumbent': 'p'}
        return engine

    def test_success_retains_all_scores_timing_but_no_io_or_messages(self):
        engine = self.engine()
        before = deepcopy(engine.s)
        feedback = engine.feedback('p')
        self.assertEqual(feedback['train_mean'], 11)
        self.assertEqual([r['raw_score'] for r in feedback['case_scores']], [10, 11, 12])
        self.assertEqual(feedback['timing']['runtime_ratio'], .1)
        self.assertLess(len(json.dumps(feedback)), 2500)
        self.assertNotIn('RAW_', json.dumps(feedback))
        self.assertNotIn('compiler error', json.dumps(feedback))
        self.assertEqual(engine.s, before)

    def test_failure_retains_bounded_diagnostics_without_raw_io(self):
        for status in ('CE', 'WA', 'RE', 'TLE'):
            engine = self.engine(status)
            feedback = engine.feedback('p')
            self.assertTrue(feedback['failed_evaluation'])
            self.assertFalse(feedback['valid'])
            self.assertNotIn('RAW_', json.dumps(feedback))
            self.assertLess(len(json.dumps(feedback)), 6000)
            for row in feedback['examples']:
                self.assertEqual(row['status'], status)
                self.assertLessEqual(len(row['message']), ERROR_CHARS)
                self.assertIn('compiler error: line 17', row['message'])
                self.assertIn('final diagnostic', row['message'])

    def test_smoke_only_failure_is_marked_failed(self):
        engine = self.engine('CE')
        del engine.s['records']['p']['full']
        engine.s['records']['p']['evaluations'] = []
        self.assertTrue(engine.feedback('p')['failed_evaluation'])

    def test_policy_view_and_meta_evidence_do_not_expose_raw_measurements(self):
        engine = self.engine('WA')
        before = deepcopy(engine.s)
        view = engine.view()
        self.assertNotIn('RAW_', json.dumps(view))
        result = view['records']['p']['full']
        self.assertEqual(sum('message' in r for r in result['cases']), 2)
        fake = SimpleNamespace(meta={'versions': [], 'active': 'baseline', 'updates': [], 'rounds': []},
                               s=engine.s, view=engine.view, config=engine.config, meta_interval=4)
        evidence = OnlineEngine.evidence(fake)
        self.assertNotIn('RAW_', json.dumps(evidence))
        self.assertEqual(evidence['public_records']['p']['public_full']['raw_mean'], 11)
        self.assertEqual(engine.s, before)
        result['cases'][0]['score'] = -1
        self.assertEqual(engine.s['records']['p']['full']['cases'][0]['score'], 10)


if __name__ == '__main__':
    unittest.main()
