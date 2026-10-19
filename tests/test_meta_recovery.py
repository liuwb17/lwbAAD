import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch
from aad.io import read_json, save_json
from aad.meta_online.policy import PolicyError
from scripts.meta_recovery import retry_state, run_with_recovery


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        state = {'signature': 'pinned', 'records': {'solver': 'unchanged'}, 'meta': {
            'error': 'Unsupported function declaration', 'active': 'baseline',
            'versions': [{'revision': 'baseline'}], 'pending': {'after_round': 4,
            'parent': 'baseline', 'job': {'kind': 'meta', 'implementation_call': 0,
            'prompt': {'history': ['complete']}}}}}
        ledger = [{'id': 0, 'kind': 'meta', 'status': 'ok', 'reserved_tokens': 100,
                   'usage': {'total_tokens': 70}}]
        return state, ledger

    def test_retry_preserves_paid_history_and_policy_and_full_evidence(self):
        s, l = self.fixture(); before = deepcopy((s, l))
        result = retry_state(s, l)
        self.assertEqual((s, l), before)
        self.assertEqual(result['signature'], s['signature'])
        self.assertEqual(result['records'], s['records'])
        self.assertEqual(result['meta']['active'], 'baseline')
        self.assertEqual(result['meta']['pending']['job']['implementation_call'], 1)
        self.assertEqual(result['meta']['pending']['job']['prompt']['history'], ['complete'])
        self.assertEqual(result['meta']['recovery_history'][0]['charged_tokens'], 70)
        self.assertNotIn('error', result['meta'])

    def test_truncation_retry_is_bounded_and_does_not_reissue_unknown_calls(self):
        s, l = self.fixture(); l[0]['status'] = 'invalid_response'; l[0]['finish_reason'] = 'length'
        result = retry_state(s, l)
        self.assertEqual(result['meta']['pending']['job']['prompt']['meta_recovery']['finish_reason'], 'length')
        s['meta']['recovery_history'] = [{'after_round': 4}] * 3
        continued = retry_state(s, l)
        self.assertIsNone(continued['meta']['pending'])
        self.assertEqual(continued['meta']['handled_round'], 4)
        self.assertEqual(continued['meta']['active'], 'baseline')
        self.assertEqual(continued['meta']['updates'][-1]['status'], 'rejected_retry_limit')
        self.assertNotIn('error', continued['meta'])
        s['meta'].pop('recovery_history'); l[0]['status'] = 'started'
        with self.assertRaises(ValueError): retry_state(s, l)

    def test_malformed_response_feedback_cannot_crash_recovery(self):
        for response in ({'choices': []}, {'choices': [None]}, {'choices': [{'message': None}]}, []):
            s, l = self.fixture(); l[0]['status'] = 'invalid_response'
            with self.subTest(response=response):
                self.assertEqual(retry_state(s, l, response)['meta']['pending']['job']['implementation_call'], 1)

    def test_retry_limit_is_per_round(self):
        s, l = self.fixture()
        s['meta']['recovery_history'] = [{'after_round': 0}] * 4
        self.assertIsNotNone(retry_state(s, l)['meta']['pending'])

    def test_repeated_rejections_continue_experiment_without_resetting_ledger(self):
        s, ledger = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / 'state.json', s)
            save_json(root / 'llm/ledger.json', ledger)

            def run(args):
                state = read_json(root / 'state.json')
                if state['meta']['pending'] is None:
                    self.assertEqual(state['meta']['active'], 'baseline')
                    return 'continued'
                call = state['meta']['pending']['job']['implementation_call']
                if call == len(ledger):
                    ledger.append({**ledger[0], 'id': call, 'status': 'invalid_response'})
                    save_json(root / 'llm/ledger.json', ledger)
                state['meta']['error'] = 'Truncated JSON'
                save_json(root / 'state.json', state)
                raise PolicyError('Rejected')

            with patch('scripts.rsi_search.main', side_effect=run):
                self.assertEqual(run_with_recovery(['run', '--run-dir', directory, '--resume']), 'continued')
            self.assertEqual(len(ledger), 4)
            self.assertEqual(len(list((root / 'meta/recoveries').glob('*.json'))), 4)
            self.assertEqual(read_json(root / 'state.json')['signature'], 'pinned')

    def test_frozen_or_adopted_proposals_are_never_retried(self):
        s, l = self.fixture(); s['frozen'] = True
        with self.assertRaises(ValueError): retry_state(s, l)
        s['frozen'] = False; s['meta']['versions'].append({'call_id': 0})
        with self.assertRaises(ValueError): retry_state(s, l)
