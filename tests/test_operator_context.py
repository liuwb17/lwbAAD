"""Regression: runtime operator context must match online preflight."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from aad.meta_online.engine import OnlineEngine
from aad.search_policy.engine import R0Engine

class OperatorContextTests(unittest.TestCase):
    def test_runtime_crossover_receives_public_donor_source(self):
        engine = object.__new__(OnlineEngine)
        engine.s = {}
        view = {'population': ['parent', 'donor'], 'round': 12}
        engine.view = lambda: deepcopy(view)
        engine.source = lambda k: 'code-' + k
        engine.feedback = lambda k: {'score': 1}
        def define(v, arm, parent):
            return {'parent': parent, 'arm': arm if v.get('sources', {}).get('donor') else 'mutation'}
        engine.components = SimpleNamespace(
            parallelism=SimpleNamespace(select_para=lambda v: 1),
            parents=SimpleNamespace(select_parents=lambda v,n,r: ['parent']),
            operator_selection=SimpleNamespace(select_operators=lambda v,p,r: ['crossover']),
            operators=SimpleNamespace(define_operator=define))
        self.assertEqual(engine.round_tasks()[0]['arm'], 'crossover')
        self.assertEqual(engine.s['parallel_round'], 13)
        self.assertNotIn('sources', view)

    def test_fixed_context_unchanged(self):
        view = {'population': ['parent']}
        self.assertIs(R0Engine.operator_view(object.__new__(R0Engine), view), view)
