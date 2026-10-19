import random
import unittest

from aad.meta_online import BASELINE
from aad.meta_online.git_store import PolicyStore
from aad.meta_online.initial_policy import starting_files
from aad.meta_online.policy import components
from pathlib import Path
from tempfile import TemporaryDirectory
from run_rsi import build_args


class ReleaseEntryTests(unittest.TestCase):
    def test_optional_manifest_never_supplies_code(self):
        from aad.meta_online.dsh import parse_final_decision
        self.assertEqual(parse_final_decision('{"action":"update","reason":"public evidence","files":["parents.py"]}'),
                         {'action': 'update', 'reason': 'public evidence'})
        with self.assertRaises(ValueError):
            parse_final_decision('{"action":"update","reason":"x","files":{"parents.py":"code"}}')

    def test_options_select_actual_policy(self):
        for origin in ('greedy', 'diversity'):
            args = build_args(['--from', origin, '--calls', '100'])
            self.assertEqual(args[args.index('--initial-policy') + 1], origin)
            self.assertEqual(args[args.index('--meta-interval') + 1], '12')
            policy = components(starting_files(origin))
            view = {'population': ['a', 'b'], 'incumbent': 'a'}
            selected = policy.parents.select_parents(view, 100, random.Random(0))
            self.assertEqual(set(selected), {'a'} if origin == 'greedy' else {'a', 'b'})

    def test_diversity_preserves_historical_minimal_policy(self):
        self.assertEqual(starting_files('diversity'), starting_files('minimal'))

    def test_explicit_meta_interval(self):
        args = build_args(['--from', 'greedy', '--meta-interval=6'])
        self.assertNotIn('--meta-interval', args)
        self.assertIn('--meta-interval=6', args)

    def test_conflicting_options_rejected(self):
        for option in ('--method=5.2', '--initial-policy=minimal'):
            with self.assertRaises(SystemExit):
                build_args(['--from', 'greedy', option])

    def test_release_baseline_creates_both_policy_worktrees(self):
        project = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as root:
            store = PolicyStore(project, root)
            for index, origin in enumerate(('greedy', 'diversity')):
                revision = store.create(index, BASELINE, starting_files(origin))
                self.assertEqual(store.verify(index, revision), starting_files(origin))


if __name__ == '__main__':
    unittest.main()
