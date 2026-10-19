"""Real local Git/workers, mocked DeepSeek/ALE. No network or solver execution."""
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
from aad.ale_batch import summarize
from aad.parallel_engine import V51Provider
from aad.meta_online import BASELINE, COMPONENTS
from aad.meta_online.engine import OnlineEngine
from aad.meta_online.git_store import PolicyStore, git
from aad.meta_online.policy import validate, components, PolicyError
from aad.meta_online.provider import OnlineProvider, META_RULE
from test_search_policy import fixed_bank, Response

PROJECT = Path(__file__).resolve().parents[1]


class WorkerTests(unittest.TestCase):
    def sources(self):
        return {n + '.py': (PROJECT / 'aad/search_policy/components' / (n + '.py')).read_text(encoding='utf-8')
                for n in COMPONENTS}

    def test_baseline_executes_in_restricted_worker(self):
        sources = self.sources()
        validate(sources)
        self.assertEqual(components(sources).parallelism.select_para({}), 4)

    def test_blocks_network_files_processes_and_reflection(self):
        for source in ('import urllib.request\n', 'import os\n', 'from pathlib import Path\n',
                       'x = (1).__class__\n', 'x = "{0.__class__}".format(1)\n',
                       'x = open("private_report.json")\n', 'x = __import__("socket")\n'):
            with self.subTest(source=source):
                sources = {**self.sources(), 'parallelism.py': source}
                with self.assertRaises(PolicyError):
                    validate(sources)

    def test_no_tools_and_explicit_no_web_and_keep_instruction(self):
        with tempfile.TemporaryDirectory() as root:
            provider = OnlineProvider(root, 'deepseek-flash', 20, 0, key='test', problem='public')
            body = provider.request_body('meta', {'evidence': 'public only'})
            self.assertNotIn('tools', body)
            self.assertIn('MUST NOT search the web', META_RULE)
            self.assertIn('NO automatic score-based rollback', META_RULE)
            self.assertIn('steadily', META_RULE)
            self.assertEqual(body['model'], 'deepseek-flash')
            self.assertEqual(body['reasoning_effort'], 'max')

    def test_meta_defaults_are_independent_and_inner_requests_match_v52(self):
        for cls in (OnlineProvider,):
            with self.subTest(provider=cls.__name__), tempfile.TemporaryDirectory() as root:
                provider = cls(Path(root)/'online', 'solver-model', 20, 0, key='test', problem='public')
                baseline = V51Provider(Path(root)/'baseline', 'solver-model', 20, 0,
                                       key='test', problem='public')
                for kind in ('program', 'edit', 'plan'):
                    self.assertEqual(provider.request_body(kind, {'operation': 'repair'}),
                                     baseline.request_body(kind, {'operation': 'repair'}))
                meta = provider.request_body('meta', {})
                self.assertEqual((meta['model'], meta['reasoning_effort']), ('deepseek-flash', 'max'))

    def test_separate_models_share_ledger_and_recover_without_resending(self):
        with tempfile.TemporaryDirectory() as root:
            options = dict(key='test', problem='public', meta_model='meta-model', meta_reasoning_effort='max')
            provider = OnlineProvider(root, 'solver-model', 20, 0, **options)
            jobs = [{'implementation_call': i, 'kind': kind, 'prompt': {}}
                    for i, kind in enumerate(('program', 'meta'))]
            with patch('aad.parallel_search.urllib.request.urlopen', return_value=Response('{}')) as http:
                provider.request_many(jobs)
                provider.request('meta', {})  # Serial transport also records the request body model.
                self.assertEqual(http.call_count, 3)
            self.assertEqual([r['requested_model'] for r in provider.ledger],
                             ['solver-model', 'meta-model', 'meta-model'])
            self.assertEqual([r['requested_reasoning_effort'] for r in provider.ledger], ['low', 'max', 'max'])
            self.assertEqual(provider.charged_tokens, 30)
            recovered = OnlineProvider(root, 'solver-model', 20, 0, **options)
            with patch('aad.parallel_search.urllib.request.urlopen') as http:
                recovered.request_many(jobs)
                http.assert_not_called()
            self.assertEqual(recovered.charged_tokens, 30)


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.base = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.repo = self.base / 'repo'
        # A private local clone prevents tests from leaving worktree registrations in the real repo.
        common = Path(git(PROJECT, 'rev-parse', '--git-common-dir'))
        if not common.is_absolute():
            common = (PROJECT / common).resolve()
        git(PROJECT, 'clone', '--no-hardlinks', '--no-checkout', str(common.parent), str(self.repo))
        self.main_before = git(self.repo, 'rev-parse', 'main')
        self.stack.enter_context(patch('aad.meta_online.engine.PolicyStore',
                                      side_effect=lambda project, root: PolicyStore(self.repo, root)))
        self.stack.enter_context(patch('aad.fixed_initials.read_bank', side_effect=lambda *a: fixed_bank()))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch('aad.parallel_search.urllib.request.urlopen', side_effect=self.respond))
        self.http_count, self.meta_count = 0, 0
        self.proposals = []
        self.meta_prompts = []
        self.on_meta = None
        self.fail_redesign = False

    def respond(self, request, **kwargs):
        self.http_count += 1
        body = json.loads(request.data)
        prompt = json.loads(body['messages'][1]['content'])
        if 'source_blobs' in prompt and 'round_effects' in prompt:
            self.meta_count += 1
            self.meta_prompts.append(prompt)
            if self.on_meta:
                self.on_meta()
            proposal = self.proposals.pop(0) if self.proposals else {
                'action': 'keep', 'reason': '当前策略仍持续改进，下一次再检查。', 'files': {}}
            return Response(json.dumps(proposal, ensure_ascii=False))
        old = int(re.search(r'return (\d+)', prompt['parent_source'])[1])
        value = old + 1
        code = f'int main(){{return {value};}}\n// {self.http_count}\n// AAD_V5_COMPLETE\n'
        if self.fail_redesign and prompt['operation'] in ('redesign', 'mutation'):
            code = '// FORCE_INVALID\n' + code
        result = {'hypothesis': 'public improvement', 'is_randomized': False}
        if prompt['operation'] == 'local':
            result.update(parent_sha256=prompt['parent_sha256'],
                          edits=[{'old': prompt['parent_source'], 'new': code}])
        else:
            result['code'] = code
        return Response(json.dumps(result))

    def make(self, name='run', calls=40, tokens=0, **options):
        root = self.base / name
        provider = OnlineProvider(root, 'deepseek-flash', calls, tokens, key='test', problem='Public fixture')
        cases = [Case(str(i), f'public {i}') for i in range(2)]

        def evaluate(code, subset, seconds, replica, *, seed=0):
            value = int(re.search(r'return (\d+)', code)[1])
            status = 'COMPILATION_ERROR' if 'FORCE_INVALID' in code else 'ACCEPTED'
            items = [SimpleNamespace(judge_result=SimpleNamespace(value=status), absolute_score=value,
                     execution_time=1.5, output_str='ok', message='', error_str='') for _ in subset]
            return summarize(items, subset, 'maximize')

        return OnlineEngine(root, provider, evaluate, cases, {
            'problem_id': 'mock', 'direction': 'maximize', 'time_limit': 2, 'calls': calls,
            'token_budget': tokens, 'failure_experience': True, 'smoke_count': 1, **options})

    def test_twelve_round_interval_and_resume_identity(self):
        engine = self.make(calls=53, meta_interval=12)
        engine.run()
        self.assertEqual([u['after_round'] for u in engine.meta['updates']], [12])
        self.assertEqual(len(self.meta_prompts[0]['round_effects']), 12)
        self.assertEqual(self.meta_prompts[0]['settings']['meta_interval'], 12)
        body = engine.provider.request_body('meta', self.meta_prompts[0])
        self.assertIn('every 12\ncompleted', body['messages'][0]['content'])
        resumed = self.make(calls=53, meta_interval=12)
        self.assertEqual(resumed.meta['active'], engine.meta['active'])
        with self.assertRaises(ValueError):
            self.make(calls=53, meta_interval=4)

    def test_minimal_start_uses_same_bank_and_is_not_v52_policy(self):
        from aad.meta_online.initial_policy import MINIMAL
        engine = self.make(calls=5, meta_interval=12, meta_initial_policy='minimal')
        engine.initialize()
        self.assertEqual(len(engine.s['records']), 4)
        self.assertEqual(engine.policy_sources, MINIMAL)
        self.assertNotEqual(engine.meta['active'], BASELINE)
        self.assertEqual(git(self.repo, 'rev-parse', 'main'), self.main_before)
        engine.preflight(engine.policy_sources)
        engine.update_population()
        with patch.dict(engine.s, engine.s.copy(), clear=True):
            tasks = engine.round_tasks()
        self.assertEqual([t['arm'] for t in tasks], ['mutation'] * 4)
        self.assertTrue(all(t['parent'] in engine.s['population'] for t in tasks))
        self.assertTrue(all('donor' not in t for t in tasks))
        body = engine.provider.request_body('meta', engine.evidence())
        self.assertIn('minimal initial harness', body['messages'][0]['content'])
        self.assertIn('every 12\ncompleted', body['messages'][0]['content'])
        engine.run()
        self.assertTrue(any(e.get('arm') == 'mutation' for e in engine.s['events']))
        resumed = self.make(calls=5, meta_interval=12, meta_initial_policy='minimal')
        self.assertEqual(resumed.policy_sources, MINIMAL)
        with self.assertRaises(ValueError):
            self.make(calls=5, meta_interval=12)

    def test_twelve_round_review_waits_for_minimal_repairs(self):
        self.fail_redesign = True
        engine = self.make(calls=120, meta_interval=12, meta_initial_policy='minimal')
        self.on_meta = lambda: (engine.root / 'STOP_AFTER_CURRENT').touch()
        engine.run()
        self.assertEqual(self.meta_count, 1)
        self.assertEqual(len(engine.meta['rounds']), 12)
        self.assertTrue(all(r['requests'] == 8 for r in engine.meta['rounds']))
        self.assertEqual(engine.meta['updates'][0]['after_round'], 12)
        self.assertEqual(engine.meta['updates'][0]['call_id'], 96)
        self.assertFalse(engine.s['parallel_batch'])

    def test_minimal_repairs_once_and_population_retains_tied_incumbent(self):
        from copy import deepcopy
        engine = self.make(calls=5, meta_interval=12, meta_initial_policy='minimal')
        self.fail_redesign = True
        engine.run()
        self.assertTrue(any(e.get('arm') == 'repair' for e in engine.s['events']))
        self.assertEqual(engine.s['parallel_round'], 1)
        self.assertIsNone(engine.s.get('parallel_batch'))
        view = engine.view()
        record = deepcopy(view['records'][engine.s['incumbent']])
        view['records'] = {str(i): deepcopy(record) for i in range(10)}
        view['incumbent'] = '9'
        selected = engine.components.population.update_population(view)
        self.assertEqual(len(selected), 8)
        self.assertIn('9', selected)

    def test_invalid_interval_and_initial_policy_rejected(self):
        for interval in (0, -1, True, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.make(meta_interval=interval)
        with self.assertRaises(ValueError):
            self.make(meta_initial_policy='unknown')

    def change_para(self, value=1):
        return {'action': 'update', 'reason': 'Use the observed local gains to change round allocation',
                'files': {'parallelism.py': f'def select_para(view):\n    return {value}\n'}}

    def test_four_rounds_keep_full_history_and_shared_accounting(self):
        engine = self.make(calls=35)
        self.assertEqual(engine.config['meta_model'], 'deepseek-flash')
        self.assertEqual(engine.config['meta_reasoning_effort'], 'max')
        engine.run()
        self.assertEqual([x['after_round'] for x in engine.meta['updates']], [4, 8])
        self.assertEqual(self.meta_count, 2)
        self.assertEqual(len(engine.meta['versions']), 1)
        self.assertEqual(engine.meta['active'], BASELINE)
        self.assertEqual(len(self.meta_prompts[0]['round_effects']), 4)
        self.assertEqual(len(self.meta_prompts[1]['round_effects']), 8)
        self.assertEqual(self.meta_prompts[1]['updates'][0]['status'], 'kept')
        self.assertEqual(len(self.meta_prompts[0]['source_blobs']), 6)
        self.assertEqual(engine.provider.requests, self.http_count)
        engine.finalize()
        report = read_json(engine.root / 'report.json')
        self.assertEqual(report['meta_requests'], 2)
        self.assertEqual(report['requests'], report['meta_requests'] + report['solve_requests'])
        self.assertEqual(report['charged_tokens'], self.http_count * 10)

    def test_adopt_next_round_no_score_trial_or_rollback_new_run_resets(self):
        self.proposals = [self.change_para()]
        engine = self.make(calls=26)
        engine.run()
        revision = engine.meta['active']
        self.assertNotEqual(revision, BASELINE)
        self.assertEqual(engine.meta['versions'][1]['activated_after_round'], 4)
        self.assertEqual([r['requests'] for r in engine.meta['rounds'][:5]], [4, 4, 4, 4, 1])
        self.assertEqual(engine.meta['rounds'][4]['revision'], revision)
        self.assertEqual(len(self.meta_prompts[1]['versions']), 2)
        old_source = self.meta_prompts[1]['versions'][0]['files']['parallelism.py']
        new_source = self.meta_prompts[1]['versions'][1]['files']['parallelism.py']
        self.assertNotEqual(old_source, new_source)
        self.assertIn('return 4', self.meta_prompts[1]['source_blobs'][old_source])
        self.assertIn('return 1', self.meta_prompts[1]['source_blobs'][new_source])
        fresh = self.make('new', calls=1)
        fresh.run()
        self.assertEqual(fresh.meta['active'], BASELINE)
        self.assertEqual(fresh.meta['rounds'][0]['requests'], 4)
        self.assertEqual(git(self.repo, 'rev-parse', 'main'), self.main_before)

    def test_score_stagnation_does_not_revert_new_policy(self):
        self.proposals = [self.change_para()]
        engine = self.make(calls=23)
        original = engine.evaluate
        def evaluate(code, *args, **kwargs):
            result = original(code, *args, **kwargs)
            if self.meta_count:
                result['mean'] = result['raw_mean'] = 0
                for row in result['cases']:
                    row['score'] = row['raw_score'] = 0
            return result
        engine.evaluate = evaluate
        engine.run()
        self.assertNotEqual(engine.meta['active'], BASELINE)
        self.assertTrue(all(r['gain'] == 0 for r in engine.meta['rounds'][4:]))
        self.assertEqual(engine.meta['updates'][1]['status'], 'kept')

    def test_completed_meta_response_is_recovered_without_new_http(self):
        self.proposals = [self.change_para()]
        engine = self.make(calls=22)
        original = engine.preflight
        with patch.object(engine, 'preflight', side_effect=OSError('power loss')):
            with self.assertRaisesRegex(OSError, 'power loss'):
                engine.run()
        self.assertEqual(self.meta_count, 1)
        self.assertIsNotNone(engine.meta['pending'])
        resumed = self.make(calls=22)
        resumed.run()
        self.assertEqual(self.meta_count, 1)  # Recovery never resends; the last request is reserved for a solver.
        self.assertEqual(resumed.meta['updates'][0]['call_id'], 16)
        self.assertEqual(len(resumed.meta['versions']), 2)
        self.assertEqual(resumed.provider.requests, self.http_count)

    def test_recover_commit_before_activation_checkpoint(self):
        self.proposals = [self.change_para()]
        # Git normalizes Python sources to LF; replay must normalize the saved proposal identically.
        self.proposals[0]['files']['parallelism.py'] = self.proposals[0]['files']['parallelism.py'].replace('\n', '\r\n')
        engine = self.make(calls=19)
        original = engine.store.create
        def interrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            raise OSError('crash after git commit')
        with patch.object(engine.store, 'create', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'crash after git commit'):
                engine.run()
        resumed = self.make(calls=19)
        resumed.run()
        self.assertEqual(self.meta_count, 1)
        self.assertEqual(len(resumed.meta['versions']), 2)

    def test_stop_marker_prevents_meta_and_private_finalization_remains_explicit(self):
        engine = self.make(calls=25)
        original = engine.finish_batch
        def finish(batch):
            original(batch)
            if len(engine.meta['rounds']) == 4:
                (engine.root / 'STOP_AFTER_CURRENT').touch()
        engine.finish_batch = finish
        engine.run()
        self.assertEqual(self.meta_count, 0)
        self.assertEqual(len(engine.meta['rounds']), 4)
        self.assertFalse(engine.s['frozen'])

    def test_invalid_update_halts_without_activation_or_free_trial(self):
        self.proposals = [self.change_para(0)]
        engine = self.make(calls=25)
        with self.assertRaisesRegex(PolicyError, 'without rollback'):
            engine.run()
        self.assertEqual(engine.meta['active'], BASELINE)
        self.assertEqual(engine.provider.requests, 17)
        self.assertEqual(len(engine.meta['rounds']), 4)
        with self.assertRaises(PolicyError):
            engine.finalize()
        resumed = self.make(calls=25)
        with self.assertRaises(PolicyError):
            resumed.run()
        self.assertEqual(self.meta_count, 1)

    def test_failed_meta_request_charged_once_no_retry(self):
        engine = self.make(calls=25)
        self.on_meta = lambda: (_ for _ in ()).throw(TimeoutError('unknown usage'))
        with self.assertRaisesRegex(PolicyError, 'reserved usage remains charged'):
            engine.run()
        self.assertEqual(engine.provider.requests, 17)
        self.assertGreater(engine.provider.charged_tokens, 160)
        with self.assertRaises(PolicyError):
            self.make(calls=25).run()
        self.assertEqual(self.meta_count, 1)

    def test_public_evidence_does_not_read_private_files_or_internal_config(self):
        engine = self.make(calls=1)
        engine.run()
        (engine.root / 'private_report.json').write_text('SECRET_PRIVATE_CANARY', encoding='utf-8')
        engine.config['private_hint'] = 'SECRET_PRIVATE_CANARY'
        evidence = json.dumps(engine.evidence())
        self.assertNotIn('SECRET_PRIVATE_CANARY', evidence)
        engine.finalize()
        with self.assertRaisesRegex(ValueError, 'closed'):
            engine.run()

    def test_tampered_worktree_rejected(self):
        engine = self.make(calls=1)
        path = engine.store.path(0) / 'aad/search_policy/components/parallelism.py'
        path.write_text('def select_para(view): return 99\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed outside'):
            self.make(calls=1)

    def test_last_request_not_spent_on_meta(self):
        engine = self.make(calls=17)
        engine.run()
        self.assertEqual(self.meta_count, 0)
        self.assertEqual(engine.meta['updates'][0]['status'], 'skipped_no_solver_budget')

    def test_repairs_finish_before_four_round_trigger_and_are_charged(self):
        self.fail_redesign = True
        engine = self.make(calls=60)
        self.on_meta = lambda: (engine.root / 'STOP_AFTER_CURRENT').touch()
        engine.run()
        self.assertEqual(self.meta_count, 1)
        self.assertEqual(len(engine.meta['rounds']), 4)
        self.assertTrue(all(r['requests'] > 4 for r in engine.meta['rounds']))
        self.assertEqual(engine.meta['updates'][0]['call_id'],
                         sum(r['requests'] for r in engine.meta['rounds']))
        self.assertFalse(engine.s['parallel_batch'])
        self.assertEqual(engine.provider.charged_tokens, self.http_count * 10)

    def test_token_reservation_can_skip_meta_without_free_request(self):
        engine = self.make(calls=40)
        original = engine.finish_batch
        def finish(batch):
            original(batch)
            if len(engine.meta['rounds']) == 4:
                (engine.root / 'STOP_AFTER_CURRENT').touch()
        engine.finish_batch = finish
        engine.run()
        (engine.root / 'STOP_AFTER_CURRENT').unlink()
        engine.provider.max_tokens_total = engine.provider.charged_tokens + 1
        self.assertTrue(engine.before_round())
        self.assertEqual(self.meta_count, 0)
        self.assertEqual(engine.meta['updates'][0]['status'], 'skipped_token_reservation')
        self.assertIsNone(engine.meta['pending'])
        self.assertEqual(engine.provider.requests, 16)

    def test_resume_partially_evaluated_round_keeps_revision_and_no_resend(self):
        engine = self.make(calls=2)
        engine.initialize()
        engine.evaluate = lambda *a, **k: (_ for _ in ()).throw(OSError('evaluation interrupted'))
        with self.assertRaises(OSError):
            engine.run()
        requests = self.http_count
        resumed = self.make(calls=2)
        resumed.run()
        self.assertEqual(self.http_count, requests)
        self.assertEqual(resumed.meta['rounds'][0]['revision'], BASELINE)
        self.assertEqual(resumed.meta['rounds'][0]['requests'], 4)

    def test_update_cannot_write_paths_outside_six_components(self):
        self.proposals = [{'action': 'update', 'reason': 'bad path',
                           'files': {'../engine.py': 'print(1)'}}]
        engine = self.make(calls=20)
        with self.assertRaisesRegex(PolicyError, 'schema'):
            engine.run()
        self.assertEqual(engine.meta['active'], BASELINE)
        self.assertFalse((engine.store.path(0) / 'aad/search_policy/engine.py').read_text().startswith('print'))

    def test_freeze_initials_without_a_model_ledger(self):
        engine = self.make(calls=1)
        engine.initialize()
        engine.provider = None
        (engine.root / 'llm' / 'ledger.json').unlink()
        engine.finalize()
        report = read_json(engine.root / 'report.json')
        self.assertEqual(report['meta_requests'], 0)
        self.assertEqual(report['solve_requests'], 0)


if __name__ == '__main__':
    unittest.main()
