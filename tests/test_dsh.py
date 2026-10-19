"""Official SDK boundary mocked; actual local MCP, policies and engine exercised."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aad.io import digest, read_json, save_json
from aad.provider import BudgetExhausted
from aad.parallel_engine import V51Provider
from aad.meta_online.accounting import dsh_totals
from aad.meta_online.dsh import DshProvider, profile_patch, usage_from_events, TOOLS_SCRIPT
from aad.meta_online.dsh_tools import PolicyTools, FILES, prepare_workspace, sources_at
import test_online_meta as fixtures


def evidence():
    sources = fixtures.WorkerTests().sources()
    return {'active_revision': 'baseline', 'versions': [
        {'revision': 'baseline', 'files': {name: digest(s) for name, s in sources.items()}}],
        'source_blobs': {digest(s): s for s in sources.values()}, 'round_effects': [{'gain': 4}],
        'public_records': {}, 'problem': 'public fixture'}


class FakeHarness:
    launches = []
    fail = False
    finish = 'completed'
    action = 'update'

    def __init__(self, **kwargs):
        self.options = kwargs
        self.launches.append(kwargs)
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self): pass
    def run(self, prompt, *, session_id, on_notification):
        server = PolicyTools(Path(self.options['cwd']).parent)
        assert 'evidence/round_effects.json' in server.call('list_files', {'prefix': 'evidence/'})
        server.call('read_file', {'path': 'evidence/round_effects.json'})
        source = server.call('read_file', {'path': 'draft/parallelism.py'})['text'] + '\n'
        server.call('write_component', {'name': 'parallelism.py', 'source': source + '\n# DSH edit\n'})
        server.call('check_components', {})
        for seq in (0, 1):
            on_notification(SimpleNamespace(method='session.event', payload={
                'sessionId': session_id, 'event': {'seq': seq, 'type': 'assistant/message',
                    'data': {'usage': {'inputTokens': 100, 'outputTokens': 10, 'cacheReadTokens': 20}}}}))
            if self.fail:
                raise RuntimeError('simulated transport failure')
        return SimpleNamespace(final_response=json.dumps({'action': self.action, 'reason': 'public evidence'}),
                               finish_reason=self.finish)


class DshTests(unittest.TestCase):
    def setUp(self):
        FakeHarness.launches = []
        FakeHarness.fail, FakeHarness.finish, FakeHarness.action = False, 'completed', 'update'

    def provider(self, root, tokens=1, cls=DshProvider):
        return cls(root, 'solver-model', 20, tokens, key='fixture-secret', problem='public fixture')

    def test_real_sdk_boundary_tools_edit_and_tokens_exempt(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.meta_online.dsh.sdk_class', return_value=FakeHarness):
            provider = self.provider(root)
            job = {'kind': 'meta', 'prompt': evidence(), 'implementation_call': 0}
            provider.request_many([job])
            row = provider.ledger[0]
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['dsh_usage']['reported_tokens'], 260)
            self.assertEqual(provider.charged_tokens, 0)  # Works even though solver budget is only 1 token.
            self.assertEqual(dsh_totals(provider.ledger)['budget_tokens'], 0)
            payload = read_json(Path(root)/'llm/00000.response.json')
            proposal = json.loads(payload['choices'][0]['message']['content'])
            self.assertEqual(list(proposal['files']), ['parallelism.py'])
            self.assertIn('# DSH edit', proposal['files']['parallelism.py'])
            options = FakeHarness.launches[0]
            self.assertEqual((options['model'], options['reasoning_effort']), ('deepseek-flash', 'max'))
            self.assertEqual(options['profile'], 'sdk-minimal')
            # The native provider appends /chat/completions itself.
            self.assertEqual(options['base_url'], 'https://api.deepseek.com')
            saved = read_json(Path(root)/'llm/00000.request.json')
            self.assertNotIn('fixture-secret', json.dumps(saved))
            self.assertNotIn('fixture-secret', Path(options['patches'][0]).read_text())
            recovered = self.provider(root)
            recovered.request_many([job])
            self.assertEqual(len(FakeHarness.launches), 1)
            with self.assertRaises(BudgetExhausted):
                recovered.request('program', {})  # Inner budget remains enforced.

    def test_interruption_keeps_partial_usage_and_is_not_reissued(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.meta_online.dsh.sdk_class', return_value=FakeHarness):
            FakeHarness.fail = True
            provider = self.provider(root)
            job = {'kind': 'meta', 'prompt': evidence(), 'implementation_call': 0}
            with self.assertRaisesRegex(RuntimeError, 'no API fallback'):
                provider.request_many([job])
            row = provider.ledger[0]
            self.assertEqual(row['status'], 'dsh_error_usage_unknown')
            self.assertFalse(row['dsh_usage']['complete'])
            self.assertEqual(row['dsh_usage']['reported_tokens'], 130)
            self.provider(root).request_many([job])
            self.assertEqual(len(FakeHarness.launches), 1)

    def test_keep_discards_draft_and_truncation_cannot_be_adopted(self):
        for finish in ('completed', 'max-tokens'):
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as root, \
                 patch('aad.meta_online.dsh.sdk_class', return_value=FakeHarness):
                FakeHarness.action, FakeHarness.finish = 'keep', finish
                provider = self.provider(root)
                result = provider.request('meta', evidence())
                if finish == 'completed':
                    self.assertEqual(json.loads(result['content'])['files'], {})
                else:
                    self.assertEqual(provider.ledger[0]['status'], 'invalid_response')
                    self.assertFalse((Path(root)/'llm/00000.response.json').exists())

    def test_inner_request_bodies_unchanged(self):
        for cls in (DshProvider,):
            with tempfile.TemporaryDirectory() as root:
                provider = self.provider(Path(root)/'dsh', cls=cls)
                baseline = V51Provider(Path(root)/'baseline', 'solver-model', 20, 1,
                                       key='fixture-secret', problem='public fixture')
                for kind in ('program', 'edit', 'plan'):
                    self.assertEqual(provider.request_body(kind, {}), baseline.request_body(kind, {}))

    def test_no_shell_web_or_arbitrary_files_and_history_is_immutable(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            task = evidence()
            original = {n: task['source_blobs'][sha] for n, sha in task['versions'][0]['files'].items()}
            save_json(root/'evidence.json', task)
            save_json(root/'draft.json', original)
            prepare_workspace(root, task, original)
            server = PolicyTools(root)
            for name, args in [('read_component', {'name': '../private_report.json'}),
                               ('write_component', {'name': 'provider.py', 'source': 'bad'}),
                               ('shell', {'command': 'curl example.com'}),
                               ('read_component', {'name': 'parents.py', 'path': '/etc/passwd'})]:
                with self.assertRaises(ValueError):
                    server.call(name, args)
            server.call('write_component', {'name': 'parents.py', 'source': 'bad'})
            self.assertEqual(server.call('read_component', {'name': 'parents.py', 'revision': 'baseline'})['source'],
                             original['parents.py'])
            with self.assertRaises(ValueError):
                server.call('check_components', {})
            tree = profile_patch(root, 'No network research')
            disabled = {r['id'] for r in tree if r.get('disabled')}
            self.assertTrue({'persistent-bash', 'persistent-pwsh', 'tool-working-directory'} <= disabled)
            clients = tree[-1]['insert']
            self.assertEqual(len(clients), 1)
            self.assertEqual(clients[0]['config']['transport'], 'stdio')

    def test_mcp_stdio_process_negotiates_and_exposes_local_workspace_tools(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            save_json(root/'evidence.json', evidence())
            prepare_workspace(root, evidence(), sources_at(evidence(), 'baseline'))
            requests = [dict(jsonrpc='2.0', id=1, method='initialize', params={'protocolVersion': '2024-11-05'}),
                        dict(jsonrpc='2.0', method='notifications/initialized'),
                        dict(jsonrpc='2.0', id=2, method='tools/list')]
            result = subprocess.run([sys.executable, '-I', '-B', str(TOOLS_SCRIPT), str(root)],
                                    input='\n'.join(map(json.dumps, requests))+'\n',
                                    text=True, encoding='utf-8', capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            responses = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(responses[0]['result']['protocolVersion'], '2024-11-05')
            self.assertEqual({t['name'] for t in responses[1]['result']['tools']},
                             {'list_files', 'read_file', 'search_files', 'read_evidence',
                              'read_component', 'write_component', 'check_components'})

    def test_folder_exploration_pagination_and_no_private_or_runtime_files(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            task = evidence()
            original = sources_at(task, 'baseline')
            revised = {**original, 'parallelism.py': original['parallelism.py'] + '\n# later version\n'}
            task['source_blobs'].update({digest(s): s for s in revised.values()})
            task['versions'].append({'revision': 'newer', 'files': {n: digest(s) for n, s in revised.items()}})
            task['round_effects'] = [{'gain': i} for i in range(210)]
            save_json(root/'evidence.json', task)
            prepare_workspace(root, task, original)
            (root/'workspace/private_report.json').write_text('PRIVATE')
            (root/'credentials.txt').write_text('SECRET')
            server = PolicyTools(root)
            paths = server.call('list_files', {})
            self.assertNotIn('private_report.json', paths)
            self.assertIn('history/00001/parallelism.py', paths)
            self.assertIn('later version', server.call('read_file', {'path': 'history/00001/parallelism.py'})['text'])
            first = server.call('read_file', {'path': 'evidence/round_effects.json', 'max_lines': 3})
            second = server.call('read_file', {'path': first['path'], 'start_line': first['next_line'], 'max_lines': 2})
            self.assertEqual(second['start_line'], 4)
            search = server.call('search_files', {'text': '"gain"', 'prefix': 'evidence/round_effects'})
            self.assertEqual(search['next_offset'], 100)
            last = server.call('search_files', {'text': '"gain"', 'prefix': 'evidence/round_effects', 'offset': 200})
            self.assertEqual(len(last['matches']), 10)
            self.assertIsNone(last['next_offset'])
            for path in ('../credentials.txt', 'private_report.json', '../evidence.json', str(root/'credentials.txt')):
                with self.assertRaises(ValueError):
                    server.call('read_file', {'path': path})
            for args in ({'path': 'README.txt', 'start_line': 0}, {'path': 'README.txt', 'max_lines': True}):
                with self.assertRaises(ValueError):
                    server.call('read_file', args)
            server.call('write_component', {'name': 'parallelism.py', 'source': revised['parallelism.py']})
            self.assertEqual((root/'workspace/draft/parallelism.py').read_text(), revised['parallelism.py'])
            self.assertEqual((root/'workspace/history/00000/parallelism.py').read_text(), original['parallelism.py'])

    def test_sdk_value_error_is_infrastructure_failure_not_retryable_proposal(self):
        with tempfile.TemporaryDirectory() as root, patch('aad.meta_online.dsh.sdk_class') as sdk:
            sdk.return_value.side_effect = ValueError('invalid SDK configuration')
            provider = self.provider(root)
            with self.assertRaisesRegex(RuntimeError, 'no API fallback'):
                provider.request('meta', evidence())
            self.assertEqual(provider.ledger[0]['status'], 'dsh_error_usage_unknown')

    def test_missing_or_failed_usage_is_not_reported_as_complete_zero(self):
        result = usage_from_events([{'type': 'assistant/attempt', 'data': {}}], True)
        self.assertEqual(result['unknown_usage_attempts'], 1)
        self.assertFalse(result['complete'])
        result = usage_from_events([{'type': 'assistant/message', 'data': {'usage': {
            'inputTokens': 10, 'outputTokens': 5, 'reasoningTokens': 3, 'totalTokens': 17}}}], True)
        self.assertEqual(result['reported_tokens'], 17)


class DshEngineTests(unittest.TestCase):
    setUp = fixtures.OnlineTests.setUp
    respond = fixtures.OnlineTests.respond

    def test_minimal_twelve_round_sdk_run_resume_and_freeze(self):
        from aad.meta_online.initial_policy import MINIMAL
        FakeHarness.action, FakeHarness.fail, FakeHarness.finish = 'keep', False, 'completed'
        with patch.object(fixtures, 'OnlineProvider', DshProvider), \
             patch('aad.meta_online.dsh.sdk_class', return_value=FakeHarness):
            engine = fixtures.OnlineTests.make(self, calls=53, meta_interval=12, meta_initial_policy='minimal')
            engine.run()
            self.assertEqual([u['after_round'] for u in engine.meta['updates']], [12])
            call = engine.meta['updates'][0]['call_id']
            root = engine.root / 'meta/dsh' / f'{call:05d}'
            evidence = read_json(root / 'evidence.json')
            self.assertEqual(len(evidence['versions']), 1)
            self.assertEqual(sources_at(evidence, evidence['active_revision']), MINIMAL)
            rule = read_json(root / 'policy.patch.json')[-2]['config']['personaPrefix']
            self.assertIn('every 12\ncompleted', rule)
            self.assertIn('minimal initial harness', rule)
            self.assertNotIn('Every fresh run begins with the preserved v5.2 policy.', rule)
            resumed = fixtures.OnlineTests.make(self, calls=53, meta_interval=12, meta_initial_policy='minimal')
            self.assertEqual(resumed.s['incumbent'], engine.s['incumbent'])
            resumed.finalize()
            report = read_json(engine.root / 'report.json')
            self.assertEqual(report['meta_interval'], 12)
            self.assertEqual(report['initial_policy'], 'minimal')

    def test_large_dsh_history_is_preserved_and_reaches_sdk(self):
        FakeHarness.action, FakeHarness.fail, FakeHarness.finish = 'keep', False, 'completed'
        with patch.object(fixtures, 'OnlineProvider', DshProvider), \
             patch('aad.meta_online.dsh.sdk_class', return_value=FakeHarness):
            engine = fixtures.OnlineTests.make(self, calls=22)
            original = engine.evidence
            def large_evidence():
                data = original()
                data['round_effects'].append({'large_history': 'x' * 1_000_001})
                return data
            with patch.object(engine, 'evidence', side_effect=large_evidence):
                engine.run()
        self.assertEqual(engine.meta['updates'][0]['status'], 'kept')
        call = engine.meta['updates'][0]['call_id']
        saved = read_json(engine.root / 'meta/dsh' / f'{call:05d}' / 'evidence.json')
        self.assertEqual(saved['round_effects'][-1]['large_history'], 'x' * 1_000_001)

    def test_direct_api_still_rejects_oversized_history(self):
        engine = fixtures.OnlineTests.make(self, calls=22)
        with patch.object(engine, 'evidence', return_value={'history': 'x' * 1_000_001}):
            with self.assertRaisesRegex(ValueError, 'Complete meta history exceeds'):
                engine.run()
        self.assertEqual(self.meta_count, 0)

    def test_four_rounds_trigger_sdk_and_final_report_separates_budget(self):
        FakeHarness.action, FakeHarness.fail, FakeHarness.finish = 'keep', False, 'completed'
        with patch.object(fixtures, 'OnlineProvider', DshProvider), \
             patch('aad.meta_online.dsh.sdk_class', return_value=FakeHarness):
            engine = fixtures.OnlineTests.make(self, calls=22)
            engine.run()
            engine.finalize()
        report = read_json(engine.root/'report.json')
        self.assertEqual(engine.meta['updates'][0]['after_round'], 4)
        self.assertEqual(engine.meta['updates'][0]['charged_tokens'], 0)
        self.assertEqual(report['charged_tokens'], self.http_count * 10)
        self.assertEqual(report['dsh_usage']['sessions'], 1)
        self.assertEqual(report['dsh_usage']['reported_tokens'], 260)
        self.assertEqual(report['token_budget_scope'], 'solver_only_dsh_exempt')


if __name__ == '__main__':
    unittest.main()
