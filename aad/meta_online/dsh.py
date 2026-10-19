"""Official DSH SDK agent for meta updates; inner calls remain v5.2 calls."""
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
import time

from ..io import digest, read_json, save_json
from ..provider import BudgetExhausted
from .accounting import budget_tokens
from .provider import OnlineProvider
from .dsh_tools import sources_at, prepare_workspace, read_draft

SDK_VERSION = '0.1.5rc1'
DSH_PROTOCOL = 'dsh-sdk-public-workspace-2'
TOOLS_SCRIPT = Path(__file__).with_name('dsh_tools.py')


def parse_final_decision(text):
    """Accept one unambiguous decision, including explanatory text/fences.

    Never infer an action from prose or accept truncated/ambiguous JSON.
    The caller must independently require a completed SDK session.
    """
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate decision key')
            result[key] = value
        return result
    decoder = json.JSONDecoder(object_pairs_hook=unique)
    decisions = []
    offset = 0
    while offset < len(text):
        offset = text.find('{', offset)
        if offset < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            offset += 1
            continue
        if isinstance(obj, dict) and set(obj) == {'action', 'reason', 'files'}:
            from .dsh_tools import FILES
            manifest = obj['files']
            if (not isinstance(manifest, list) or any(not isinstance(name, str) for name in manifest)
                    or len(set(manifest)) != len(manifest) or not set(manifest) <= set(FILES)):
                raise ValueError('Invalid filename manifest')
            obj = {key: obj[key] for key in ('action', 'reason')}
        if (isinstance(obj, dict) and set(obj) == {'action', 'reason'}
                and obj['action'] in ('keep', 'update')
                and isinstance(obj['reason'], str) and 1 <= len(obj['reason']) <= 10000):
            decisions.append((obj, end))
        offset = end
    if len(decisions) != 1:
        raise ValueError('Expected one unambiguous DSH decision')
    proposal, end = decisions[0]
    if text[end:].strip() not in ('', '```'):
        raise ValueError('Unexpected content after DSH decision')
    return proposal


def sdk_class():
    try:
        from deepseek_harness import DeepSeekHarness
        versions = [importlib.metadata.version(name) for name in
                    ('deepseek-harness-sdk', 'deepseek-harness-runtime-bin')]
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError('Install the pinned dsh-meta extra before starting a DSH run') from error
    if versions != [SDK_VERSION, SDK_VERSION]:
        raise RuntimeError('DSH SDK/runtime version mismatch; use the pinned dsh-meta extra')
    return DeepSeekHarness


def profile_patch(root, rule):
    # sdk-minimal is a standalone tree, not an overlay on the full web agent.
    # No filesystem/shell tool can bypass our exact-name MCP operations.
    disabled = ('persistent-bash', 'persistent-pwsh', 'terminal-bash', 'terminal-pwsh',
                'pty', 'tool-working-directory', 'session-log-deepseek',
                'plugin-package-inventory-deepseek')
    return [*({'id': name, 'disabled': True} for name in disabled),
            {'id': 'system-prompt', 'config': {'includeHarnessIdentity': False,
             'includeRuntimeContext': False, 'personaPrefix': rule}},
            {'insert': [{'id': 'rsi-policy-mcp', 'name': '@deepseek-ai/dsh-mcp-client',
              'config': {'serverName': 'rsi_policy', 'transport': 'stdio',
                         'command': sys.executable, 'args': ['-I', '-B', str(TOOLS_SCRIPT), str(root)],
                         'cwd': str(root), 'failOnStartupError': True,
                         'reconnect': {'enabled': False}}}]}]


def usage_from_events(events, completed):
    attempts, unknown, reported, fields = 0, 0, 0, {}
    for event in events:
        if event.get('type') not in ('assistant/message', 'assistant/attempt'):
            continue
        attempts += 1
        usage = event.get('data', {}).get('usage')
        if not isinstance(usage, dict) or not usage:
            unknown += 1
            continue
        for key, value in usage.items():
            if type(value) in (int, float) and value >= 0:
                fields[key] = fields.get(key, 0) + value
        if not all(type(usage.get(k)) in (int, float) for k in ('inputTokens', 'outputTokens')):
            unknown += 1
        if type(usage.get('totalTokens')) in (int, float) and usage['totalTokens'] >= 0:
            reported += usage['totalTokens']
        else:
            reported += sum(usage.get(k, 0) for k in
                            ('inputTokens', 'outputTokens', 'cacheReadTokens', 'cacheWriteTokens')
                            if type(usage.get(k, 0)) in (int, float) and usage.get(k, 0) >= 0)
    # DSH TokenUsage splits uncached input, cache reads and cache writes.
    return {'model_attempts': attempts, 'unknown_usage_attempts': unknown,
            'fields': fields, 'reported_tokens': reported,
            'complete': completed and attempts > 0 and unknown == 0}


class DshProvider(OnlineProvider):
    """One ledger row per meta session, with all underlying DSH attempts audited."""
    meta_backend = 'dsh'

    @staticmethod
    def check_dependencies():
        sdk_class()

    @property
    def charged_tokens(self):
        return sum(budget_tokens(row) for row in self.ledger)

    def request_many(self, jobs):
        if not any(job['kind'] == 'meta' for job in jobs):
            return super().request_many(jobs)
        if len(jobs) != 1 or jobs[0]['kind'] != 'meta':
            raise ValueError('DSH meta sessions run at a round boundary, separate from solver batches')
        job = jobs[0]
        call = job['implementation_call']
        if call < self.requests:
            if self.ledger[call].get('backend') != 'dsh':
                raise ValueError('Cannot recover a direct-API meta request as a DSH session')
            return  # Successful response or unknown interruption is never silently reissued.
        if call != self.requests:
            raise ValueError('Noncontiguous DSH call ID')
        self.request('meta', job['prompt'])

    def request(self, kind, task):
        if kind != 'meta':
            return super().request(kind, task)
        if self.requests >= self.max_calls:
            raise BudgetExhausted('Meta session request boundary exhausted')
        harness_cls = sdk_class()  # Missing installation is not a paid attempt.
        call = self.requests
        root = (self.root / 'meta' / 'dsh' / f'{call:05d}').resolve()
        root.mkdir(parents=True, exist_ok=False)
        parent = task['active_revision']
        original = sources_at(task, parent)
        evidence = {**task, 'problem': self.problem}
        save_json(root / 'evidence.json', evidence)
        save_json(root / 'draft.json', original)
        prepare_workspace(root, evidence, original)
        rule = self.request_body('meta', evidence)['messages'][0]['content']
        rule = rule[:rule.index('Return ONLY JSON:')].replace(
            'There are no browser, search, filesystem or shell tools.',
            'Use only the supplied local policy MCP tools. There are no browser, search or shell tools.')
        rule += ('\nYour public workspace contains evidence/, history/ and draft/; README.txt maps the folders. '
                 'Explore files and analyze the evidence yourself using list_files, read_file and search_files. '
                 'Choose what to inspect and in what order; history contains every saved policy. '
                 'Use write_component to edit draft/*.py, and check_components to check them. '
                 'All edits are staged, never changes to the baseline or an active policy. '
                 'Finish with ONLY JSON {"action":"keep"|"update","reason":"evidence and rationale"}. '
                 'Do not repeat source in the final response. For update the controller extracts your '
                 'draft changes and checks interfaces before adopting; for keep the draft is discarded. '
                 'DSH tokens are recorded separately and do not consume the solver token budget.')
        if getattr(self, 'shared_total_budget', False):
            rule = rule.replace('DSH tokens are recorded separately and do not consume the solver token budget.', 'All inner solver and meta request tokens share the same total budget cap.')
        patch_path = root / 'policy.patch.json'  # JSON is valid YAML; no interpolation of generated text.
        save_json(patch_path, profile_patch(root, rule))
        request_record = {'backend': 'dsh', 'protocol': DSH_PROTOCOL, 'sdk_version': SDK_VERSION,
                          'model': self.meta_model, 'reasoning_effort': self.meta_reasoning_effort,
                          'parent_revision': parent, 'evidence_sha256': digest(json.dumps(evidence, sort_keys=True)),
                          'profile_sha256': digest(patch_path.read_text(encoding='utf-8')),
                          'session_id': f'rsi-meta-{call:05d}', 'budget_tokens': 0}
        save_json(self.folder / f'{call:05d}.request.json', request_record)
        row = {'id': call, 'kind': 'meta', 'backend': 'dsh', 'status': 'started',
               'reserved_tokens': 0, 'budget_tokens': 0, 'requested_model': self.meta_model,
               'requested_reasoning_effort': self.meta_reasoning_effort, 'sdk_version': SDK_VERSION}
        self.ledger.append(row)
        save_json(self.path, self.ledger)
        started, events, completed = time.monotonic(), [], False

        def observe(notification):
            if notification.method != 'session.event':
                return
            event = notification.payload.get('event')
            if not isinstance(event, dict):
                return
            # Event streams never contain credentials supplied in the child environment.
            # Persist incrementally: failed or interrupted sessions retain partial usage.
            with (root / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                stream.flush()
            events.append(event)
            if event.get('type') in ('assistant/message', 'assistant/attempt'):
                row['dsh_usage'] = usage_from_events(events, False)
                save_json(self.path, self.ledger)

        try:
            with harness_cls(provider='deepseek-official', model=self.meta_model,
                             reasoning_effort=self.meta_reasoning_effort, max_tokens=65536,
                             cwd=str(root / 'workspace'), runtime_cwd=str(root), dsh_home=str(root / 'home'),
                             profile='sdk-minimal', patches=(str(patch_path),), api_key=self.key,
                             base_url=getattr(self, 'meta_base_url', 'https://api.deepseek.com'),
                             initialize_timeout_seconds=60, request_timeout_seconds=1800) as harness:
                # Session.run may wait on notifications beyond an individual RPC timeout.
                deadline = threading.Timer(1800, harness.close)
                deadline.daemon = True
                deadline.start()
                try:
                    result = harness.run('Explore your workspace folders and analyze the public evidence and '
                                         'policy history yourself. Decide whether to keep or improve the policy now.',
                                         session_id=request_record['session_id'], on_notification=observe)
                finally:
                    deadline.cancel()
            save_json(root / 'result.json', {'final_response': result.final_response,
                                           'finish_reason': result.finish_reason})
            completed = result.finish_reason == 'completed'
            row['finish_reason'] = result.finish_reason
            if not completed:
                row['status'] = 'invalid_response' if result.finish_reason == 'max-tokens' else 'dsh_error_usage_unknown'
                return {'call_id': call, 'content': '', 'usage': {}}
            proposal = parse_final_decision(result.final_response)
            if (not isinstance(proposal, dict) or set(proposal) != {'action', 'reason'}
                    or proposal['action'] not in ('keep', 'update') or not isinstance(proposal['reason'], str)):
                raise ValueError('DSH must return action and reason')
            draft = read_draft(root)
            if set(draft) != set(original):
                raise ValueError('DSH draft component set changed')
            changes = {name: source for name, source in draft.items() if source != original[name]}
            proposal['files'] = changes if proposal['action'] == 'update' else {}
            content = json.dumps(proposal, ensure_ascii=False)
            payload = {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
                       'backend': 'dsh', 'session_id': request_record['session_id']}
            save_json(self.folder / f'{call:05d}.response.json', payload)
            row.update(status='ok', finish_reason='stop', content_sha256=digest(content))
            return {'call_id': call, 'content': content, 'usage': {}}
        except (ValueError, TypeError, KeyError) as error:
            if not completed:
                row.update(status='dsh_error_usage_unknown', error_type=type(error).__name__)
                raise RuntimeError('DSH session failed; inspect its local audit, no API fallback or automatic resend') from None
            row.update(status='invalid_response', error_type=type(error).__name__)
            return {'call_id': call, 'content': '', 'usage': {}}
        except Exception as error:
            row.update(status='dsh_error_usage_unknown', error_type=type(error).__name__)
            raise RuntimeError('DSH session failed; inspect its local audit, no API fallback or automatic resend') from None
        finally:
            row['seconds'] = time.monotonic() - started
            row['dsh_usage'] = usage_from_events(events, completed)
            save_json(self.path, self.ledger)
