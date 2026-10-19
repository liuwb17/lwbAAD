"""Resilient entry point: python scripts/meta_recovery.py run ...

An operational wrapper for pinned runs. It never edits an existing request,
response, ledger row, adopted policy, protocol signature, or solver record.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aad.io import read_json, save_json
from aad.meta_online.policy import PolicyError
from aad.meta_online.accounting import budget_tokens


def retry_state(state, ledger, response=None, limit=3):
    result = deepcopy(state)
    meta = result['meta']
    pending = meta.get('pending')
    if result.get('frozen') or result.get('development_closed') or not meta.get('error') or not pending:
        raise ValueError('Only a halted, unadopted meta proposal can be retried')
    call = pending['job']['implementation_call']
    row = ledger[call]
    if row['kind'] != 'meta' or row['status'] not in ('ok', 'invalid_response'):
        raise ValueError('Unknown usage, transport and authentication failures require separate review')
    if any(v.get('call_id') == call for v in meta['versions']):
        raise ValueError('Proposal already adopted; cannot retry')
    history = meta.setdefault('recovery_history', [])
    exhausted = sum(h['after_round'] == pending['after_round'] for h in history) >= limit
    history.append({'after_round': pending['after_round'], 'failed_call_id': call,
                    'error': meta['error'], 'pending': deepcopy(pending),
                    'charged_tokens': budget_tokens(row),
                    'prepared_utc': datetime.now(timezone.utc).isoformat()})
    if exhausted:
        # This rejected proposal was never adopted. Retain the active strategy;
        # this is not a score-based rollback or a fabricated meta keep decision.
        meta.setdefault('updates', []).append({
            'after_round': pending['after_round'], 'call_id': call,
            'status': 'rejected_retry_limit', 'action': 'retain_active_after_invalid_proposals',
            'revision': meta['active'], 'error': meta['error'],
            'charged_tokens': history[-1]['charged_tokens']})
        meta['handled_round'] = pending['after_round']
        meta['pending'] = None
        del meta['error']
        return result
    feedback = {'failed_call_id': call, 'validation_error': meta['error'],
                'finish_reason': row.get('finish_reason'),
                'instruction': 'The previous proposal was NOT adopted. Retry this same meta decision. '
                'Use all supplied public history and decide keep/update yourself. Return concise, complete JSON. '
                'Avoid lengthy reasoning and large rewrites so the final JSON fits the output limit. '
                'All helper function and variable names must be public (no leading underscore). '
                'No annotations or decorators. Keep all interface and sandbox constraints. '
                'Do not change model settings, budget, scoring or access private evidence.'}
    if isinstance(response, dict):
        choices = response.get('choices')
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get('message') if isinstance(choice, dict) else None
        content = message.get('content') if isinstance(message, dict) else None
        if isinstance(content, str) and len(content) <= 32000:
            feedback['rejected_output_untrusted'] = content
    pending['job']['prompt']['meta_recovery'] = feedback
    pending['job']['implementation_call'] = len(ledger)
    del meta['error']
    return result


def run_with_recovery(args):
    from scripts.rsi_search import main, run_lock
    root = Path(args[args.index('--run-dir') + 1])
    while True:
        try:
            return main(args)
        except PolicyError:
            if (root / 'STOP_AFTER_CURRENT').exists():
                raise
            with run_lock(root):
                state = read_json(root / 'state.json')
                pending = state.get('meta', {}).get('pending')
                if not pending:
                    raise
                call = pending['job']['implementation_call']
                response_path = root / 'llm' / f'{call:05d}.response.json'
                response = read_json(response_path) if response_path.exists() else None
                updated = retry_state(state, read_json(root / 'llm/ledger.json'), response)
                audit = root / 'meta' / 'recoveries' / f'{call:05d}.state-before.json'
                if not audit.exists():
                    save_json(audit, state)
                save_json(root / 'state.json', updated)
                next_pending = updated['meta']['pending']
                print({'meta_recovery': 'retry' if next_pending else 'retain_active', 'failed_call': call,
                       'next_call': next_pending['job']['implementation_call'] if next_pending else None}, flush=True)
            if '--resume' not in args:
                args = [*args, '--resume']


if __name__ == '__main__':
    run_with_recovery(sys.argv[1:])
