"""Parent-specific memories, written only after the attempt's result is known."""
from copy import deepcopy
import json

from .candidate_policy import valid_for_archive


def augment_task(engine, pending, task):
    if not engine.config['failure_experience'] or not pending.get('parent'):
        return task
    parent = engine.s['records'][pending['parent']]
    return {**task,
            'parent_failure_experience': deepcopy(parent.get('failure_experience', [])[-8:]),
            'instruction': task['instruction'] +
                ' In hypothesis, use one concise sentence to describe the concrete change relative '
                'to the main parent and why it should improve the result. The supplied '
                'parent_failure_experience lists previous unsuccessful attempts on this exact parent; '
                'consider a different approach and avoid repeating those failed changes.'}


def note_response(engine, pending, content):
    if not engine.config['failure_experience'] or not pending.get('parent'):
        return
    # Preserve this attempt's idea even if code parsing fails or source is a duplicate.
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return
    hypothesis = payload.get('hypothesis') if isinstance(payload, dict) else None
    if isinstance(hypothesis, str) and hypothesis.strip():
        pending['attempt_hypothesis'] = ' '.join(hypothesis.split())[:3000]


def record_attempt(engine, pending):
    if not engine.config['failure_experience'] or not pending.get('parent'):
        return
    hypothesis = pending.get('attempt_hypothesis')
    # A transport failure with no usable proposal is not algorithmic evidence.
    if not hypothesis or pending.get('status') == 'budget_exhausted':
        return
    parent = engine.s['records'][pending['parent']]
    child = engine.s['records'].get(pending.get('candidate'), {})
    entry = {'call_id': pending['implementation_call'], 'operation': pending['arm'],
             'hypothesis': hypothesis, 'candidate': pending.get('candidate')}
    if pending.get('status') == 'failed':
        entry.update(outcome='invalid_proposal', error=pending.get('error', '')[:600])
    elif not valid_for_archive(child):
        result = next((r for r in child.get('evaluations', []) if not r['valid']),
                      child.get('full', child.get('smoke', {})))
        statuses = sorted({r['status'] for r in result.get('cases', []) if r['status'] != 'AC'})
        entry.update(outcome='invalid_candidate', evaluation_statuses=statuses)
    elif not valid_for_archive(parent) or child['full']['mean'] > parent['full']['mean']:
        # A valid repair of an invalid parent, or any strict mean improvement, succeeded.
        # The global incumbent and promotion margin do not affect this comparison.
        return
    else:
        entry.update(outcome='no_improvement',
                     parent_mean=engine.sign * parent['full']['mean'],
                     child_mean=engine.sign * child['full']['mean'],
                     direction=engine.config['direction'])
    history = parent.setdefault('failure_experience', [])
    if not any(e['call_id'] == entry['call_id'] for e in history):
        history.append(entry)
        del history[:-8]
