"""5. Context projection and derived memory; authoritative evidence is external."""
from copy import deepcopy

from ...candidate_policy import valid_for_archive


def build_context(view, item):
    parent = item['parent']
    prompt = {'operation': item['arm'], 'instruction': item['instruction'],
              'parent_source': view['sources'][parent], 'parent_sha256': parent,
              'feedback': view['feedback'][parent],
              'recent_experiments': [{k: e.get(k) for k in ('arm', 'status', 'parent_gain', 'error')}
                                     for e in view['events'][-8:]]}
    if item.get('donor'):
        prompt['donor_source'] = view['sources'][item['donor']]
    if view['config']['failure_experience']:
        prompt['parent_failure_experience'] = deepcopy(view['records'][parent].get('failure_experience', [])[-8:])
        prompt['instruction'] += (
            ' In hypothesis, use one concise sentence to describe the concrete change relative '
            'to the main parent and why it should improve the result. The supplied '
            'parent_failure_experience lists previous unsuccessful attempts on this exact parent; '
            'consider a different approach and avoid repeating those failed changes.')
    return prompt


def update_memory(view, attempts):
    """Commit failures in slot order after the batch; replay is idempotent."""
    updates = {}
    if not view['config']['failure_experience']:
        return updates
    for attempt in attempts:
        hypothesis = attempt.get('attempt_hypothesis')
        if not hypothesis or not attempt.get('parent') or attempt.get('status') == 'budget_exhausted':
            continue
        parent = view['records'][attempt['parent']]
        child = view['records'].get(attempt.get('candidate'), {})
        entry = {'call_id': attempt['implementation_call'], 'operation': attempt['arm'],
                 'hypothesis': hypothesis, 'candidate': attempt.get('candidate')}
        if attempt['status'] == 'failed':
            entry.update(outcome='invalid_proposal', error=attempt.get('error', '')[:600])
        elif not valid_for_archive(child):
            result = next((r for r in child.get('evaluations', []) if not r['valid']),
                          child.get('full', child.get('smoke', {})))
            entry.update(outcome='invalid_candidate', evaluation_statuses=sorted(
                {r['status'] for r in result.get('cases', []) if r['status'] != 'AC'}))
        elif not valid_for_archive(parent) or child['full']['mean'] > parent['full']['mean']:
            continue
        else:
            sign = 1 if view['config']['direction'] == 'maximize' else -1
            entry.update(outcome='no_improvement', parent_mean=sign * parent['full']['mean'],
                         child_mean=sign * child['full']['mean'], direction=view['config']['direction'])
        history = updates.setdefault(attempt['parent'], deepcopy(parent.get('failure_experience', [])))
        if not any(e['call_id'] == entry['call_id'] for e in history):
            history.append(entry)
            del history[:-8]
    return updates
