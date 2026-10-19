"""One problem-only call yielding four short cold-start hints."""
import json

from .io import digest, read_json, save_json


PLAN_RULE = (
    'You are a senior algorithm designer. '
    'Analyze the problem structure, constraints, objective and computational budget. '
    'Propose exactly four substantially different plausible algorithm design directions '
    'to diversify four independent cold-start solvers. Distinguish their representations, '
    'construction or search mechanisms, not just parameter values. '
    'Return only JSON {"directions": ["...", "...", "...", "..."]}. '
    'Each direction must be one concise sentence on one line, at most 500 characters. '
    'Give only an idea hint, no source code, pseudocode, diagnostic tools or full algorithm skeleton. '
    'Invent from the supplied problem only; do not reconstruct named submissions.'
)


def parse_directions(content):
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {'directions'}:
        raise ValueError('Plan must contain only directions')
    directions = value['directions']
    if not isinstance(directions, list) or len(directions) != 4:
        raise ValueError('Plan must contain exactly four directions')
    if any(not isinstance(d, str) or not d.strip() or len(d) > 500
           or '\n' in d or '\r' in d or '```' in d for d in directions):
        raise ValueError('Each Plan direction must be a short single-line idea hint')
    directions = [d.strip() for d in directions]
    if len({d.casefold() for d in directions}) != 4:
        raise ValueError('Plan directions must be distinct')
    return directions


def ensure_plan(engine):
    """Reserve once; reuse a completed response; never silently retry a failed call."""
    state, provider, root = engine.s, engine.provider.planner, engine.root
    if 'plan_call' not in state:
        if state['records'] or state['events'] or state['pending'] or engine.provider.requests or provider.requests:
            raise ValueError('Plan must precede all solver requests')
        state['plan_call'] = provider.requests
        engine.save()
    call = state['plan_call']
    if call < provider.requests:
        row = provider.ledger[call]
        if row['status'] != 'ok' or row.get('kind') != 'plan':
            raise ValueError('Initial Plan request failed or was interrupted; no automatic retry')
        payload = read_json(provider.root / 'llm' / f'{call:05d}.response.json')
        choice = payload['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Incomplete saved Plan response')
        content = choice['message']['content']
        if digest(content) != row.get('content_sha256'):
            raise ValueError('Saved Plan response changed')
    elif call == provider.requests and not state.get('plan'):
        response = provider.request('plan', {})
        content = response['content']
    else:
        raise ValueError('Plan request ledger inconsistent')
    artifact = {'call_id': call, 'content_sha256': digest(content),
                'directions': parse_directions(content)}
    if state.get('plan') is not None and state['plan'] != artifact:
        raise ValueError('Saved Plan changed')
    path = root / 'plan.json'
    if path.exists() and read_json(path) != artifact:
        raise ValueError('Plan artifact changed')
    state['plan'] = artifact
    engine.save()
    save_json(path, artifact)
