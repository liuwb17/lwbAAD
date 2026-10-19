"""Four-lineage synchronous rounds; concurrent transport, single-writer evaluation.

Reserve every call before transport. A saved batch fixes prompts, random choices
and call IDs. Recovery never resends a reserved request with unknown usage.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import time
import urllib.error
import urllib.request

from .io import digest, read_json, save_json
from .provider import BudgetExhausted
from .candidate_policy import pool, valid_for_archive


def request_many(provider, jobs):
    """Complete new reservations concurrently, recover old ones without HTTP.

    Only the caller thread modifies the ledger or writes checkpoint files.
    The engine has already persisted each job's implementation_call.
    """
    new = []
    for job in jobs:
        call = job['implementation_call']
        if call < provider.requests:
            continue
        if call != provider.requests + len(new):
            raise ValueError('Noncontiguous parallel call IDs')
        body = provider.request_body(job['kind'], {**job['prompt'], 'problem': provider.problem})
        reserve = len(json.dumps(body).encode('utf-8')) + 1024 + 65536
        new.append((call, job['kind'], body, reserve))
    if provider.requests + len(new) > provider.max_calls:
        raise BudgetExhausted('Parallel call budget exhausted')
    if (provider.max_tokens_total > 0 and provider.charged_tokens + sum(j[3] for j in new)
            > provider.max_tokens_total):
        raise BudgetExhausted('Token budget cannot reserve the complete parallel batch')
    for call, kind, body, reserve in new:
        save_json(provider.folder / f'{call:05d}.request.json', body)
        provider.ledger.append({'id': call, 'kind': kind, 'status': 'started',
                                'reserved_tokens': reserve, 'requested_model': body['model'],
                                'requested_reasoning_effort': body.get('reasoning_effort')})
        save_json(provider.path, provider.ledger)

    def transport(body):
        start = time.monotonic()
        try:
            request = urllib.request.Request('https://api.deepseek.com/chat/completions',
                data=json.dumps(body).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + provider.key})
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError('Response too large')
            return json.loads(raw), None, time.monotonic() - start
        except Exception as error:
            return None, error, time.monotonic() - start

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(transport, body): call for call, _, body, _ in new}
        for future in as_completed(futures):
            call = futures[future]
            payload, error, seconds = future.result()
            row = provider.ledger[call]
            row['seconds'] = seconds
            try:
                if error is not None:
                    raise error
                save_json(provider.folder / f'{call:05d}.response.json', payload)
                row.update(usage=payload.get('usage', {}), response_model=payload.get('model'),
                           system_fingerprint=payload.get('system_fingerprint'))
                choice = payload['choices'][0]
                row['finish_reason'] = choice.get('finish_reason')
                if row['finish_reason'] != 'stop':
                    raise ValueError('Incomplete response')
                content = choice['message'].get('content')
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('Empty final response')
                row.update(status='ok', content_sha256=digest(content))
            except urllib.error.HTTPError as error:
                row.update(status=f'http_{error.code}', error=f'HTTP {error.code}')
            except (urllib.error.URLError, TimeoutError):
                row.update(status='network_error', error='Network failure; usage unknown')
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                row.update(status='invalid_response', error=str(error)[:1500])
            finally:
                save_json(provider.path, provider.ledger)


INSTRUCTIONS = {
    'local': 'Address one measured weakness. Up to six coordinated edits across helpers and call '
             'sites are allowed. Preserve unrelated working mechanisms.',
    'redesign': 'Escape representation and reachable-state limitations. Make a substantive '
                'structural redesign retaining useful parent mechanisms. Return a complete program.',
    'crossover': 'Integrate a complementary mechanism from the donor into the main parent. Related '
                 'parts may change together. Do not switch whole solvers, average outputs, or route known cases.',
    'independent': 'Invent a complete optimizer from the specification without seed code. '
                   'Design construction and useful search; prioritize algorithm quality.',
    'repair': 'Repair the demonstrated error including seed-specific failure. Keep intended search '
              'quality. Return the full corrected program.',
    'runtime': 'The main parent uses at most 30% of the time limit on average. Diagnose early '
               'termination and reachable-state limits. Design score-improving continuation or '
               'additional search that makes useful use of the available time. Do not add padding, '
               'sleeps or repeated identical work. Finishing early is fine if no useful search exists; '
               'explain in comments. Return a complete program.',
}


def task(arm, parent=None, **metadata):
    return {'arm': arm, 'parent': parent, 'instruction': INSTRUCTIONS[arm], **metadata}


def round_tasks(engine):
    records = engine.s['records']
    archive = pool(records)
    # Initialization retries only missing slots, so no fifth valid lineage is created.
    completed = {r.get('initial_slot') for r in records.values() if valid_for_archive(r)}
    missing = [i for i in range(4) if i not in completed]
    if missing and not engine.config.get('allow_invalid_initials', False):
        tasks = []
        for index in missing:
            item = task('independent', initial_slot=index)
            if engine.config['plan']:
                item['initial_direction'] = index
                item['instruction'] += ' Suggested design direction: ' + engine.s['plan']['directions'][index]
            tasks.append(item)
        engine.s['root_attempts'] += len(tasks)
        return 'initial', tasks
    best = archive[0]
    others = [k for k in archive if records[k]['lineage'] != records[best]['lineage']]
    representatives = {}
    for ident in others:
        representatives.setdefault(records[ident]['lineage'], ident)
    if len(representatives) != 3 and not engine.config.get('allow_invalid_initials', False):
        raise ValueError('Parallel search requires exactly three other lineages')
    scores = {r['name']: r['score'] for r in records[best]['full']['cases']}
    donor = max(others or [best], key=lambda k: sum(max(0, r['score'] - scores[r['name']])
                                         for r in records[k]['full']['cases']))
    if any(r['score'] > scores[r['name']] for r in records[donor]['full']['cases']):
        third = task('crossover', best, donor=donor)
    else:
        third = task(random.choice(('redesign', 'local')), best)
    number = engine.s.get('parallel_round', 0)
    if records[best]['runtime_ratio'] <= 0.30:
        fourth = task('runtime', best)
    else:
        parent = random.choice(list(representatives.values()) or [best])
        fourth = task('redesign', parent) if number % 2 == 0 else task('crossover', parent, donor=best)
    engine.s['parallel_round'] = number + 1
    return 'evolution', [task('local', best), task('redesign', best), third, fourth]


def start_batch(engine, phase, tasks):
    for offset, item in enumerate(tasks):
        parent = item['parent']
        prompt = {'operation': item['arm'], 'instruction': item['instruction']}
        if parent:
            prompt.update(parent_source=engine.source(parent), parent_sha256=parent,
                          feedback=engine.feedback(parent),
                          recent_experiments=[{k: e.get(k) for k in ('arm', 'status', 'parent_gain', 'error')}
                                              for e in engine.s['events'][-8:]])
        if item.get('donor'):
            prompt['donor_source'] = engine.source(item['donor'])
        prompt = engine.augment_task(item, prompt)
        item.update(prompt=prompt, kind='edit' if item['arm'] == 'local' else 'program',
                    implementation_call=engine.provider.requests + offset)
    engine.s['parallel_batch'] = {'phase': phase, 'tasks': tasks}
    engine.s['effective_calls'] = max(engine.s['effective_calls'], engine.provider.requests + len(tasks))
    engine.save()


def consume(engine, item):
    if item.get('done'):
        return
    parent = item['parent']
    before = engine.mean(engine.s['incumbent']) if engine.s['incumbent'] else 0
    row = engine.provider.ledger[item['implementation_call']]
    try:
        if row['status'] != 'ok':
            raise ValueError(row.get('error', 'Reserved request did not complete successfully'))
        if row['kind'] != item['kind']:
            raise ValueError('Saved response kind differs')
        if 'candidate' not in item:
            payload = read_json(engine.provider.folder / f"{item['implementation_call']:05d}.response.json")
            choice = payload['choices'][0]
            content = choice['message']['content']
            if choice.get('finish_reason') != 'stop' or digest(content) != row['content_sha256']:
                raise ValueError('Saved response changed')
            engine.note_response(item, content)
            code, hypothesis, metadata = engine.parse_response(item['kind'], parent, content)
            ident = digest(code)
            if ident in engine.s['records']:
                if any(engine.s['records'][ident].get(k) != v for k, v in metadata.items()):
                    raise ValueError('Conflicting metadata for identical candidate source')
                item['duplicate'] = True
            else:
                folder = engine.root / 'candidates' / ident
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'solution.cpp').write_text(code, encoding='utf-8')
                initial_slot = engine.s['records'][parent]['initial_slot'] if parent else item['initial_slot']
                engine.s['records'][ident] = {
                    'parent': parent, 'donor': item.get('donor'),
                    'lineage': engine.s['records'][parent]['lineage'] if parent else ident,
                    'initial_slot': initial_slot, 'hypothesis': hypothesis,
                    'implementation_call': item['implementation_call'], 'implementation_kind': item['kind'],
                    **metadata}
            item.update(candidate=ident, **metadata)
            engine.save()
        if item.get('duplicate'):
            item['status'] = 'duplicate'
        else:
            engine.qualify(item['candidate'], item)
            # Repairs are collected for the whole round, never a single shared slot.
            engine.s['repair'] = None
    except (ValueError, RuntimeError) as error:
        item.update(status='failed', error=str(error)[:1500])
    record = engine.s['records'].get(item.get('candidate'), {})
    item['parent_gain'] = (engine.mean(item['candidate']) - engine.mean(parent)
                           if parent and valid_for_archive(record)
                           and valid_for_archive(engine.s['records'][parent]) else 0)
    item['gain'] = (engine.mean(engine.s['incumbent']) if engine.s['incumbent'] else 0) - before
    item['tokens'] = row.get('usage', {}).get('total_tokens', row.get('reserved_tokens', 0))
    engine.record_attempt(item)
    item['done'] = True
    engine.s['events'].append({k: v for k, v in item.items() if k not in ('prompt', 'done')})
    engine.save()


def run_parallel(engine):
    provider = engine.provider
    engine.s.setdefault('requested_calls', provider.max_calls)
    engine.s.setdefault('effective_calls', provider.max_calls)
    while engine.s.get('parallel_batch') or provider.requests < engine.s['requested_calls']:
        if (engine.root / 'STOP_AFTER_CURRENT').exists():
            return
        if not engine.s.get('parallel_batch'):
            phase, tasks = round_tasks(engine)
            start_batch(engine, phase, tasks)
        batch = engine.s['parallel_batch']
        provider.max_calls = engine.s['effective_calls']
        try:
            provider.request_many(batch['tasks'])
        except BudgetExhausted:
            # Token budget is a hard limit: do not partially submit a batch.
            batch['budget_exhausted'] = True
            engine.save()
            return
        fatal = next((provider.ledger[t['implementation_call']]['status'] for t in batch['tasks']
                      if provider.ledger[t['implementation_call']]['status'] in ('http_401', 'http_402')), None)
        for item in batch['tasks']:
            consume(engine, item)
        repairs = [task('repair', item['candidate']) for item in batch['tasks']
                   if item['status'] == 'invalid_candidate' and item['arm'] != 'repair']
        if repairs:
            start_batch(engine, 'repair', repairs)
        else:
            engine.s['parallel_batch'] = None
            engine.save()
        if fatal:
            raise RuntimeError('HTTP ' + fatal.removeprefix('http_'))
        print({'requests': provider.requests, 'phase': batch['phase'],
               'effective_calls': engine.s['effective_calls'],
               'statuses': [t['status'] for t in batch['tasks']]}, flush=True)
