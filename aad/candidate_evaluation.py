"""One seed-zero evaluation for every algorithm."""
import json
import re
import statistics

from .candidate_policy import program, edit

EVALUATION_PROTOCOL = 'declared-randomness-2-seed-zero'
SEEDS = (0,)
RANDOMNESS_RULE = (
    'Include is_randomized as a JSON boolean describing the resulting complete algorithm, '
    'including all randomness inherited from the parent. Use true for any randomized '
    'construction/search, even when pseudorandomness has a fixed reproducible seed; '
    'use false only if no randomized decisions are used. '
    'For randomized algorithms initialize every PRNG from the framework-defined C++ macro '
    'AAD_RANDOM_SEED (an unsigned integer, tested with seed 0). '
    'Do not define, undefine or override AAD_RANDOM_SEED. Do not seed from time, random_device '
    'or external entropy. Deterministic algorithms are evaluated once; randomized algorithms '
    'are also evaluated once with seed 0 and ranked by their mean case score.'
)


def evaluation_seeds(is_randomized):
    if type(is_randomized) is not bool:
        raise ValueError('is_randomized must be a JSON boolean')
    return SEEDS if is_randomized else SEEDS[:1]


def parse_candidate(kind, parent_code, content):
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError('Expected candidate JSON object')
    randomized = payload.get('is_randomized')
    evaluation_seeds(randomized)
    if kind == 'edit':
        code, hypothesis = edit(parent_code, content)
    elif kind == 'program':
        hypothesis = payload.get('hypothesis')
        if not isinstance(hypothesis, str) or not 1 <= len(hypothesis) <= 3000:
            raise ValueError('Missing concise change hypothesis')
        if not isinstance(payload.get('code'), str):
            raise ValueError('Missing candidate code')
        code = program(payload['code'])
    else:
        raise ValueError('Unknown candidate kind')
    if re.search(r'^\s*#\s*(?:define|undef)\s+AAD_RANDOM_SEED\b', code, re.MULTILINE):
        raise ValueError('Candidate must not override AAD_RANDOM_SEED')
    return code, hypothesis, {'is_randomized': randomized}


def seeded_source(code, seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Expected evaluation seed 0')
    return f'#define AAD_RANDOM_SEED {seed}ULL\n#line 1 "candidate.cpp"\n' + code


def from_rows(rows):
    if not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty unique evaluation cases')
    return {'valid': all(r['status'] == 'AC' for r in rows),
            'raw_mean': statistics.mean(r['raw_score'] for r in rows),
            'mean': statistics.mean(r['score'] for r in rows), 'cases': rows}


def aggregate_evaluations(results):
    if not results:
        raise ValueError('No evaluations')
    names = [r['name'] for r in results[0]['cases']]
    if any([r['name'] for r in result['cases']] != names for result in results):
        raise ValueError('Different evaluation case sets')
    rows = []
    for group in zip(*(result['cases'] for result in results), strict=True):
        # Preserve one actual output and its actual score for coherent model feedback.
        sample = min(group, key=lambda row: (row['status'] == 'AC', row['score']))
        rows.append({**sample, 'sample_raw_score': sample['raw_score'],
                     'raw_score': statistics.mean(r['raw_score'] for r in group),
                     'score': statistics.mean(r['score'] for r in group),
                     'seconds': statistics.mean(r['seconds'] for r in group)})
    return {**from_rows(rows), 'evaluation_count': len(results)}
