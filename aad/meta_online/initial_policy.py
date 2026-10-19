"""Explicit starting harnesses; solver initial sources are never changed here."""

MINIMAL = {
    'parallelism.py': '''def select_para(view):
    return 4
''',
    'parents.py': '''def select_parents(view, para, rng):
    return [rng.choice(view['population']) for _ in range(para)]
''',
    'operator_selection.py': '''def select_operators(view, parents, rng):
    return ['mutation' for parent in parents]
''',
    'operators.py': '''def define_operator(view, arm, parent):
    if arm == 'mutation':
        return {'arm': arm, 'parent': parent, 'kind': 'program', 'repair': 'repair',
                'instruction': 'Improve the parent algorithm using its public evaluation feedback. '
                               'Choose the changes you judge useful, including structural changes. '
                               'Return the complete improved program.'}
    if arm == 'repair':
        return {'arm': arm, 'parent': parent, 'kind': 'program', 'repair': None,
                'instruction': 'Repair the demonstrated error including seed-specific failure. '
                               'Keep intended search quality. Return the full corrected program.'}
    raise ValueError('Unknown operator')
''',
    'memory.py': '''def build_context(view, item):
    parent = item['parent']
    return {'operation': item['arm'], 'instruction': item['instruction'],
            'parent_source': view['sources'][parent], 'parent_sha256': parent,
            'feedback': view['feedback'][parent]}

def update_memory(view, attempts):
    return {}
''',
    'population.py': '''from ...candidate_policy import valid_for_archive

def update_population(view):
    records = view['records']
    ranked = sorted((k for k in records if valid_for_archive(records[k])),
                    key=lambda k: records[k]['full']['mean'], reverse=True)
    selected = ranked[:8]
    incumbent = view['incumbent']
    if incumbent not in selected:
        selected = selected[:7] + [incumbent]
    return selected
''',
}


def starting_files(name):
    if name == 'v52':
        return {}
    if name in ('minimal', 'diversity'):
        return dict(MINIMAL)
    if name == 'greedy':
        return {**MINIMAL, 'parents.py': '''def select_parents(view, para, rng):
    return [view['incumbent'] for _ in range(para)]
'''}
    raise ValueError('Unknown initial harness policy: ' + str(name))
