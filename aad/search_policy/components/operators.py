"""3. Operator definitions: instruction, output kind, donor and one-shot repair."""

OPERATORS = {
    'local': {
        'kind': 'edit', 'repair': 'repair',
        'instruction': 'Address one measured weakness. Up to six coordinated edits across helpers and call '
                       'sites are allowed. Preserve unrelated working mechanisms.'},
    'redesign': {
        'kind': 'program', 'repair': 'repair',
        'instruction': 'Escape representation and reachable-state limitations. Make a substantive '
                       'structural redesign retaining useful parent mechanisms. Return a complete program.'},
    'crossover': {
        'kind': 'program', 'repair': 'repair',
        'instruction': 'Integrate a complementary mechanism from the donor into the main parent. Related '
                       'parts may change together. Do not switch whole solvers, average outputs, or route known cases.'},
    'runtime': {
        'kind': 'program', 'repair': 'repair',
        'instruction': 'The main parent uses at most 30% of the time limit on average. Diagnose early '
                       'termination and reachable-state limits. Design score-improving continuation or '
                       'additional search that makes useful use of the available time. Do not add padding, '
                       'sleeps or repeated identical work. Finishing early is fine if no useful search exists; '
                       'explain in comments. Return a complete program.'},
    'repair': {
        'kind': 'program', 'repair': None,
        'instruction': 'Repair the demonstrated error including seed-specific failure. Keep intended search '
                       'quality. Return the full corrected program.'},
}


def complementary_donor(view, parent):
    records = view['records']
    others = [k for k in view['population'] if records[k]['lineage'] != records[parent]['lineage']]
    if not others:
        return None
    scores = {r['name']: r['score'] for r in records[parent]['full']['cases']}
    donor = max(others, key=lambda k: sum(max(0, r['score'] - scores[r['name']])
                                         for r in records[k]['full']['cases']))
    return donor if any(r['score'] > scores[r['name']] for r in records[donor]['full']['cases']) else None


def define_operator(view, arm, parent):
    item = {'arm': arm, 'parent': parent, **OPERATORS[arm]}
    if arm == 'crossover':
        best = view['population'][0]
        donor = complementary_donor(view, parent) if parent == best else best
        if donor is None:
            raise ValueError('Crossover requires a donor')
        item['donor'] = donor
    return item
