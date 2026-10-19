"""2. Joint parent allocation for the whole round; repetition is intentional."""


def select_parents(view, para, rng):
    records, population = view['records'], view['population']
    best = population[0]
    representatives = {}
    for ident in population:
        if records[ident]['lineage'] != records[best]['lineage']:
            representatives.setdefault(records[ident]['lineage'], ident)
    parents = [best] * min(3, para)
    for _ in range(3, para):
        parents.append(best if records[best]['runtime_ratio'] <= 0.30
                       else rng.choice(list(representatives.values())))
    return parents
