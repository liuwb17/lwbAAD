"""6. Active archive membership; complete source/evaluation history is retained."""
from ...candidate_policy import valid_for_archive


def update_population(view):
    records = view['records']
    ranked = sorted((k for k in records if valid_for_archive(records[k])),
                    key=lambda k: records[k]['full']['mean'], reverse=True)
    population, counts = [], {}
    for ident in ranked:
        lineage = records[ident]['lineage']
        if counts.get(lineage, 0) < 2:
            population.append(ident)
            counts[lineage] = counts.get(lineage, 0) + 1
        if len(population) == 8:
            break
    return population
