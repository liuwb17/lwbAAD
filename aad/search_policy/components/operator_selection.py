"""4. Joint operator allocation after the complete parent allocation."""
from .operators import complementary_donor


def select_operators(view, parents, rng):
    arms = []
    for index, parent in enumerate(parents):
        if index == 0:
            arm = 'local'
        elif index == 1:
            arm = 'redesign'
        elif index == 2:
            arm = 'crossover' if complementary_donor(view, parent) else rng.choice(('redesign', 'local'))
        elif parent == view['population'][0] and view['records'][parent]['runtime_ratio'] <= 0.30:
            arm = 'runtime'
        else:
            arm = 'redesign' if view['round'] % 2 == 0 else 'crossover'
        arms.append(arm)
    return arms
