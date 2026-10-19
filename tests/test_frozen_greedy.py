import random
import unittest
from aad.meta_online.initial_policy import starting_files
from aad.meta_online.policy import components, validate

class FrozenGreedyTests(unittest.TestCase):
    def test_only_parent_selection_changes(self):
        minimal, greedy = starting_files('minimal'), starting_files('greedy')
        self.assertEqual([k for k in minimal if minimal[k] != greedy[k]], ['parents.py'])
        validate(greedy)

    def test_four_current_incumbents_and_next_batch_switch(self):
        policy = components(starting_files('greedy'))
        view = {'population': ['weak', 'best', 'next'], 'incumbent': 'best'}
        self.assertEqual(policy.parallelism.select_para(view), 4)
        first = policy.parents.select_parents(view, 4, random.Random(7))
        self.assertEqual(first, ['best'] * 4)
        view['incumbent'] = 'next'
        self.assertEqual(first, ['best'] * 4)
        self.assertEqual(policy.parents.select_parents(view, 4, random.Random(8)), ['next'] * 4)

    def test_minimal_keeps_uniform_sampling_and_same_feedback(self):
        minimal, greedy = [components(starting_files(n)) for n in ['minimal', 'greedy']]
        view = {'population': ['weak', 'best'], 'incumbent': 'best',
                'sources': {'best': 'int main(){}'}, 'feedback': {'best': {'score': 123}}}
        sampled = minimal.parents.select_parents(view, 100, random.Random(0))
        self.assertEqual(set(sampled), {'weak', 'best'})
        for arm in ['mutation', 'repair']:
            a = minimal.operators.define_operator(view, arm, 'best')
            b = greedy.operators.define_operator(view, arm, 'best')
            self.assertEqual(a, b)
            self.assertEqual(minimal.memory.build_context(view, a), greedy.memory.build_context(view, b))

if __name__ == '__main__': unittest.main()
