"""Fixed round loop; editable decisions live in six components/*.py files.

Reuse v5.2 transport, candidate parsing, official measurement, promotion,
response recovery and freeze/private gates. No meta-agent is run here.
"""
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

from ..io import digest
from ..provider import BudgetExhausted
from ..candidate_policy import valid_for_archive
from ..parallel_search import consume
from ..fixed_initials import FixedInitialEngine
from ..evaluation_feedback import score_record
from . import __version__
from .components import parallelism, parents, operators, operator_selection, memory, population


class R0Engine(FixedInitialEngine):
    harness_version = __version__
    components = SimpleNamespace(parallelism=parallelism, parents=parents, operators=operators,
                                 operator_selection=operator_selection, memory=memory, population=population)

    def before_round(self):
        """Extension point at a completed-round boundary; false ends development."""
        return True

    def __init__(self, root, provider, evaluate, train, config, *, bank=None):
        component_dir = Path(__file__).parent / 'components'
        config = {**config, 'component_hashes': {
                      p.name: digest(p.read_text(encoding='utf-8')) for p in sorted(component_dir.glob('*.py'))},
                  'round_protocol': 'parents-then-operators-batch-memory-1'}
        super().__init__(root, provider, evaluate, train, config, bank=bank)

    def view(self):
        """Detached public evidence, without an engine/provider/filesystem handle."""
        return deepcopy({'records': {key: score_record(record) for key, record in self.s['records'].items()},
                         'events': self.s['events'],
                         'population': self.s.get('population', []),
                         'config': self.config, 'round': self.s.get('parallel_round', 0),
                         'budget': {'requested_calls': self.s.get('requested_calls', self.config.get('calls')),
                                    'requests': self.provider.requests if self.provider else 0,
                                    'charged_tokens': self.provider.charged_tokens if self.provider else 0}})

    def update_population(self):
        selected = self.components.population.update_population(self.view())
        if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                or any(k not in self.s['records'] or not valid_for_archive(self.s['records'][k])
                       for k in selected)):
            raise ValueError('Population must contain unique valid candidate IDs')
        self.s['population'] = selected

    def operator_view(self, view):
        return view

    def round_tasks(self):
        view = self.view()
        para = self.components.parallelism.select_para(view)
        if type(para) is not int or para < 1:
            raise ValueError('Round para must be a positive integer')
        selected_parents = self.components.parents.select_parents(view, para, random)
        if len(selected_parents) != para or any(p not in view['population'] for p in selected_parents):
            raise ValueError('Expected one active main parent per slot')
        arms = self.components.operator_selection.select_operators(view, selected_parents, random)
        if len(arms) != para:
            raise ValueError('Expected one operator per slot')
        view = self.operator_view(view)
        tasks = [self.components.operators.define_operator(view, arm, parent)
                 for arm, parent in zip(arms, selected_parents)]
        if any(t['parent'] != p or t['arm'] != a for t, p, a in zip(tasks, selected_parents, arms)):
            raise ValueError('Operator definitions cannot replace selected main parents/operators')
        self.s['parallel_round'] = view['round'] + 1
        return tasks

    def start_batch(self, phase, tasks):
        view = self.view()
        references = {t['parent'] for t in tasks} | {t['donor'] for t in tasks if t.get('donor')}
        if any(k not in view['records'] for k in references):
            raise ValueError('Unknown parent/donor')
        for item in tasks:
            if item['kind'] not in ('edit', 'program'):
                raise ValueError('Unsupported response kind')
            if phase != 'repair' and not valid_for_archive(view['records'][item['parent']]):
                raise ValueError('Evolution requires a valid parent')
            if item.get('donor') and not valid_for_archive(view['records'][item['donor']]):
                raise ValueError('Donor must have a valid official evaluation')
        view['sources'] = {k: self.source(k) for k in references}
        view['feedback'] = {t['parent']: self.feedback(t['parent']) for t in tasks}
        for offset, item in enumerate(tasks):
            item.update(prompt=deepcopy(self.components.memory.build_context(view, item)),
                        implementation_call=self.provider.requests + offset)
        self.s['parallel_batch'] = {'phase': phase, 'tasks': tasks}
        self.s['effective_calls'] = max(self.s['effective_calls'], self.provider.requests + len(tasks))
        self.save()  # Choices, prompts and call IDs precede every HTTP reservation.

    def record_attempt(self, pending):
        # consume() saves each attempt; memories are committed only at batch end.
        pass

    def finish_batch(self, batch):
        updates = self.components.memory.update_memory(self.view(), deepcopy(batch['tasks']))
        for parent, history in updates.items():
            if parent not in self.s['records'] or not isinstance(history, list):
                raise ValueError('Invalid memory update')
            self.s['records'][parent]['failure_experience'] = history
        self.update_population()
        view = self.view()
        repairs = [self.components.operators.define_operator(view, t['repair'], t['candidate'])
                   for t in batch['tasks'] if t['status'] == 'invalid_candidate' and t.get('repair')]
        if repairs:
            self.start_batch('repair', repairs)
        else:
            self.s['parallel_batch'] = None
            self.save()

    def run(self):
        if self.s['frozen'] or self.s.get('development_closed') or self.s['validation_started']:
            raise ValueError('Development is closed')
        if (self.root / 'STOP_AFTER_CURRENT').exists():
            return
        self.initialize()
        if 'population' not in self.s:
            self.update_population()
        provider = self.provider
        self.s.setdefault('requested_calls', provider.max_calls)
        self.s.setdefault('effective_calls', provider.max_calls)
        while self.s.get('parallel_batch') or provider.requests < self.s['requested_calls']:
            if (self.root / 'STOP_AFTER_CURRENT').exists():
                return
            if not self.s.get('parallel_batch'):
                if not self.before_round():
                    return
                self.start_batch('evolution', self.round_tasks())
            batch = self.s['parallel_batch']
            provider.max_calls = self.s['effective_calls']
            try:
                provider.request_many(batch['tasks'])
            except BudgetExhausted:
                batch['budget_exhausted'] = True
                self.save()
                return
            fatal = next((provider.ledger[t['implementation_call']]['status'] for t in batch['tasks']
                          if provider.ledger[t['implementation_call']]['status'] in ('http_401', 'http_402')), None)
            for item in batch['tasks']:
                consume(self, item)
            self.finish_batch(batch)
            if fatal:
                raise RuntimeError('HTTP ' + fatal.removeprefix('http_'))
            print({'requests': provider.requests, 'phase': batch['phase'],
                   'effective_calls': self.s['effective_calls'],
                   'statuses': [t['status'] for t in batch['tasks']]}, flush=True)
