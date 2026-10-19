"""Online meta updates at a recorded interval. No trial or automatic rollback."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import time

from ..io import digest, read_json, save_json
from ..provider import BudgetExhausted
from ..candidate_policy import valid_for_archive
from ..search_policy.engine import R0Engine
from . import BASELINE, COMPONENTS, VERSION
from .git_store import PolicyStore
from .policy import components, validate, PolicyError
from .provider import DEFAULT_META_MODEL, DEFAULT_META_REASONING_EFFORT
from .accounting import budget_tokens, dsh_totals
from .initial_policy import starting_files

INTERVAL = 4
MAX_PARA = 32
TASK_KEYS = {'arm', 'parent', 'instruction', 'kind', 'repair', 'donor'}


class OnlineEngine(R0Engine):
    harness_version = VERSION
    meta_protocol = 'online-four-rounds-no-rollback-1'
    meta_interval = INTERVAL

    def __init__(self, root, provider, evaluate, train, config, *, bank=None):
        interval = config.get('meta_interval', self.meta_interval)
        if type(interval) is not int or interval < 1:
            raise ValueError('meta_interval must be a positive integer')
        self.meta_interval = interval
        initial_policy = config.get('meta_initial_policy', 'v52')
        initial_files = starting_files(initial_policy)
        if initial_files:
            validate(initial_files)
        protocol = self.meta_protocol
        if protocol == 'online-four-rounds-no-rollback-1' and (interval != 4 or initial_files):
            protocol = 'online-configurable-interval-no-rollback-2'
        config = {'meta_backend': getattr(provider, 'meta_backend', 'api'), 'meta_model': DEFAULT_META_MODEL,
                  'meta_reasoning_effort': DEFAULT_META_REASONING_EFFORT, **config}
        if provider is not None and (
                provider.meta_model != config['meta_model'] or
                provider.meta_reasoning_effort != config['meta_reasoning_effort']):
            raise ValueError('Meta provider does not match the run configuration')
        config = {**config, 'meta_protocol': protocol,
                  'meta_interval': self.meta_interval, 'meta_baseline': BASELINE, 'meta_max_para': MAX_PARA,
                  'meta_network_research': False,
                  'meta_budget': ('dsh-tokens-exempt-reported-separately-1' if config['meta_backend'] == 'dsh'
                                  else 'shared-ledger-round-boundary')}
        if config.get('budget_scope') == 'solver-plus-meta':
            config['meta_budget'] = 'shared-total-with-request-reservations-1'
        if initial_policy != 'v52':
            config.update(meta_initial_policy=initial_policy,
                          meta_initial_policy_hashes={n: digest(s) for n, s in initial_files.items()})
        super().__init__(root, provider, evaluate, train, config, bank=bank)
        project = Path(__file__).resolve().parents[2]
        self.store = PolicyStore(project, self.root / 'meta' / 'worktrees')
        if 'meta' not in self.s:
            revision = self.store.create(0, BASELINE, initial_files)
            self.s['meta'] = {'active': revision, 'handled_round': 0, 'pending': None,
                              'versions': [{'index': 0, 'revision': revision, 'parent': None,
                                            'activated_after_round': 0,
                                            'reason': ('Preserved v5.2 starting policy' if initial_policy == 'v52'
                                                       else 'Explicit minimal initial harness; fixed solver bank unchanged')}],
                              'updates': [], 'rounds': [], 'round_start': None}
            self.save()
        # Frozen runs can be inspected without invoking policy code or exposing private results.
        self.activate(self.s['meta']['active'])

    @property
    def meta(self):
        return self.s['meta']

    def activate(self, revision):
        version = next(v for v in self.meta['versions'] if v['revision'] == revision)
        self.policy_sources = self.store.verify(version['index'], revision)
        self.components = components(self.policy_sources)

    def view(self):
        view = super().view()
        view['incumbent'] = self.s['incumbent']
        # Only public strategy settings are visible to generated functions.
        view['config'] = {k: self.config[k] for k in (
            'problem_id', 'direction', 'time_limit', 'failure_experience', 'promotion_margin',
            'smoke_count', 'calls', 'token_budget') if k in self.config}
        return view

    def update_population(self):
        super().update_population()
        if self.s['incumbent'] not in self.s['population']:
            raise PolicyError('Active population must retain the official incumbent')

    @staticmethod
    def check_task(task, phase):
        if (not isinstance(task, dict) or set(task) - TASK_KEYS
                or not {'arm', 'parent', 'instruction', 'kind', 'repair'} <= set(task)
                or not isinstance(task['arm'], str) or not isinstance(task['instruction'], str)
                or task['kind'] not in ('edit', 'program')
                or (task['repair'] is not None and not isinstance(task['repair'], str))):
            raise PolicyError('Invalid operator definition; infrastructure fields are fixed')
        if phase == 'repair' and task.get('repair') is not None:
            raise PolicyError('Only one repair is allowed')

    def operator_view(self, view):
        """Match preflight: operator definitions may read active public sources."""
        view = deepcopy(view)
        view['sources'] = {k: self.source(k) for k in view['population']}
        view['feedback'] = {k: self.feedback(k) for k in view['population']}
        return view

    def start_batch(self, phase, tasks):
        for item in tasks:
            self.check_task(item, phase)
        if phase == 'evolution':
            self.meta['round_start'] = {
                'round': self.s['parallel_round'], 'revision': self.meta['active'],
                'before': self.mean(self.s['incumbent']), 'event_start': len(self.s['events']),
                'call_start': self.provider.requests, 'tokens_start': self.provider.charged_tokens}
        # Metadata added by the fixed engine, not returned by generated operators.
        for item in tasks:
            item.update(harness_revision=self.meta['active'], evolution_round=self.s['parallel_round'])
        super().start_batch(phase, tasks)
        for item in self.s['parallel_batch']['tasks']:
            if not isinstance(item['prompt'], dict):
                raise PolicyError('Context must be a JSON object')

    def finish_batch(self, batch):
        super().finish_batch(batch)
        self.complete_round()

    def complete_round(self):
        start = self.meta['round_start']
        if not start or self.s.get('parallel_batch'):
            return
        events = self.s['events'][start['event_start']:]
        valid = [e for e in events if valid_for_archive(self.s['records'].get(e.get('candidate'), {}))]
        ledger_path = self.root / 'llm' / 'ledger.json'
        ledger = self.provider.ledger if self.provider else (read_json(ledger_path) if ledger_path.exists() else [])
        charged = sum(budget_tokens(r) for r in ledger)
        self.meta['rounds'].append({**start, 'after': self.mean(self.s['incumbent']),
            'raw_best_before': self.sign * start['before'],
            'raw_best_after': self.sign * self.mean(self.s['incumbent']),
            'direction': self.config['direction'],
            'gain': self.mean(self.s['incumbent']) - start['before'],
            'requests': len(ledger) - start['call_start'],
            'charged_tokens': charged - start['tokens_start'],
            'statuses': dict(Counter(e.get('status', 'unknown') for e in events)),
            'valid_offspring': len([e for e in valid if not e.get('duplicate')]),
            'parent_improvements': sum(e.get('parent_gain', 0) > 0 for e in events),
            'candidate_ids': [e['candidate'] for e in events if e.get('candidate')]})
        self.meta['round_start'] = None
        self.save()

    def evidence(self):
        """Every historical policy source and every round's measured public effect."""
        blobs, versions = {}, []
        for version in self.meta['versions']:
            sources = self.store.verify(version['index'], version['revision'])
            refs = {}
            for name, source in sources.items():
                ident = digest(source)
                blobs[ident] = source
                refs[name] = ident
            versions.append({**version, 'files': refs})
        records = {ident: {k: r[k] for k in (
            'parent', 'donor', 'lineage', 'initial_slot', 'hypothesis', 'runtime_ratio',
            'is_randomized', 'failure_experience') if k in r} | {
                'valid': valid_for_archive(r), 'public_full': r.get('full'), 'public_smoke': r.get('smoke')}
                   for ident, r in self.view()['records'].items()}
        return {'instructions': 'Use source_blobs to resolve every version.files hash; history is complete.',
                'interfaces': {
                    'parallelism.py': 'select_para(view) -> int 1..32',
                    'parents.py': 'select_parents(view, para, rng) -> active parent IDs, length para',
                    'operator_selection.py': 'select_operators(view, parents, rng) -> arm names, same length',
                    'operators.py': 'define_operator(view, arm, parent) -> arm,parent,instruction,kind,repair[,donor]',
                    'memory.py': 'build_context(view, task) -> dict; update_memory(view, attempts) -> {parent_id: list}',
                    'population.py': 'update_population(view) -> unique valid IDs, including incumbent'},
                'view_contract': {'keys': ['records', 'events', 'population', 'incumbent', 'config', 'round', 'budget'],
                                  'context_extra_keys': ['sources', 'feedback'],
                                  'round': 'zero-based next round index',
                                  'scores': 'full.mean and case.score are signed utilities; higher is better'},
                'settings': {**self.view()['config'], 'meta_interval': self.meta_interval,
                             'meta_initial_policy': self.config.get('meta_initial_policy', 'v52')},
                'budget': self.view()['budget'],
                'active_revision': self.meta['active'], 'versions': versions, 'source_blobs': blobs,
                'updates': deepcopy(self.meta['updates']), 'round_effects': deepcopy(self.meta['rounds']),
                'public_records': records, 'recent_events': deepcopy(self.s['events'][-64:]),
                'population': self.s['population'], 'incumbent': self.s['incumbent']}

    def preflight(self, sources):
        """Interface checks on current evidence; no solver calls, ALE trial or score gate."""
        validate(sources)
        policy = components(sources)
        view = self.view()
        selected = policy.population.update_population(deepcopy(view))
        if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                or self.s['incumbent'] not in selected
                or any(k not in view['records'] or not valid_for_archive(view['records'][k]) for k in selected)):
            raise PolicyError('Invalid active population')
        view['population'] = selected
        para = policy.parallelism.select_para(deepcopy(view))
        if type(para) is not int or not 1 <= para <= MAX_PARA:
            raise PolicyError('Round para must be 1..32')
        parents = policy.parents.select_parents(deepcopy(view), para, random.Random(0))
        if not isinstance(parents, list) or len(parents) != para or any(p not in selected for p in parents):
            raise PolicyError('Invalid joint parent allocation')
        arms = policy.operator_selection.select_operators(deepcopy(view), parents, random.Random(1))
        if not isinstance(arms, list) or len(arms) != para or any(not isinstance(a, str) for a in arms):
            raise PolicyError('Invalid joint operator allocation')
        view = self.operator_view(view)
        for parent, arm in zip(parents, arms):
            item = policy.operators.define_operator(deepcopy(view), arm, parent)
            self.check_task(item, 'evolution')
            if item['parent'] != parent or item['arm'] != arm:
                raise PolicyError('Operators must preserve allocated parents and arms')
            if item.get('donor') and not valid_for_archive(view['records'].get(item['donor'], {})):
                raise PolicyError('Invalid donor')
            if item.get('donor') and item['donor'] not in view['sources']:
                view['sources'][item['donor']] = self.source(item['donor'])
            if not isinstance(policy.memory.build_context(deepcopy(view), item), dict):
                raise PolicyError('Context must be a JSON object')
            if item.get('repair'):
                repair = policy.operators.define_operator(deepcopy(view), item['repair'], parent)
                self.check_task(repair, 'repair')
                if repair['parent'] != parent or repair['arm'] != item['repair']:
                    raise PolicyError('Repair changed allocated parent or operator')
        updates = policy.memory.update_memory(deepcopy(view), deepcopy(self.s['events'][-8:]))
        if not isinstance(updates, dict) or any(k not in view['records'] or not isinstance(v, list)
                                               for k, v in updates.items()):
            raise PolicyError('Invalid memory update')

    def before_round(self):
        self.complete_round()  # Recover the gap between clearing a batch and recording its effect.
        if self.meta.get('error'):
            raise RuntimeError('Meta update halted: ' + self.meta['error'])
        number = self.s.get('parallel_round', 0)
        if self.meta['pending'] or (number > self.meta['handled_round'] and number % self.meta_interval == 0):
            return self.update_policy(number)
        return True

    def proposal_parent(self):
        return self.meta['active']

    def update_policy(self, number):
        if self.meta['pending'] is None:
            # Do not spend the last request on a policy that cannot generate an offspring.
            if self.provider.requests + 1 >= self.s['requested_calls']:
                self.meta['updates'].append({'after_round': number, 'status': 'skipped_no_solver_budget'})
                self.meta['handled_round'] = number
                self.save()
                return True
            prompt = self.evidence()
            # DSH reads complete history from its local workspace, not one HTTP prompt.
            if self.config.get('meta_backend') != 'dsh' and len(json.dumps(prompt).encode()) > 1_000_000:
                raise PolicyError('Complete meta history exceeds request size limit; not silently truncating')
            self.meta['pending'] = {'after_round': number, 'parent': self.proposal_parent(),
                                     'job': {'kind': 'meta', 'prompt': prompt,
                                             'implementation_call': self.provider.requests}}
            self.save()
        pending = self.meta['pending']
        job = pending['job']
        self.provider.max_calls = self.s['requested_calls']
        try:
            self.provider.request_many([job])  # Saved reservations/responses are never reissued.
        except BudgetExhausted:
            # A failed reservation did not make an HTTP call; retain the policy and let the
            # normal solver batch budget check decide whether development can continue.
            self.meta['updates'].append({'after_round': number, 'status': 'skipped_token_reservation'})
            self.meta['handled_round'], self.meta['pending'] = number, None
            self.save()
            return True
        row = self.provider.ledger[job['implementation_call']]
        try:
            if row['status'] != 'ok' or row['kind'] != 'meta':
                raise PolicyError('Meta request failed: ' + row['status'] + '; reserved usage remains charged')
            payload = read_json(self.provider.folder / f"{job['implementation_call']:05d}.response.json")
            choice = payload['choices'][0]
            content = choice['message']['content']
            if choice.get('finish_reason') != 'stop' or digest(content) != row['content_sha256']:
                raise PolicyError('Saved meta response identity mismatch')
            proposal = json.loads(content)
            if (not isinstance(proposal, dict) or set(proposal) != {'action', 'reason', 'files'}
                    or proposal['action'] not in ('keep', 'update')
                    or not isinstance(proposal['reason'], str) or not 1 <= len(proposal['reason']) <= 10000
                    or not isinstance(proposal['files'], dict)
                    or not set(proposal['files']) <= {n + '.py' for n in COMPONENTS}):
                raise PolicyError('Invalid meta proposal schema')
            files = proposal['files']
            parent_version = next(v for v in self.meta['versions'] if v['revision'] == pending['parent'])
            parent_sources = self.store.verify(parent_version['index'], pending['parent'])
            if (proposal['action'] == 'keep') != (not files):
                raise PolicyError('keep requires no edits; update requires changed files')
            for name, source in files.items():
                if not isinstance(source, str):
                    raise PolicyError('Update must contain complete source strings')
                files[name] = source.replace('\r\n', '\n').replace('\r', '\n')
                if files[name] == parent_sources[name]:
                    raise PolicyError('Update must contain changed, complete source strings')
            entry = {'after_round': number, 'call_id': job['implementation_call'],
                     'parent': pending['parent'], 'action': proposal['action'], 'reason': proposal['reason'],
                     'charged_tokens': budget_tokens(row)}
            if row.get('backend') == 'dsh':
                entry['dsh_usage'] = deepcopy(row.get('dsh_usage', {}))
            if proposal['action'] == 'update':
                sources = {**parent_sources, **files}
                started = time.monotonic()
                self.preflight(sources)
                entry['validation_seconds'] = time.monotonic() - started
                index = len(self.meta['versions'])
                revision = self.store.create(index, pending['parent'], files)
                self.meta['versions'].append({'index': index, 'revision': revision, 'parent': pending['parent'],
                    'activated_after_round': number, 'call_id': job['implementation_call'], 'reason': proposal['reason']})
                self.meta['active'] = revision
                entry.update(status='adopted', revision=revision)
            else:
                entry.update(status='kept', revision=self.meta['active'])
            self.meta['updates'].append(entry)
            self.meta['handled_round'], self.meta['pending'] = number, None
            self.save()  # Active pointer and history are one atomic checkpoint.
            self.activate(self.meta['active'])
            # Recompute membership using the new policy before allocating the next round.
            self.update_population()
            self.save()
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.meta['error'] = str(error)[:1500]
            self.save()
            raise PolicyError('Meta update halted without rollback: ' + self.meta['error']) from error
        return not (self.root / 'STOP_AFTER_CURRENT').exists()

    def run(self):
        if self.meta.get('error'):
            raise PolicyError('Meta update halted: ' + self.meta['error'])
        # Adoption is checkpointed before population recomputation; recover that gap.
        if self.s.get('population') and not self.s.get('parallel_batch') and not self.s['frozen']:
            self.update_population()
        super().run()
        if self.provider:
            self.complete_round()

    def finalize(self):
        if self.meta.get('error') or self.meta['pending']:
            raise PolicyError('Resolve pending/failed meta update before freezing')
        self.complete_round()
        winner = super().finalize()
        ledger_path = self.root / 'llm' / 'ledger.json'
        ledger = self.provider.ledger if self.provider else (read_json(ledger_path) if ledger_path.exists() else [])
        report = read_json(self.root / 'report.json')
        report.update(meta_requests=sum(r['kind'] == 'meta' for r in ledger),
                      solve_requests=sum(r['kind'] != 'meta' for r in ledger),
                      charged_tokens=sum(budget_tokens(r) for r in ledger),
                      harness_baseline=BASELINE, active_harness_revision=self.meta['active'])
        report.update(meta_interval=self.meta_interval,
                      initial_policy=self.config.get('meta_initial_policy', 'v52'),
                      initial_harness_revision=self.meta['versions'][0]['revision'])
        report['meta_history'] = {k: deepcopy(self.meta[k]) for k in ('versions', 'updates', 'rounds')}
        if self.config.get('meta_backend') == 'dsh':
            report['dsh_usage'] = dsh_totals(ledger)
            report['token_budget_scope'] = 'solver_only_dsh_exempt'
        save_json(self.root / 'report.json', report)
        return winner
