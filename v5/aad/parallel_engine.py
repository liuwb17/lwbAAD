"""Task-independent v5 search, raw directional scores, training-only freeze.

The v5 run loop, source contract, archive and request recovery are reused.
Result ``score``/``mean`` are signed utilities; ``raw_score``/``raw_mean``
retain official scores. No normalization or AHC001 diagnostics are applied.
"""
import json
import math
import statistics
from pathlib import Path

from .io import digest, read_json, save_json
from .evolution_engine import V5Engine
from .candidate_policy import CYCLE, pool, valid_for_archive
from .evolution_provider import V5Provider
from .search_plan import PLAN_RULE, ensure_plan
from .parallel_search import request_many, run_parallel
from .search_memory import augment_task, note_response, record_attempt
from .candidate_evaluation import (EVALUATION_PROTOCOL, RANDOMNESS_RULE, aggregate_evaluations,
                              evaluation_seeds, from_rows, parse_candidate)


class V51Provider(V5Provider):
    def __init__(self, *args, problem, **kwargs):
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError('Task statement is required')
        self.problem = problem
        self._planner = None
        super().__init__(*args, **kwargs)

    @property
    def planner(self):
        if self._planner is None:
            self._planner = V51Provider(self.root / 'planning', self.model, 1, 0,
                                       key=self.key, problem=self.problem)
        return self._planner

    def request(self, kind, task):
        # v5 merges task after its legacy problem field. Replace it explicitly.
        return super().request(kind, {**task, 'problem': self.problem})

    def request_many(self, jobs):
        return request_many(self, jobs)

    def request_body(self, kind, task):
        if kind != 'plan':
            body = super().request_body(kind, task)
            rule = ('Return only JSON with hypothesis (short string), code (complete executable '
                    'C++17 source ending with // AAD_V5_COMPLETE), and is_randomized (boolean). '
                    if kind == 'program' else
                    'Return only JSON with hypothesis, parent_sha256, edits (1..6 objects with old/new '
                    'source strings), and is_randomized (boolean). Anchors must be unique, disjoint '
                    'substrings of the ORIGINAL parent. Preserve // AAD_V5_COMPLETE. ')
            body['messages'][0]['content'] = (
                'You are an optimization algorithm researcher. Invent from the problem and this run only; '
                'never reconstruct named submissions. Reason about search and feasible states before '
                'implementation. Do not sacrifice algorithm quality merely to minimize source size. '
                + rule + RANDOMNESS_RULE)
            body['response_format'] = {'type': 'json_object'}
            return body
        return {'model': self.model, 'max_tokens': 65536,
                'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low',
                'response_format': {'type': 'json_object'},
                'messages': [{'role': 'system', 'content': PLAN_RULE},
                             {'role': 'user', 'content': json.dumps({'problem': self.problem},
                                                                    ensure_ascii=False)}]}


class V51Engine(V5Engine):
    def __init__(self, root, provider, evaluate, train, config):
        if any(key in config for key in ('repeats', 'repeat_slack')):
            raise ValueError('Legacy repeat confirmation settings are no longer supported')
        config = {'initial_valid_roots': 4, 'max_initial_attempts': 20,
                  'promotion_margin': 0.0, 'smoke_count': 6,
                  'plan': False, 'failure_experience': False,
                  **config}
        config['evaluation_protocol'] = EVALUATION_PROTOCOL
        config['runtime_ratio_protocol'] = 'mean-total-process-time-1'
        if type(config['failure_experience']) is not bool:
            raise ValueError('failure_experience must be a boolean')
        if config['failure_experience']:
            config['failure_experience_protocol'] = 'parent-last-eight-1'
        para = config.get('para')
        if para is not None and (type(para) is not int or para != 4):
            raise ValueError('para must be None or 4')
        if para == 4:
            if config['initial_valid_roots'] != 4:
                raise ValueError('para=4 requires four initial roots')
            config['parallel_protocol'] = 'four-lineages-rounds-2-runtime'
        if type(config['plan']) is not bool:
            raise ValueError('plan must be a boolean')
        if config['plan'] and config['initial_valid_roots'] != 4:
            raise ValueError('Plan requires four initial roots')
        if config['plan']:
            config['plan_protocol'] = 'four-cold-start-directions-2-separate-budget'
        for key in ('initial_valid_roots', 'max_initial_attempts', 'smoke_count'):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('promotion_margin',):
            if not math.isfinite(config[key]) or config[key] < 0:
                raise ValueError(f'Invalid {key}')
        if config.get('direction') not in ('maximize', 'minimize'):
            raise ValueError('Expected maximize or minimize')
        if not math.isfinite(config.get('time_limit', 0)) or config['time_limit'] <= 0:
            raise ValueError('A positive task time_limit is required')
        if (not train or len({c.name for c in train}) != len(train)
                or len({c.fingerprint for c in train}) != len(train)):
            raise ValueError('Training cases must be nonempty and unique')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'state.json'
        self.provider, self.evaluate, self.train, self.config = provider, evaluate, train, config
        self.sign = 1 if config['direction'] == 'maximize' else -1
        protocol = {'version': '5.2', 'config': config,
                    'inputs': {c.name: c.fingerprint for c in train},
                    'train': [c.name for c in train], 'validation': []}
        signature = digest(json.dumps(protocol, sort_keys=True))
        if self.path.exists():
            self.s = read_json(self.path)
            if self.s['signature'] != signature:
                raise ValueError('Protocol changed')
        else:
            self.s = {'signature': signature, 'records': {}, 'events': [],
                      'root_attempts': 0, 'schedule_slot': 0, 'incumbent': None,
                      'pending': None, 'repair': None, 'validation_started': False,
                      'frozen': False}
        save_json(self.root / 'protocol.json', protocol)
        self.save()

    def run(self):
        if self.s['frozen'] or self.s['validation_started'] or self.s.get('development_closed'):
            raise ValueError('Development is closed')
        if (self.root / 'STOP_AFTER_CURRENT').exists():
            return
        if self.config['plan']:
            ensure_plan(self)
        if self.config.get('para') == 4:
            return run_parallel(self)
        super().run()

    def mean(self, ident):
        return self.s['records'][ident]['full']['mean']

    def parse_response(self, kind, parent, content):
        return parse_candidate(kind, self.source(parent) if parent else None, content)

    def augment_task(self, pending, task):
        return augment_task(self, pending, task)

    def note_response(self, pending, content):
        note_response(self, pending, content)

    def record_attempt(self, pending):
        record_attempt(self, pending)

    def measure(self, ident):
        record = self.s['records'][ident]
        seeds = evaluation_seeds(record.get('is_randomized'))
        evaluations = record.setdefault('evaluations', [])
        if 'full' in record:
            self.record_runtime(record, record['full'])
            return record['full']['valid']
        if 'smoke' not in record:
            record['smoke'] = self.evaluate(self.source(ident),
                self.train[:self.config['smoke_count']], self.config['time_limit'], 'smoke', seed=seeds[0])
            self.save()
        if not record['smoke']['valid']:
            self.record_runtime(record, record['smoke'])
            return False
        if not evaluations:
            # Reuse precheck rows so each case is evaluated only once for the first seed.
            remaining = self.train[self.config['smoke_count']:]
            rows = list(record['smoke']['cases'])
            if remaining:
                rest = self.evaluate(self.source(ident), remaining, self.config['time_limit'],
                                     'train-r0', seed=seeds[0])
                rows.extend(rest['cases'])
            evaluations.append({**from_rows(rows), 'seed': seeds[0]})
            self.save()
        while len(evaluations) < len(seeds) and all(r['valid'] for r in evaluations):
            i = len(evaluations)
            evaluations.append(self.evaluate(self.source(ident), self.train,
                self.config['time_limit'], f'train-r{i}', seed=seeds[i]))
            self.save()
        record['full'] = aggregate_evaluations(evaluations)
        self.record_runtime(record, record['full'])
        return record['full']['valid']

    def record_runtime(self, record, result):
        # full.cases already averages each case over its actual algorithm seeds.
        # Failed prechecks describe only the cases that were actually executed.
        record['runtime_ratio'] = statistics.mean(r['seconds'] for r in result['cases']) / self.config['time_limit']
        record['runtime_ratio_basis'] = 'total_process_time'
        self.save()

    def timing(self, record):
        elapsed = statistics.mean(r['seconds'] for r in record['full']['cases'])
        limit = self.config['time_limit']
        return {'allocated_seconds': limit, 'mean_process_seconds': elapsed,
                'runtime_ratio': record.get('runtime_ratio', elapsed / limit),
                'runtime_ratio_basis': 'total_process_time',
                'utilization': elapsed / limit, 'underutilized': elapsed < limit / 2,
                'instruction': 'Seek useful score-improving search, never padding or sleep. '
                               'Finishing early is fine when additional search is not useful.'}

    def feedback(self, ident):
        record = self.s['records'][ident]
        result = record.get('full', record['smoke'])
        failed = next((r for r in record.get('evaluations', []) if not r['valid']), None)
        if failed:
            result = failed
        by_name = {c.name: c for c in self.train}
        rows = sorted(result['cases'], key=lambda r: (r['status'] == 'AC', r['score']))[:2]
        return {'train_mean': result['raw_mean'], 'direction': self.config['direction'],
                'is_randomized': record['is_randomized'],
                'evaluation_count': len(record.get('evaluations', [])),
                'failed_evaluation': failed is not None,
                'examples': [{'name': r['name'], 'score': r.get('sample_raw_score', r['raw_score']),
                              'mean_score': r['raw_score'], 'seed': r.get('seed', 0),
                              'status': r['status'], 'input': by_name[r['name']].text,
                              'output': r.get('stdout', ''),
                              'message': r.get('message', '')[-6000:]} for r in rows],
                'timing': self.timing(record) if record.get('full', {}).get('valid') else None,
                'target': 'Improve mean raw score in the stated direction; all cases must be valid.'}

    def task(self):
        records, events = self.s['records'], self.s['events']
        archive = pool(records)
        if self.s['repair']:
            parent, self.s['repair'] = self.s['repair'], None
            return {'arm': 'repair', 'parent': parent,
                    'instruction': 'Repair the demonstrated error including seed-specific failure. '
                                   'Keep intended search quality. Return the full corrected program.'}
        roots = {r['lineage'] for r in records.values() if valid_for_archive(r)}
        if not archive or (len(roots) < self.config['initial_valid_roots']
                           and (self.config['plan'] or self.s['root_attempts'] < self.config['max_initial_attempts'])):
            if not self.config['plan'] and self.s['root_attempts'] >= self.config['max_initial_attempts'] and not archive:
                raise RuntimeError('No legal cold start within attempt cap')
            self.s['root_attempts'] += 1
            task = {'arm': 'independent', 'parent': None,
                    'instruction': 'Invent a complete optimizer from the specification without seed code. '
                                   'Design construction and useful search; prioritize algorithm quality.'}
            if self.config['plan']:
                completed = {r.get('initial_direction') for r in records.values() if valid_for_archive(r)}
                remaining = [i for i in range(4) if i not in completed]
                # Try all four once before retrying directions that failed; repairs keep their lineage.
                attempts = [sum(e.get('initial_direction') == i and e['arm'] == 'independent'
                                for e in events) for i in range(4)]
                index = min(remaining, key=lambda i: (attempts[i], i))
                task['initial_direction'] = index
                task['instruction'] += ' Suggested design direction: ' + self.s['plan']['directions'][index]
            return task
        slot = self.s['schedule_slot']
        self.s['schedule_slot'] += 1
        arm = CYCLE[slot % len(CYCLE)]
        if arm == 'crossover' and any(e['arm'] == arm for e in events[-7:]):
            arm = 'redesign'
        if arm not in ('independent', 'redesign'):
            recent = [e for e in events[-16:] if e['arm'] == arm][-3:]
            if len(recent) == 3 and all(e.get('parent_gain', 0) <= self.config['promotion_margin']
                                        for e in recent):
                arm = 'redesign'
        if arm == 'independent':
            return {'arm': arm, 'parent': None,
                    'instruction': 'Invent a fresh optimizer from the specification only. '
                                   'Explore beyond a single greedy local optimum.'}
        parent = archive[0] if slot % 4 != 3 else archive[(slot // 4) % len(archive)]
        recent_runtime = [i for i, e in enumerate(events)
                          if e['arm'] == 'runtime' and e.get('parent') == parent]
        if self.timing(records[parent])['underutilized'] and (
                not recent_runtime or len(events) - recent_runtime[-1] >= 8):
            arm = 'runtime'
        directions = {
            'local': 'Address one measured weakness. Up to six coordinated edits across helpers '
                     'and call sites are allowed. Preserve unrelated working mechanisms.',
            'redesign': 'Escape representation and reachable-state limitations. Make a substantive '
                        'structural redesign retaining useful parent mechanisms. Return a complete program.',
            'runtime': 'Diagnose early termination and reachable-state limits. Design score-improving '
                       'continuation, not padding, sleeps or repeated identical work. Finishing early '
                       'is fine if no useful search exists; explain in comments. Return a complete program.',
            'crossover': 'Integrate a complementary mechanism from the donor. Related parts may change '
                         'together. Do not switch whole solvers, average outputs, or route known cases.'}
        task = {'arm': arm, 'parent': parent, 'instruction': directions[arm]}
        if arm == 'crossover':
            others = [k for k in archive if records[k]['lineage'] != records[parent]['lineage']]
            if others:
                scores = {r['name']: r['score'] for r in records[parent]['full']['cases']}
                donor = max(others, key=lambda k: sum(max(0, r['score'] - scores[r['name']])
                                                     for r in records[k]['full']['cases']))
                if any(r['score'] > scores[r['name']] for r in records[donor]['full']['cases']):
                    task['donor'] = donor
            if 'donor' not in task:
                task.update(arm='redesign', instruction=directions['redesign'])
        return task

    def qualify(self, ident, pending):
        record = self.s['records'][ident]
        if self.config['plan']:
            index = pending.get('initial_direction')
            if index is None and pending.get('parent'):
                index = self.s['records'][pending['parent']].get('initial_direction')
            if index is not None:
                record['initial_direction'] = index
        if not self.measure(ident):
            pending['status'] = 'invalid_candidate'
            if pending['arm'] != 'repair':
                self.s['repair'] = ident
            return
        incumbent = self.s['incumbent']
        gain = None if incumbent is None else self.mean(ident) - self.mean(incumbent)
        pending['selection'] = {'mean_gain': gain,
                                'evaluation_count': len(record['evaluations'])}
        if incumbent is None or gain > self.config['promotion_margin']:
            self.s['incumbent'] = ident
            pending['status'] = 'incumbent_promoted'
        else:
            pending['status'] = 'archived'

    def finalize(self):
        if self.config.get('para') == 4:
            batch = self.s.get('parallel_batch')
            if batch:
                ledger_path = self.root / 'llm/ledger.json'
                requests = self.provider.requests if self.provider else (
                    len(read_json(ledger_path)) if ledger_path.exists() else 0)
                if not batch.get('budget_exhausted') or any(
                        t['implementation_call'] < requests for t in batch['tasks']):
                    raise ValueError('Pending parallel round')
            roots = {r['lineage'] for r in self.s['records'].values() if valid_for_archive(r)}
            if len(roots) != 4:
                raise ValueError('Parallel initialization incomplete: need four valid lineages')
            if batch:
                self.s['unsubmitted_batch'] = batch
                self.s['parallel_batch'] = None
        if self.config['plan']:
            completed = {r.get('initial_direction') for r in self.s['records'].values()
                         if valid_for_archive(r)}
            if not set(range(4)).issubset(completed):
                raise ValueError('Plan initialization incomplete: need four valid initial directions')
        if self.s['pending']:
            if self.s['pending'].get('status') != 'budget_exhausted':
                raise ValueError('Pending experiment')
            self.s['events'].append(dict(self.s['pending']))
            self.s['pending'] = None
        winner = self.s.get('champion') if self.s['frozen'] else self.s['incumbent']
        if not winner:
            raise ValueError('No valid evaluated incumbent')
        record = self.s['records'][winner]
        if (len(record.get('evaluations', [])) != len(evaluation_seeds(record.get('is_randomized')))
                or not valid_for_archive(record)):
            raise ValueError('Champion must have all required valid evaluations')
        code = self.source(winner)
        self.s.update(frozen=True, development_closed=True, champion=winner)
        self.save()
        (self.root / 'best.cpp').write_text(code, encoding='utf-8')
        if not (self.root / 'report.json').exists():
            ledger = read_json(self.root / 'llm/ledger.json') if (self.root / 'llm/ledger.json').exists() else []
            plan_path = self.root / 'planning/llm/ledger.json'
            plan_ledger = read_json(plan_path) if plan_path.exists() else []
            solve_requests = self.provider.requests if self.provider else len(ledger)
            save_json(self.root / 'report.json', {
                'version': '5.2', 'champion': winner, 'direction': self.config['direction'],
                'train_mean': self.sign * self.mean(winner), 'train_count': len(self.train),
                'requests': solve_requests + len(plan_ledger),
                'solve_requests': solve_requests, 'plan_requests': len(plan_ledger),
                'para': self.config.get('para'),
                'failure_experience': self.config['failure_experience'],
                'requested_calls': self.config.get('calls', self.s.get('requested_calls')),
                'effective_calls': self.s.get('effective_calls', self.config.get('calls')),
                'plan_enabled': self.config['plan'], 'plan_call': self.s.get('plan_call'),
                'is_randomized': record['is_randomized'], 'evaluation_count': len(record['evaluations']),
                'private_evaluated': False, 'selection': 'mean score; one deterministic or three seeded evaluations'})
            if self.s.get('migration'):
                report = read_json(self.root / 'report.json')
                report['migration'] = self.s['migration']
                save_json(self.root / 'report.json', report)
        return winner
