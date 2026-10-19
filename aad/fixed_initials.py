"""Shared fixed-bank adapter for the unchanged v5.2 search and v5.2.r.0.

Import/evaluation is common preparation, with no Plan or solver generation.
Invalid originals stop by default; explicit acceptance retains them without replacement.
"""
import importlib.util
import json
from pathlib import Path

from .io import digest, read_json, save_json
from .parallel_engine import V51Engine
from .candidate_policy import valid_for_archive


WORKSPACE = Path(__file__).resolve().parents[1]
READER_PATH = WORKSPACE / 'tools' / 'fixed_initials.py'


def read_bank(problem_id, bank=None):
    # Resolve from this checkout, not the caller's working directory/sys.path.
    spec = importlib.util.spec_from_file_location('rsi_fixed_initial_reader', READER_PATH)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    bank = Path(bank) if bank is not None else reader.DEFAULT_BANK
    initials = reader.load_initials(problem_id, bank)
    task = reader.read_json(bank / 'tasks' / problem_id / 'task.json')
    if task['problem_type'] not in ('batch', 'reactive'):
        raise ValueError('Only batch or reactive fixed initials are supported')
    plan = reader.read_json(bank / 'tasks' / problem_id / 'plan' / 'artifact.json')
    return initials, plan


class FixedInitialEngine(V51Engine):
    """v5.2 para=4 unchanged after a common, audited initialization."""
    harness_version = '5.2'

    def load_initial_bank(self, config, bank):
        return read_bank(config['problem_id'], bank)

    def __init__(self, root, provider, evaluate, train, config, *, bank=None):
        if (config.get('plan', False) is not False or type(config.get('para', 4)) is not int
                or config.get('para', 4) != 4):
            raise ValueError('Fixed-bank runner requires plan=False and para=4')
        if type(config.get('allow_invalid_initials', False)) is not bool:
            raise ValueError('allow_invalid_initials must be a boolean')
        self.initials, self.initial_plan = self.load_initial_bank(config, bank)
        ids = [item['source_sha256'] for item in self.initials]
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError('Expected four distinct fixed sources')
        selection = []
        for index, item in enumerate(self.initials):
            if item['initial_direction'] != index or digest(item['code']) != ids[index]:
                raise ValueError('Fixed initial identity/order mismatch')
            metadata = {k: v for k, v in item.items() if k not in ('code', 'source_bytes')}
            selection.append(metadata)
        config = {**config, 'para': 4, 'plan': False,
                  'harness_version': self.harness_version,
                  'allow_invalid_initials': config.get('allow_invalid_initials', False),
                  'initialization_protocol': ('fixed-bank-originals-2-retain-invalid'
                      if config.get('allow_invalid_initials', False) else 'fixed-bank-originals-1-stop-on-invalid'),
                  'initial_evaluation_cost': 'shared-preparation-reported-separately',
                  'fixed_initials': selection,
                  'fixed_plan_sha256': digest(json.dumps(self.initial_plan, sort_keys=True))}
        super().__init__(root, provider, evaluate, train, config)

    def source(self, ident):
        record = self.s['records'][ident]
        if record.get('fixed_initial'):
            code = (self.root / 'candidates' / ident / 'solution.cpp').read_bytes().decode('utf-8')
            if digest(code) != ident:
                raise ValueError('Fixed initial source changed')
            return code
        return super().source(ident)

    def initialize(self):
        """Persist slots before evaluation; resume each original without regenerating."""
        for item in self.initials:
            ident, index = item['source_sha256'], item['initial_direction']
            if ident not in self.s['records']:
                folder = self.root / 'candidates' / ident
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'solution.cpp').write_bytes(item['source_bytes'])
                self.s['records'][ident] = {
                    'parent': None, 'donor': None, 'lineage': ident,
                    'initial_slot': index, 'initial_direction': index,
                    'is_randomized': item['is_randomized'], 'hypothesis': item['hypothesis'],
                    'fixed_initial': self.config['fixed_initials'][index]}
            elif self.s['records'][ident].get('fixed_initial') != self.config['fixed_initials'][index]:
                raise ValueError('Saved fixed initial metadata changed')
            self.source(ident)
        self.s.setdefault('initialization', {'completed': [], 'status': 'evaluating'})
        save_json(self.root / 'initial_selection.json', {
            'selected_sources': self.config['fixed_initials'], 'plan': self.initial_plan,
            'evaluation_cost': self.config['initial_evaluation_cost']})
        self.save()
        state = self.s['initialization']
        for item in self.initials:
            ident = item['source_sha256']
            if ident in state['completed']:
                continue
            event = {'arm': 'fixed_initial', 'parent': None, 'candidate': ident,
                     'initial_slot': item['initial_direction'], 'tokens': 0}
            self.qualify(ident, event)
            self.s['repair'] = None  # Never auto-generate replacements for originals.
            self.s['events'].append(event)
            state['completed'].append(ident)
            self.save()
        invalid = [i['source_sha256'] for i in self.initials
                   if not valid_for_archive(self.s['records'][i['source_sha256']])]
        accepted = self.config['allow_invalid_initials'] and len(invalid) < 4
        state.update(status='invalid_originals' if invalid and not accepted else 'complete',
                     invalid=invalid, invalid_originals_accepted=bool(invalid and accepted))
        state['evaluated_case_runs'] = sum(
            len(r['full']['cases']) * len(r['evaluations']) if 'full' in r
            else len(r.get('smoke', {}).get('cases', []))
            for r in (self.s['records'][i['source_sha256']] for i in self.initials))
        state['official_process_seconds'] = sum(
            sum(row['seconds'] for result in r.get('evaluations', []) for row in result['cases'])
            if r.get('evaluations') else sum(row['seconds'] for row in r.get('smoke', {}).get('cases', []))
            for r in (self.s['records'][i['source_sha256']] for i in self.initials))
        self.save()
        if invalid and not accepted:
            raise ValueError('Invalid fixed originals; retained all slots, no replacement generation')

    def run(self):
        if self.s['frozen'] or self.s.get('development_closed') or self.s['validation_started']:
            raise ValueError('Development is closed')
        if (self.root / 'STOP_AFTER_CURRENT').exists():
            return
        self.initialize()
        return super().run()

    def finalize(self):
        if self.s.get('initialization', {}).get('status') != 'complete':
            raise ValueError('Fixed initialization incomplete')
        winner = super().finalize()
        report = read_json(self.root / 'report.json')
        report.update(harness_version=self.harness_version,
                      initialization=self.s['initialization'],
                      initial_evaluation_cost=self.config['initial_evaluation_cost'])
        save_json(self.root / 'report.json', report)
        return winner
