"""Shared RSI evaluation: public 0..19 by default, realtime idle-core admission."""
import json
from pathlib import Path

from .io import digest
from . import ale_batch as legacy
from .idle_cpu_pool import idle_policy, make_runner


DEFAULT_PUBLIC_SEEDS = tuple(range(20))
SUPPORTED_TYPES = ('batch', 'reactive')
REACTIVE_PROTOCOL = 'reactive-tester-file-fsync-watchdog-1'
statement_for = legacy.statement_for


def load_task(root):
    return legacy.load_task(root, supported_types=SUPPORTED_TYPES)


def prepare(archive, problem_id, destination, seeds=None):
    return legacy.prepare(archive, problem_id, destination,
                          DEFAULT_PUBLIC_SEEDS if seeds is None else seeds, supported_types=SUPPORTED_TYPES)


class AleBatchEvaluator(legacy.AleBatchEvaluator):
    """Historical name retained for callers; dispatches both batch and reactive."""
    def __init__(self, task_root, cache_root):
        self.task_root, self.cache_root = Path(task_root).resolve(), Path(cache_root)
        self.task, _ = load_task(self.task_root)
        self.runtime = legacy.runtime_identity(self.task_root)
        if self.task['problem_type'] == 'reactive':
            self.runtime['execution_protocol'] = REACTIVE_PROTOCOL
            self.runtime['problem_type'] = 'reactive'
            self.runtime['interaction'] = 'official-tester-starts-submission'
        self.runtime['cpu_policy'] = idle_policy(self.runtime['cpu_policy'])
        self.run_cases = make_runner(self.runtime['cpu_policy'], self.cache_root / 'cpu_leases')
        self.signature = digest(json.dumps({'task': self.task, 'runtime': self.runtime}, sort_keys=True))


def private_test(run, task_root, archive):
    return legacy.private_test(run, task_root, archive, evaluator_factory=AleBatchEvaluator,
                               supported_types=SUPPORTED_TYPES)


AleEvaluator = AleBatchEvaluator
