"""Bound ALE batch completion without synchronizing unrelated host filesystems."""
import inspect
import math
from pathlib import Path
import shlex
import time

from .io import save_json


EXECUTION_PROTOCOL = 'batch-file-fsync-watchdog-1'
CONTAINER_GRACE_SECONDS = 30
DOCKER_API_TIMEOUT = 30


class EvaluationInfrastructureError(OSError):
    """Abort evaluation, preserving pending work; never score this as solver failure."""


def container_timeout(time_limit):
    # Match ALE's solver timeout, with separate time for container startup/teardown.
    return math.ceil(time_limit + 0.1) + 0.2 + CONTAINER_GRACE_SECONDS


def scoped_batch_command(builder):
    signature = inspect.signature(builder)

    def build(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        command = builder(*args, **kwargs)
        if not command.endswith('; sync'):
            raise EvaluationInfrastructureError('ALE batch completion command changed; review adapter')
        # fsync only these two files on the local evaluation filesystem. Keep the
        # solver's timeout, GNU time invocation, redirections and exit semantics.
        files = [arguments.arguments[name] for name in ('output_file', 'profiles_file')]
        return command[:-len('; sync')] + '; sync -- ' + ' '.join(shlex.quote(f) for f in files)

    return build


class GuardedContainer:
    def __init__(self, container, deadline, audit_path):
        self.container, self.deadline, self.audit_path = container, deadline, Path(audit_path)

    def __getattr__(self, name):
        return getattr(self.container, name)

    def wait(self, timeout=None, **kwargs):
        remaining = self.deadline - time.monotonic()
        try:
            if remaining <= 0:
                raise TimeoutError('Container startup exceeded execution watchdog')
            result = self.container.wait(timeout=min(timeout, remaining) if timeout is not None else remaining,
                                         **kwargs)
            if result.get('Error') or result.get('StatusCode') != 0:
                raise EvaluationInfrastructureError('Container completion/fsync failed: ' + str(result))
            return result
        except Exception as error:
            save_json(self.audit_path, {
                'protocol': EXECUTION_PROTOCOL, 'container_id': self.container.id,
                'error_type': type(error).__name__, 'error': str(error)[:1500],
                'classification': 'infrastructure_failure_not_candidate_score',
            })
            raise EvaluationInfrastructureError('Official evaluation container failed to complete safely') from error
