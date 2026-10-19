"""Daemon-wide CPU leases and a private copy of ALE's batch runner.

Docker names provide atomic reservations across processes/WSL distributions.
A lease covers both SMT siblings; only one logical CPU executes a submission.
Stale leases are deliberately not stolen: their owner may still be executing.
"""
from contextlib import contextmanager
import inspect
from pathlib import Path
import threading
import time
from types import FunctionType
import uuid

from .io import save_json
from .container_execution import (DOCKER_API_TIMEOUT, GuardedContainer, container_timeout,
                             scoped_batch_command)


CPU_PROTOCOL = '9950x3d-smt-leases-1'
LEASE_LABEL = 'aad.v51.cpu.lease'
ROLE_LABEL = 'aad.v51.cpu.role'
TOPOLOGY_COMMAND = (
    'cat /sys/devices/system/cpu/online; '
    'for d in /sys/devices/system/cpu/cpu[0-9]*; do '
    'printf "%s " "${d##*cpu}"; cat "$d/topology/thread_siblings_list"; done'
)


def cpu_list(text):
    result = set()
    for part in text.strip().split(','):
        ends = part.split('-')
        start, stop = int(ends[0]), int(ends[-1])
        if len(ends) > 2 or start < 0 or stop < start:
            raise ValueError('Invalid CPU list')
        result.update(range(start, stop + 1))
    return sorted(result)


def topology_policy(output):
    lines = output.strip().splitlines()
    online = cpu_list(lines[0])
    siblings = {}
    for line in lines[1:]:
        cpu, group = line.split()
        siblings[int(cpu)] = cpu_list(group)
    if online != list(range(32)) or set(siblings) != set(online):
        raise ValueError('Expected this 9950X3D host: 32 online logical CPUs')
    groups = sorted({tuple(group) for group in siblings.values()})
    if (len(groups) != 16 or any(len(g) != 2 for g in groups)
            or any(cpu not in group or any(siblings.get(s) != group for s in group)
                   for cpu, group in siblings.items())):
        raise ValueError('Expected 16 consistent two-thread core groups')
    # Reserve two complete core groups. Never assume a hidden CCD/cache mapping.
    return {'protocol': CPU_PROTOCOL, 'core_groups': [list(g) for g in groups],
            'service_cpus': sorted(cpu for group in groups[:2] for cpu in group),
            'evaluation_groups': [list(g) for g in groups[2:]],
            'scope': 'Docker-visible SMT groups; no Windows host or CCD isolation guarantee'}


def detect_policy(client, image):
    if client.info().get('OSType') != 'linux':
        raise ValueError('CPU leasing requires Linux Docker')
    output = client.containers.run(image, ['sh', '-c', TOPOLOGY_COMMAND],
                                   entrypoint=[], remove=True, network_disabled=True,
                                   cpuset_cpus='0', mem_limit='64m', pids_limit=32)
    return topology_policy(output.decode('utf-8'))


class CorePool:
    def __init__(self, client_factory, image, policy, audit_root):
        self.client_factory, self.image, self.policy = client_factory, image, policy
        self.audit_root = Path(audit_root)

    @property
    def capacity(self):
        return len(self.policy['evaluation_groups'])

    @contextmanager
    def lease(self):
        from docker.errors import APIError
        started = time.monotonic()
        token = uuid.uuid4().hex
        reservation = None
        with self.client_factory() as client:
            while reservation is None:
                for group in self.policy['evaluation_groups']:
                    try:
                        reservation = client.containers.create(
                            self.image, command=['true'], entrypoint=[],
                            name='aad-v51-core-' + '-'.join(map(str, group)),
                            labels={LEASE_LABEL: token, ROLE_LABEL: 'reservation'},
                            network_disabled=True, cpuset_cpus=str(group[0]),
                            mem_limit='16m', pids_limit=8)
                        break
                    except APIError as error:
                        if error.status_code != 409:
                            raise
                if reservation is None:
                    time.sleep(0.25)
            lease = {'token': token, 'cpu': group[0], 'siblings': list(group),
                     'reservation_id': reservation.id,
                     'wait_seconds': time.monotonic() - started, 'state': 'acquired'}
            audit = self.audit_root / f'{token}.json'
            try:
                save_json(audit, lease)
                yield lease
            finally:
                # Fail closed if cleanup cannot be verified: leave the reservation.
                # Includes partially created/started containers on SDK failures.
                for container in client.containers.list(all=True, filters={
                        'label': [f'{LEASE_LABEL}={token}', f'{ROLE_LABEL}=submission']}):
                    container.remove(force=True)
                reservation.remove(force=True)
                lease['state'] = 'released'
                save_json(audit, lease)


class PinnedContainers:
    def __init__(self, containers, local, policy):
        self.containers, self.local, self.policy = containers, local, policy

    def run(self, *args, **kwargs):
        lease = getattr(self.local, 'lease', None)
        kwargs['cpuset_cpus'] = str(lease['cpu']) if lease else ','.join(map(str, self.policy['service_cpus']))
        if lease:
            kwargs['labels'] = {**kwargs.get('labels', {}), LEASE_LABEL: lease['token'], ROLE_LABEL: 'submission'}
        else:
            kwargs['labels'] = {**kwargs.get('labels', {}), ROLE_LABEL: 'service'}
        container = self.containers.run(*args, **kwargs)
        if lease and getattr(self.local, 'deadline', None) is not None:
            return GuardedContainer(container, self.local.deadline, self.local.failure_path)
        return container


class PinnedClient:
    def __init__(self, client, local, policy):
        self.containers = PinnedContainers(client.containers, local, policy)


def isolated_runner(module, pool):
    """Reuse official code without patching its files or process-global module.

    All module-defined functions share a copied globals dict, so the official
    timer/scorer sees our Docker factory. Imported dependencies remain unchanged.
    The lease is acquired OUTSIDE run_batch_run_container's wall-clock timer.
    """
    namespace = dict(vars(module))
    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            copied = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            copied.__kwdefaults__ = value.__kwdefaults__
            copied.__annotations__ = value.__annotations__
            namespace[name] = copied
    local = threading.local()

    @contextmanager
    def docker_client():
        with pool.client_factory() as client:
            yield PinnedClient(client, local, pool.policy)

    original = namespace['run_batch_run_container']
    signature = inspect.signature(original)
    namespace['build_batch_run_command'] = scoped_batch_command(namespace['build_batch_run_command'])

    def submission(*args, **kwargs):
        with pool.lease() as lease:
            local.lease = lease
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            if 'time_limit' in arguments.arguments:
                local.deadline = time.monotonic() + container_timeout(arguments.arguments['time_limit'])
                local.failure_path = pool.audit_root / (lease['token'] + '.failure.json')
            try:
                return original(*args, **kwargs)
            finally:
                del local.lease
                if hasattr(local, 'deadline'):
                    del local.deadline
                    del local.failure_path

    namespace['docker_client'] = docker_client
    namespace['run_batch_run_container'] = submission

    def run_cases(**kwargs):
        if kwargs.get('reuse_containers') or kwargs['problem_type'].value != 'batch':
            raise ValueError('CPU leases support non-reusable batch evaluation only')
        kwargs['num_workers'] = min(pool.capacity, len(kwargs['inputs']))
        return namespace['run_cases'](**kwargs)

    return run_cases


def make_runner(policy, audit_root):
    from ale_bench import constants
    from ale_bench.tool_wrappers import case_runner
    from ale_bench.utils import docker_client
    @contextmanager
    def bounded_client():
        with docker_client() as client:
            # Bound daemon RPCs, including failed forced cleanup; keep the lease
            # if cleanup cannot be confirmed. container.wait supplies its own limit.
            client.api.timeout = DOCKER_API_TIMEOUT
            yield client

    pool = CorePool(bounded_client, constants.RUST_TOOL_DOCKER_IMAGE, policy, audit_root)
    return isolated_runner(case_runner, pool)
