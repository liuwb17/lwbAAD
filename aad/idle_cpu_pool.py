"""Realtime idle-core admission shared by public and private ALE evaluation.

Sample one second, require >3 idle physical cores, then reserve ONE core.
An atomic Docker admission lock prevents different controllers spending the
same headroom. Queueing/sampling happens outside the official solver timer.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import threading
import uuid

from .io import save_json
from .cpu_pool import CorePool, LEASE_LABEL, ROLE_LABEL, isolated_runner
from .container_execution import DOCKER_API_TIMEOUT, EvaluationInfrastructureError


CPU_PROTOCOL = 'realtime-idle-physical-cores-2-shared-monitor'
WINDOW_SECONDS = 1
IDLE_PERCENT = 10
HEADROOM = 3
WINDOWS_PROBE = Path(__file__).with_name('assets') / 'windows_cpu_idle.ps1'


def windows_host():
    return os.name == 'nt' or 'microsoft' in platform.release().lower()


def idle_policy(original):
    return {**original, 'protocol': CPU_PROTOCOL, 'topology_protocol': original['protocol'],
            'sample_seconds': WINDOW_SECONDS, 'idle_percent': IDLE_PERCENT,
            'idle_core_headroom': HEADROOM, 'physical_core_load': 'max-smt-sibling',
            'windows_host_guard': windows_host(),
            'scope': 'Realtime Linux evaluation cores plus Windows host idle-core count when on Windows/WSL'}


def parse_cpu_times(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            # guest/guest_nice already belong to user/nice; never count twice.
            values = [int(n) for n in fields[1:9]]
            if len(values) != 8 or any(n < 0 for n in values):
                raise ValueError('Incomplete CPU counters')
            result[int(fields[0][3:])] = values
    if not result:
        raise ValueError('No per-CPU counters')
    return result


def core_loads(before, after, groups):
    loads = []
    for group in groups:
        siblings = []
        for cpu in group:
            delta = [b - a for a, b in zip(before[cpu], after[cpu])]
            # iowait can decrease; treat that observation conservatively as busy.
            total = sum(delta)
            if total <= 0 or any(d < 0 for d in delta):
                siblings.append(100.0)
            else:
                # Only actual idle time is free; iowait/steal are conservative busy.
                siblings.append(100 * (total - delta[3]) / total)
        loads.append(max(siblings))  # One busy SMT sibling makes the core busy.
    return loads


def sample_linux(client, image, policy):
    output = client.containers.run(
        image, ['sh', '-c', 'cat /proc/stat; echo AAD_CPU_SAMPLE; sleep 1; cat /proc/stat'],
        entrypoint=[], remove=True, network_disabled=True,
        cpuset_cpus=','.join(map(str, policy['service_cpus'])), mem_limit='64m', pids_limit=16)
    before, after = output.decode('utf-8').split('AAD_CPU_SAMPLE\n')
    return core_loads(parse_cpu_times(before), parse_cpu_times(after), policy['core_groups'])


def sample_windows():
    path = str(WINDOWS_PROBE.resolve())
    if os.name != 'nt':
        path = subprocess.run(['wslpath', '-w', path], check=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', path],
                                check=True, capture_output=True, encoding='utf-8', errors='replace',
                                timeout=30, **kwargs)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or '').strip()
        raise EvaluationInfrastructureError(
            f'Windows CPU probe failed (exit {error.returncode}): {detail}') from error
    loads = json.loads(result.stdout)['core_loads']
    if not loads or any(not isinstance(n, (int, float)) or not math.isfinite(n) or not 0 <= n <= 100
                        for n in loads):
        raise ValueError('Invalid Windows physical-core load sample')
    return loads


def sample_load(client, image, policy):
    if policy.get('protocol') == CPU_PROTOCOL:
        from .cpu_monitor import read_sample
        return read_sample(policy)
    try:
        if policy['windows_host_guard']:
            with ThreadPoolExecutor(max_workers=2) as executor:
                host = executor.submit(sample_windows)
                loads = sample_linux(client, image, policy)
                host_loads = host.result()
            if len(host_loads) != len(policy['core_groups']):
                raise ValueError('Windows/Docker physical-core counts differ')
        else:
            loads, host_loads = sample_linux(client, image, policy), None
        return {'core_loads': loads, 'windows_core_loads': host_loads,
                'sample_seconds': WINDOW_SECONDS, 'sampled_unix': time.time()}
    except Exception as error:
        raise EvaluationInfrastructureError('Cannot measure realtime CPU load; no task admitted') from error


def available_groups(policy, sample, leased):
    loads = sample['core_loads']
    if (len(loads) != len(policy['core_groups'])
            or any(not math.isfinite(n) or not 0 <= n <= 100 for n in loads)):
        raise EvaluationInfrastructureError('Invalid CPU load sample')
    idle = [tuple(g) for g, load in zip(policy['core_groups'], loads)
            if load <= IDLE_PERCENT and tuple(g) not in leased]
    host = sample.get('windows_core_loads')
    if policy.get('windows_host_guard') and host is None:
        raise EvaluationInfrastructureError('Missing Windows host sample')
    if host is not None and (len(host) != len(loads) or any(
            not math.isfinite(n) or not 0 <= n <= 100 for n in host)):
        raise EvaluationInfrastructureError('Invalid Windows host sample')
    if len(idle) <= HEADROOM or (host is not None and
            sum(n <= IDLE_PERCENT for n in host) - sample.get('host_pending_reservations', 0) <= HEADROOM):
        return []
    eligible = {tuple(g) for g in policy['evaluation_groups']}
    return [g for g in idle if g in eligible]


class IdleCorePool(CorePool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One admission contender per evaluator, not one per pending case.
        # Release immediately after reservation; running cases remain concurrent.
        self._admission_gate = threading.Lock()

    @contextmanager
    def admission_lock(self, client):
        if self.policy.get('protocol') == CPU_PROTOCOL:
            import fcntl
            from .cpu_monitor import BASE
            BASE.mkdir(parents=True, exist_ok=True)
            with (BASE / 'admission.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            return
        from docker.errors import APIError
        while True:
            try:
                lock = client.containers.create(
                    self.image, command=['true'], entrypoint=[], name='aad-v52-cpu-admission',
                    labels={ROLE_LABEL: 'admission'}, network_disabled=True, mem_limit='16m', pids_limit=8)
                break
            except APIError as error:
                if error.status_code != 409:
                    raise
                time.sleep(.25)
        try:
            yield
        finally:
            lock.remove(force=True)

    @contextmanager
    def lease(self):
        from docker.errors import APIError
        started, token = time.monotonic(), uuid.uuid4().hex
        reservation = None
        with self.client_factory() as client:
            attempts, lock_wait, docker_list_time = 0, 0.0, 0.0
            with self._admission_gate:
                local_wait = time.monotonic() - started
                while reservation is None:
                    attempts += 1
                    lock_started = time.monotonic()
                    with self.admission_lock(client):
                        lock_wait += time.monotonic() - lock_started
                        sample = sample_load(client, self.image, self.policy)
                        leased = set()
                        pending_host = 0
                        # Sparse listings already contain names and labels. Avoid SDK
                        # inspect calls racing with another worker releasing a lease.
                        listing_started = time.monotonic()
                        reservations = client.containers.list(all=True, sparse=True, filters={'label': [f'{ROLE_LABEL}=reservation']})
                        docker_list_time += time.monotonic() - listing_started
                        for container in reservations:
                            name = container.attrs['Names'][0].lstrip('/')
                            if name.startswith('aad-v51-core-'):
                                leased.add(tuple(int(c) for c in name.removeprefix('aad-v51-core-').split('-')))
                                # Shared samples predate new reservations. Charge those
                                # immediately so concurrent controllers cannot spend
                                # the same Windows headroom repeatedly.
                                acquired = container.attrs.get('Labels', {}).get('aad.v52.cpu.acquired_unix')
                                if acquired is None or float(acquired) >= sample.get('window_started_unix', 0):
                                    pending_host += 1
                        if self.policy.get('protocol') == CPU_PROTOCOL:
                            from .cpu_monitor import validate_sample
                            validate_sample(self.policy, sample)
                            sample = {**sample, 'host_pending_reservations': pending_host}
                        candidates = available_groups(self.policy, sample, leased)
                        for group in candidates:
                            try:
                                reservation = client.containers.create(
                                    self.image, command=['true'], entrypoint=[],
                                    name='aad-v51-core-' + '-'.join(map(str, group)),
                                    labels={LEASE_LABEL: token, ROLE_LABEL: 'reservation',
                                            'aad.v52.cpu.acquired_unix': str(time.time())},
                                    network_disabled=True, cpuset_cpus=str(group[0]), mem_limit='16m', pids_limit=8)
                                break
                            except APIError as error:
                                if error.status_code != 409:
                                    raise
                                # A legacy controller bypassed admission: resample before proceeding.
                                break
                    if reservation is None:
                        # Shared monitor updates about once per second. Re-querying
                        # Docker at 10 Hz per case adds contention, not fresh load data.
                        time.sleep(WINDOW_SECONDS if self.policy.get('protocol') == CPU_PROTOCOL else .1)
            lease = {'token': token, 'cpu': group[0], 'siblings': list(group),
                     'reservation_id': reservation.id, 'wait_seconds': time.monotonic() - started,
                     'state': 'acquired', 'cpu_protocol': CPU_PROTOCOL, 'load_sample': sample,
                     'admission_implementation': 'single-contender-per-evaluator-1',
                     'admission_attempts': attempts, 'local_queue_wait_seconds': local_wait,
                     'global_lock_wait_seconds': lock_wait, 'docker_list_seconds': docker_list_time,
                     'unleased_idle_cores': sum(
                         load <= IDLE_PERCENT and tuple(g) not in leased
                         for g, load in zip(self.policy['core_groups'], sample['core_loads']))}
            audit = self.audit_root / f'{token}.json'
            try:
                save_json(audit, lease)
                yield lease
            finally:
                # Keep the reservation if submission cleanup cannot be confirmed.
                for container in client.containers.list(all=True, filters={
                        'label': [f'{LEASE_LABEL}={token}', f'{ROLE_LABEL}=submission']}):
                    container.remove(force=True)
                reservation.remove(force=True)
                lease['state'] = 'released'
                save_json(audit, lease)


def make_runner(policy, audit_root):
    from ale_bench import constants
    from ale_bench.tool_wrappers import case_runner
    from ale_bench.utils import docker_client

    @contextmanager
    def bounded_client():
        with docker_client() as client:
            client.api.timeout = DOCKER_API_TIMEOUT
            yield client

    pool = IdleCorePool(bounded_client, constants.RUST_TOOL_DOCKER_IMAGE, policy, audit_root)
    return isolated_runner(case_runner, pool, supported_types=('batch', 'reactive'))
