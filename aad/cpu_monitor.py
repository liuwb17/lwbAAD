"""One persistent Docker/Windows sampler per Linux evaluation host."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .container_execution import EvaluationInfrastructureError

MAX_AGE = 2.5
BASE = Path(tempfile.gettempdir()) / 'aad-v52-monitor-v2'


def identity(policy):
    return hashlib.sha256(json.dumps({k: policy[k] for k in
        ('core_groups', 'service_cpus', 'windows_host_guard')}, sort_keys=True).encode()).hexdigest()


def validate_sample(policy, sample, now=None):
    import math
    now = time.time() if now is None else now
    if sample.get('policy_sha256') != identity(policy) or sample.get('error'):
        raise EvaluationInfrastructureError('CPU monitor configuration/error: ' + str(sample.get('error', 'mismatch')))
    if not 0 <= now - sample.get('sampled_unix', 0) <= MAX_AGE:
        raise EvaluationInfrastructureError('CPU monitor sample expired; no task admitted')
    for key in ('core_loads', 'windows_core_loads'):
        loads = sample.get(key)
        if key == 'windows_core_loads' and not policy['windows_host_guard'] and loads is None:
            continue
        if (not isinstance(loads, list) or len(loads) != len(policy['core_groups'])
                or any(type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 100 for n in loads)):
            raise EvaluationInfrastructureError('Invalid shared CPU load data')
    return sample


def read_sample(policy):
    path = BASE / 'sample.json'
    try:
        sample = json.loads(path.read_text())
    except (FileNotFoundError, ValueError) as error:
        raise EvaluationInfrastructureError('CPU monitor unavailable; start scripts/cpu_monitor.py') from error
    return validate_sample(policy, sample)


def serve(policy):
    import fcntl
    import docker
    from ale_bench.constants import RUST_TOOL_DOCKER_IMAGE
    from .idle_cpu_pool import WINDOWS_PROBE, core_loads, parse_cpu_times
    BASE.mkdir(parents=True, exist_ok=True)
    lock = (BASE / 'monitor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    session = uuid.uuid4().hex
    shared = {}
    mutex = threading.Lock()
    stopped = threading.Event()
    client = docker.from_env(timeout=30)
    sampler = None
    host = None

    def record(key, value):
        with mutex:
            shared[key] = (value, time.time())

    def fail(error):
        with mutex:
            shared['error'] = str(error)

    def linux_reader(container):
        previous = None
        buffer = ''
        try:
            for chunk in container.logs(stream=True, follow=True):
                buffer += chunk.decode('utf-8')
                while 'AAD_CPU_SAMPLE\n' in buffer:
                    counters, buffer = buffer.split('AAD_CPU_SAMPLE\n', 1)
                    current = parse_cpu_times(counters)
                    if previous is not None:
                        record('core_loads', core_loads(previous, current, policy['core_groups']))
                    previous = current
                if stopped.is_set():
                    return
            raise RuntimeError('Linux monitor stream ended')
        except Exception as error:
            fail(error)

    def windows_reader(process):
        try:
            for line in process.stdout:
                record('windows_core_loads', json.loads(line)['core_loads'])
            raise RuntimeError('Windows monitor stream ended')
        except Exception as error:
            fail(error)

    def publish(value):
        temporary = BASE / (session + '.tmp')
        temporary.write_text(json.dumps(value), encoding='utf-8')
        temporary.replace(BASE / 'sample.json')

    try:
        publish({'policy_sha256': identity(policy), 'error': 'monitor starting'})
        sampler = client.containers.run(RUST_TOOL_DOCKER_IMAGE,
            ['sh', '-c', 'while true; do cat /proc/stat; echo AAD_CPU_SAMPLE; sleep 1; done'],
            entrypoint=[], detach=True, network_disabled=True,
            cpuset_cpus=','.join(map(str, policy['service_cpus'])),
            mem_limit='64m', pids_limit=16, labels={'aad.v52.role': 'cpu-monitor', 'aad.v52.session': session})
        threading.Thread(target=linux_reader, args=(sampler,), daemon=True).start()
        if policy['windows_host_guard']:
            path = subprocess.run(['wslpath', '-w', str(WINDOWS_PROBE)], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
            errors = (BASE / 'windows.stderr.log').open('a')
            host = subprocess.Popen(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', path, '-Continuous'],
                                    stdout=subprocess.PIPE, stderr=errors, text=True,
                                    encoding='utf-8', errors='replace')
            threading.Thread(target=windows_reader, args=(host,), daemon=True).start()
        (BASE / 'owner.json').write_text(json.dumps({'pid': os.getpid(), 'session': session,
            'container': sampler.id, 'policy_sha256': identity(policy)}))
        while True:
            with mutex:
                values = dict(shared)
            keys = ['core_loads'] + (['windows_core_loads'] if policy['windows_host_guard'] else [])
            if 'error' in values:
                publish({'policy_sha256': identity(policy), 'error': values['error']})
                raise RuntimeError(values['error'])
            if all(k in values for k in keys):
                timestamp = min(values[k][1] for k in keys)
                sample = {'policy_sha256': identity(policy), 'session': session,
                          'sampled_unix': timestamp, 'window_started_unix': timestamp - 1.25,
                          'sample_seconds': 1, 'windows_core_loads': None,
                          **{k: values[k][0] for k in keys}}
                # Validate freshness here as well: a hung stream stops admission.
                validate_sample(policy, sample)
                publish(sample)
            time.sleep(.1)
    finally:
        stopped.set()
        publish({'policy_sha256': identity(policy), 'error': 'monitor stopped'})
        if host is not None:
            host.terminate()
        if sampler is not None:
            sampler.remove(force=True)
        client.close()
        lock.close()
