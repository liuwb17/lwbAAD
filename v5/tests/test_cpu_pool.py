from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from aad.io import read_json
from aad.cpu_pool import (CorePool, LEASE_LABEL, ROLE_LABEL, PinnedContainers,
                         isolated_runner, topology_policy)


class APIError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class Docker:
    def __init__(self):
        self.containers = self
        self.lock = threading.Lock()
        self.items = {}
        self.waited = threading.Event()
        self.cleanup_error = False

    @contextmanager
    def client(self):
        yield self

    def create(self, image, **kwargs):
        with self.lock:
            name = kwargs['name']
            if name in self.items:
                self.waited.set()
                raise APIError(409)
            owner = self
            class Container:
                id = name
                labels = kwargs['labels']
                def remove(self, force=False):
                    with owner.lock:
                        del owner.items[name]
            self.items[name] = Container()
            return self.items[name]

    def list(self, all, filters):
        if self.cleanup_error:
            raise OSError('Cannot verify Docker cleanup')
        with self.lock:
            return [c for c in self.items.values() if all_labels(c.labels, filters['label'])]


def all_labels(labels, required):
    return all(labels.get(k) == v for k, v in (r.split('=', 1) for r in required))


def topology():
    return '0-31\n' + '\n'.join(f'{i} {i // 2 * 2}-{i // 2 * 2 + 1}' for i in range(32))


class CpuTests(unittest.TestCase):
    def test_topology_reserves_whole_cores_and_rejects_hidden_or_inconsistent_smt(self):
        policy = topology_policy(topology())
        self.assertEqual(policy['service_cpus'], [0, 1, 2, 3])
        self.assertEqual(policy['evaluation_groups'], [[i, i + 1] for i in range(4, 32, 2)])
        with self.assertRaises(ValueError):
            topology_policy(topology().replace('0-31', '0-15', 1))
        with self.assertRaises(ValueError):
            topology_policy(topology().replace('1 0-1', '1 1'))

    def test_two_pools_share_daemon_locks_wait_and_release_even_on_exception(self):
        docker = Docker()
        policy = {'evaluation_groups': [[4, 5]]}
        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', {
                'docker.errors': SimpleNamespace(APIError=APIError)}):
            first = CorePool(docker.client, 'test', policy, Path(root) / 'first')
            second = CorePool(docker.client, 'test', policy, Path(root) / 'second')
            acquired = threading.Event()

            def worker():
                with second.lease() as lease:
                    acquired.set()
                    self.assertEqual(lease['siblings'], [4, 5])
                    raise ValueError('submission failed')

            with ThreadPoolExecutor(max_workers=1) as executor:
                with first.lease() as lease:
                    self.assertEqual(lease['cpu'], 4)
                    future = executor.submit(worker)
                    self.assertTrue(docker.waited.wait(2))
                    self.assertFalse(acquired.is_set())
                with self.assertRaisesRegex(ValueError, 'submission failed'):
                    future.result(timeout=3)
            self.assertFalse(docker.items)
            audits = list(Path(root).rglob('*.json'))
            self.assertEqual(len(audits), 2)
            self.assertTrue(all(read_json(p)['state'] == 'released' for p in audits))

    def test_cleanup_failure_retains_reservation_and_does_not_reuse_core(self):
        docker = Docker()
        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', {
                'docker.errors': SimpleNamespace(APIError=APIError)}):
            pool = CorePool(docker.client, 'test', {'evaluation_groups': [[4, 5]]}, root)
            with self.assertRaises(OSError), pool.lease():
                docker.cleanup_error = True
            self.assertIn('aad-v51-core-4-5', docker.items)

    def test_pin_submission_and_service_to_disjoint_cpus(self):
        seen = []
        containers = SimpleNamespace(run=lambda *a, **kw: seen.append(kw))
        local = threading.local()
        proxy = PinnedContainers(containers, local, {'service_cpus': [0, 1, 2, 3]})
        proxy.run('image', cpu_period=100000, cpu_quota=100000)
        local.lease = {'cpu': 4, 'token': 'unique'}
        proxy.run('image', cpu_period=100000, cpu_quota=100000)
        self.assertEqual(seen[0]['cpuset_cpus'], '0,1,2,3')
        self.assertEqual(seen[1]['cpuset_cpus'], '4')
        self.assertEqual(seen[1]['labels'][LEASE_LABEL], 'unique')
        self.assertEqual(seen[1]['labels'][ROLE_LABEL], 'submission')
        self.assertEqual(seen[1]['cpu_quota'], 100000)

    def test_private_runner_waits_before_timer_and_leaves_official_module_unchanged(self):
        module = ModuleType('fake_ale_runner')
        clock = [0]
        seen = []
        module.clock, module.seen = clock, seen
        exec('''
def docker_client():
    raise AssertionError('Original client must not be used')

def build_batch_run_command(output_file='/tmp/output.txt', profiles_file='/tmp/profiles.json'):
    return 'solver; sync'
def run_batch_run_container():
    start = clock[0]
    with docker_client() as client:
        client.containers.run('solver')
    clock[0] += 1
    return clock[0] - start
def run_cases(**kwargs):
    seen.append(kwargs['num_workers'])
    with docker_client() as client:
        client.containers.run('compiler')
    return run_batch_run_container()
''', vars(module))

        class Pool:
            capacity = 14
            policy = {'service_cpus': [0, 1, 2, 3]}
            @contextmanager
            def lease(self):
                clock[0] += 100  # Queue delay, outside official measurement.
                yield {'cpu': 4, 'token': 'test'}
            @contextmanager
            def client_factory(self):
                yield SimpleNamespace(containers=SimpleNamespace(run=lambda *a, **kw: seen.append(kw)))

        original = module.run_batch_run_container
        run_cases = isolated_runner(module, Pool())
        elapsed = run_cases(inputs=['x'] * 50, problem_type=SimpleNamespace(value='batch'))
        self.assertEqual(elapsed, 1)
        self.assertEqual(clock[0], 101)
        self.assertEqual(seen[0], 14)
        self.assertEqual(seen[1]['cpuset_cpus'], '0,1,2,3')
        self.assertEqual(seen[2]['cpuset_cpus'], '4')
        self.assertIs(module.run_batch_run_container, original)
        with self.assertRaisesRegex(ValueError, 'non-reusable batch'):
            run_cases(inputs=['x'], problem_type=SimpleNamespace(value='batch'), reuse_containers=True)
