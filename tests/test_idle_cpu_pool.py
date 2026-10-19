"""Offline realtime-load admission tests; no Docker or host changes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import tempfile
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aad.io import read_json
from aad.cpu_pool import ROLE_LABEL
from aad.container_execution import EvaluationInfrastructureError
from aad.idle_cpu_pool import (IdleCorePool, available_groups, core_loads, idle_policy,
                         parse_cpu_times, sample_load, sample_windows)
import test_cpu_pool as fixtures


def policy(count=4):
    groups = [[2*i, 2*i+1] for i in range(count)]
    return {'protocol': 'test', 'core_groups': groups, 'evaluation_groups': groups[1:],
            'service_cpus': groups[0], 'windows_host_guard': False}


def sample(count=4):
    return {'core_loads': [0.] * count, 'windows_core_loads': None, 'sample_seconds': 1}


class Docker(fixtures.Docker):
    def list(self, all, filters, sparse=False):
        containers = super().list(all, filters)
        if sparse:
            # Match SDK sparse objects: name/labels properties are unavailable.
            return [SimpleNamespace(attrs={'Names': ['/' + c.name], 'Labels': c.labels})
                    for c in containers]
        return containers

    def create(self, image, **kwargs):
        container = super().create(image, **kwargs)
        container.name = kwargs['name']
        return container


class RealtimeCpuTests(unittest.TestCase):
    def test_queued_cases_have_one_admission_contender_but_run_concurrently(self):
        docker = Docker()
        busy = threading.Event()
        allow = threading.Event()
        launch = threading.Barrier(5)
        running = threading.Barrier(4)
        calls = []

        def load(*args):
            calls.append(threading.get_ident())
            s = sample(8)
            if not allow.is_set():
                s['core_loads'] = [100.] * 8
            return s

        def wait(_seconds):
            busy.set()
            if not allow.wait(3):
                raise AssertionError('test admission not released')

        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', {
                'docker.errors': SimpleNamespace(APIError=fixtures.APIError)}), \
             patch('aad.idle_cpu_pool.sample_load', side_effect=load), \
             patch('aad.idle_cpu_pool.time.sleep', side_effect=wait):
            pool = IdleCorePool(docker.client, 'mock', policy(8), root)
            def work():
                launch.wait(timeout=3)
                with pool.lease() as lease:
                    # If the admission gate is held through solver execution,
                    # the remaining cases cannot reach this barrier.
                    running.wait(timeout=3)
                    return lease
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(work) for _ in range(4)]
                launch.wait(timeout=3)
                try:
                    self.assertTrue(busy.wait(2))
                    self.assertFalse(pool._admission_gate.acquire(blocking=False))
                    self.assertEqual(len(set(calls)), 1)
                finally:
                    allow.set()
                leases = [f.result(timeout=5) for f in futures]
            self.assertEqual(len({tuple(v['siblings']) for v in leases}), 4)
            self.assertFalse(docker.items)
            self.assertTrue(all(v['admission_attempts'] >= 1 for v in leases))
            self.assertTrue(all(v['wait_seconds'] >= v['local_queue_wait_seconds'] for v in leases))

    def test_strictly_more_than_three_idle_physical_cores(self):
        p = policy(6)
        for count in (0, 1, 2, 3, 4, 5, 6):
            with self.subTest(idle=count):
                s = {'core_loads': [0.] * count + [100.] * (6-count)}
                choices = available_groups(p, s, set())
                self.assertEqual(bool(choices), count > 3)
                self.assertNotIn((0, 1), choices)  # Service cores count as idle but are not leased.

    def test_ten_percent_inclusive_and_reservations_excluded(self):
        p, s = policy(), sample()
        s['core_loads'][0] = 10
        self.assertTrue(available_groups(p, s, set()))
        s['core_loads'][0] = 10.000001
        self.assertFalse(available_groups(p, s, set()))
        s['core_loads'][0] = 0
        self.assertFalse(available_groups(p, s, {(2, 3)}))

    def test_windows_host_load_also_gates_dispatch(self):
        p, s = policy(8), sample(8)
        s['windows_core_loads'] = [0.] * 3 + [90.] * 5
        self.assertFalse(available_groups(p, s, set()))
        s['windows_core_loads'][3] = 10
        self.assertTrue(available_groups(p, s, set()))

    def test_cpu_deltas_ignore_guest_duplicate_and_busy_smt_sibling(self):
        before = parse_cpu_times('cpu0 0 0 0 0 0 0 0 0 0 0\ncpu1 0 0 0 0 0 0 0 0 0 0')
        after = parse_cpu_times('cpu0 10 0 0 90 0 0 0 0 10 0\ncpu1 80 0 0 20 0 0 0 0 0 0')
        self.assertEqual(core_loads(before, after, [[0], [1], [0, 1]]), [10., 80., 80.])

    def test_bad_or_reset_counters_fail_closed(self):
        with self.assertRaises(ValueError):
            parse_cpu_times('cpu0 1 2 3')
        before = {0: [1] * 8}
        self.assertEqual(core_loads(before, before, [[0]]), [100.])
        self.assertEqual(core_loads(before, {0: [0] * 8}, [[0]]), [100.])
        with self.assertRaises(EvaluationInfrastructureError):
            available_groups(policy(), {'core_loads': [float('nan')]*4}, set())

    def test_policy_records_threshold_window_and_runtime_protocol(self):
        original = fixtures.topology_policy(fixtures.topology())
        updated = idle_policy(original)
        self.assertEqual(updated['idle_core_headroom'], 3)
        self.assertEqual(updated['idle_percent'], 10)
        self.assertEqual(updated['sample_seconds'], 1)
        self.assertNotEqual(updated['protocol'], original['protocol'])
        self.assertEqual(updated['evaluation_groups'], original['evaluation_groups'])

    def test_monitor_failure_does_not_assume_idle(self):
        with patch('aad.idle_cpu_pool.sample_linux', side_effect=PermissionError('unavailable')):
            with self.assertRaises(EvaluationInfrastructureError):
                sample_load(None, 'image', policy())

    def test_windows_probe_rejects_invalid_data(self):
        with patch('aad.idle_cpu_pool.subprocess.run', return_value=SimpleNamespace(stdout='{"core_loads": [101]}')):
            with self.assertRaises(ValueError):
                sample_windows()

    def test_windows_probe_preserves_failure_and_tolerates_localized_stderr(self):
        failure = subprocess.CalledProcessError(1, ['powershell.exe'],
                                               stderr='UnauthorizedAccess: scripts disabled')
        with patch('aad.idle_cpu_pool.os.name', 'nt'), \
             patch('aad.idle_cpu_pool.subprocess.CREATE_NO_WINDOW', 0x08000000, create=True), \
             patch('aad.idle_cpu_pool.subprocess.run', side_effect=failure) as run:
            with self.assertRaisesRegex(EvaluationInfrastructureError, 'UnauthorizedAccess'):
                sample_windows()
            self.assertEqual(run.call_args.kwargs['errors'], 'replace')

    def test_second_process_waits_with_three_idle_then_refills_after_release(self):
        docker = Docker()
        saw_three = threading.Event()
        acquired = threading.Event()

        def load(*args):
            if any(c.labels[ROLE_LABEL] == 'reservation' for c in docker.items.values()):
                saw_three.set()
            return sample()

        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', {
                'docker.errors': SimpleNamespace(APIError=fixtures.APIError)}), \
             patch('aad.idle_cpu_pool.sample_load', side_effect=load):
            first = IdleCorePool(docker.client, 'mock', policy(), Path(root)/'a')
            second = IdleCorePool(docker.client, 'mock', policy(), Path(root)/'b')

            def work():
                with second.lease() as lease:
                    acquired.set()
                    self.assertEqual(lease['siblings'], [2, 3])
                    self.assertEqual(lease['unleased_idle_cores'], 4)

            with ThreadPoolExecutor(max_workers=1) as executor:
                with first.lease():
                    future = executor.submit(work)
                    self.assertTrue(saw_three.wait(2))
                    self.assertFalse(acquired.is_set())
                future.result(timeout=3)
            self.assertFalse(docker.items)
            audits = [read_json(p) for p in Path(root).rglob('*.json')]
            self.assertEqual(len(audits), 2)
            self.assertTrue(all(a['state'] == 'released' and 'load_sample' in a for a in audits))

    def test_external_busy_load_queues_even_without_framework_reservations(self):
        docker = Docker()
        busy_seen, allow, acquired = threading.Event(), threading.Event(), threading.Event()

        def load(*args):
            s = sample()
            if not allow.is_set():
                s['core_loads'][0] = 70
                busy_seen.set()
            return s

        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', {
                'docker.errors': SimpleNamespace(APIError=fixtures.APIError)}), \
             patch('aad.idle_cpu_pool.sample_load', side_effect=load):
            pool = IdleCorePool(docker.client, 'mock', policy(), root)
            def work():
                with pool.lease():
                    acquired.set()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(work)
                try:
                    self.assertTrue(busy_seen.wait(2))
                    self.assertFalse(acquired.is_set())
                finally:
                    allow.set()
                future.result(timeout=3)

    def test_admission_failures_release_lock_and_cleanup_failure_keeps_core(self):
        docker = Docker()
        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', {
                'docker.errors': SimpleNamespace(APIError=fixtures.APIError)}):
            pool = IdleCorePool(docker.client, 'mock', policy(), root)
            with patch('aad.idle_cpu_pool.sample_load', side_effect=OSError('monitor failed')):
                with self.assertRaises(OSError), pool.lease():
                    self.fail('Admission must fail')
            self.assertFalse(docker.items)
            with patch('aad.idle_cpu_pool.sample_load', return_value=sample()):
                with self.assertRaises(OSError), pool.lease():
                    docker.cleanup_error = True
            self.assertIn('aad-v51-core-2-3', docker.items)
            self.assertNotIn('aad-v52-cpu-admission', docker.items)


if __name__ == '__main__':
    unittest.main()
