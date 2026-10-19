from copy import deepcopy
import unittest
from unittest.mock import patch
from aad.cpu_monitor import identity, validate_sample, MAX_AGE
from aad.idle_cpu_pool import available_groups, CPU_PROTOCOL, sample_load
from aad.container_execution import EvaluationInfrastructureError


class MonitorTests(unittest.TestCase):
    def policy(self):
        return {'protocol': CPU_PROTOCOL, 'core_groups': [[i] for i in range(8)],
                'service_cpus': [0], 'evaluation_groups': [[i] for i in range(1, 8)],
                'windows_host_guard': True}

    def sample(self):
        return {'policy_sha256': identity(self.policy()), 'core_loads': [0.] * 8,
                'windows_core_loads': [0.] * 8, 'sampled_unix': 100., 'window_started_unix': 98.75}

    def test_stale_future_missing_and_bad_host_samples_fail_closed(self):
        p, s = self.policy(), self.sample()
        validate_sample(p, s, now=101)
        for update in ({'sampled_unix': 90}, {'sampled_unix': 103},
                       {'windows_core_loads': None}, {'windows_core_loads': [float('nan')] * 8},
                       {'policy_sha256': 'other'}, {'error': 'monitor exited'}):
            with self.subTest(update=update), self.assertRaises(EvaluationInfrastructureError):
                validate_sample(p, {**s, **update}, now=101)

    def test_shared_sample_does_not_repeat_host_or_docker_probes(self):
        with patch('aad.cpu_monitor.read_sample', return_value=self.sample()) as read, \
             patch('aad.idle_cpu_pool.sample_windows') as windows, patch('aad.idle_cpu_pool.sample_linux') as linux:
            for _ in range(4):
                sample_load(None, 'image', self.policy())
            self.assertEqual(read.call_count, 4)
            windows.assert_not_called()
            linux.assert_not_called()

    def test_new_reservations_cannot_spend_same_host_headroom(self):
        p, s = self.policy(), self.sample()
        s['windows_core_loads'] = [0.] * 5 + [100.] * 3
        self.assertTrue(available_groups(p, s, set()))
        s['host_pending_reservations'] = 1
        self.assertTrue(available_groups(p, s, {(1,)}))
        s['host_pending_reservations'] = 2
        self.assertFalse(available_groups(p, s, {(1,), (2,)}))


if __name__ == '__main__':
    unittest.main()
