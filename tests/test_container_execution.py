from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aad.io import read_json
from aad.container_execution import (EvaluationInfrastructureError, GuardedContainer,
                               container_timeout, scoped_batch_command)


class ExecutionTests(unittest.TestCase):
    def test_only_sync_scope_changes_solver_command_and_timing_preserved(self):
        solver = 'timeout 4.2 prlimit --cpu=4.1 /usr/bin/time -o /tmp/p ./a.out < /tmp/i > /tmp/o'

        def builder(output_file='/tmp/o', profiles_file='/tmp/p'):
            return solver + '; sync'

        wrapped = scoped_batch_command(builder)
        self.assertEqual(wrapped(), solver + '; sync -- /tmp/o /tmp/p')
        self.assertEqual(wrapped('/tmp/space name', '/tmp/quote\'name'),
                         solver + '; sync -- \'/tmp/space name\' \'/tmp/quote\'"\'"\'name\'')
        self.assertEqual(builder(), solver + '; sync')

    def test_changed_sdk_command_is_rejected(self):
        def builder(output_file='o', profiles_file='p'):
            return 'solver; some_new_completion'
        with self.assertRaises(EvaluationInfrastructureError):
            scoped_batch_command(builder)()

    def test_wait_uses_remaining_deadline_and_preserves_success(self):
        wrapped = SimpleNamespace(id='test', wait=Mock(return_value={'StatusCode': 0}), attrs={'x': 1})
        with tempfile.TemporaryDirectory() as folder, patch('aad.container_execution.time.monotonic', return_value=10):
            guard = GuardedContainer(wrapped, 17, Path(folder) / 'audit.json')
            self.assertEqual(guard.wait(), {'StatusCode': 0})
            wrapped.wait.assert_called_once_with(timeout=7)
            self.assertEqual(guard.attrs, {'x': 1})
            self.assertFalse((Path(folder) / 'audit.json').exists())
            guard.wait(timeout=2)
            self.assertEqual(wrapped.wait.call_args.kwargs['timeout'], 2)

    def test_wait_timeout_aborts_as_infrastructure_and_writes_evidence(self):
        wrapped = SimpleNamespace(id='test', wait=Mock(side_effect=TimeoutError('daemon stalled')))
        with tempfile.TemporaryDirectory() as folder, patch('aad.container_execution.time.monotonic', return_value=10):
            path = Path(folder) / 'audit.json'
            guard = GuardedContainer(wrapped, 17, path)
            with self.assertRaises(EvaluationInfrastructureError) as caught:
                guard.wait()
            self.assertIsInstance(caught.exception, OSError)
            self.assertNotIsInstance(caught.exception, (ValueError, RuntimeError))
            self.assertEqual(read_json(path)['classification'], 'infrastructure_failure_not_candidate_score')

    def test_fsync_exit_or_daemon_error_does_not_become_candidate_score(self):
        for result in ({'StatusCode': 1}, {'StatusCode': 0, 'Error': {'Message': 'daemon failed'}}):
            wrapped = SimpleNamespace(id='test', wait=Mock(return_value=result))
            with tempfile.TemporaryDirectory() as folder, patch('aad.container_execution.time.monotonic', return_value=10):
                with self.assertRaises(EvaluationInfrastructureError):
                    GuardedContainer(wrapped, 17, Path(folder) / 'audit.json').wait()

    def test_startup_deadline_already_elapsed_and_official_timeout_margin(self):
        self.assertAlmostEqual(container_timeout(3), 34.2)
        wrapped = SimpleNamespace(id='test', wait=Mock())
        with tempfile.TemporaryDirectory() as folder, patch('aad.container_execution.time.monotonic', return_value=20):
            with self.assertRaises(EvaluationInfrastructureError):
                GuardedContainer(wrapped, 17, Path(folder) / 'audit.json').wait()
            wrapped.wait.assert_not_called()
