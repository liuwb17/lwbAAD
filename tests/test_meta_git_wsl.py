"""Path routing for Windows-owned policy worktrees evaluated from WSL."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aad.meta_online.git_store import git


class WslGitTests(unittest.TestCase):
    def test_windows_pointer_uses_windows_git_and_translates_worktree_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '.git').write_text('gitdir: C:/repo/.git/worktrees/online\n', encoding='utf-8')
            with patch('aad.meta_online.git_store.os', SimpleNamespace(name='posix')), \
                 patch('aad.meta_online.git_store.platform.release', return_value='microsoft-standard-WSL2'), \
                 patch('aad.meta_online.git_store.shutil.which', return_value='/mnt/c/Git/git.exe'), \
                 patch('aad.meta_online.git_store.subprocess.run', side_effect=[
                     SimpleNamespace(stdout='C:\\repo\\online\n'),
                     SimpleNamespace(stdout='C:\\repo\\runs\\00000\n'),
                     SimpleNamespace(returncode=0, stdout='created\n', stderr='')]) as run:
                self.assertEqual(git(root, 'worktree', 'add', '--detach',
                                     '/mnt/c/repo/runs/00000', 'baseline'), 'created')
            command = run.call_args.args[0]
            self.assertEqual(command[0], '/mnt/c/Git/git.exe')
            self.assertIn('safe.directory=C:/repo/online', command)
            self.assertIn('C:/repo/runs/00000', command)

    def test_native_repository_keeps_native_git(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('aad.meta_online.git_store.subprocess.run',
                       return_value=SimpleNamespace(returncode=0, stdout='abc\n', stderr='')) as run:
                self.assertEqual(git(folder, 'rev-parse', 'HEAD'), 'abc')
            self.assertEqual(run.call_args.args[0][0], 'git')


if __name__ == '__main__':
    unittest.main()
