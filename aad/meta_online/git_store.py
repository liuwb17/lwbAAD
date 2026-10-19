"""One preserved worktree per policy revision; never checkout or edit main."""
import subprocess
import os
import platform
import re
import shutil
from pathlib import Path

from . import BASELINE, COMPONENTS, COMPONENT_DIR


def git(root, *args, strip=True):
    root = Path(root).resolve()
    executable, directory = 'git', root.as_posix()
    pointer = root / '.git'
    # Windows-created worktrees contain drive-letter Git pointers. Native WSL
    # Git cannot resolve those; keep the owning Git implementation and translate
    # only filesystem arguments, without rewriting the preserved repository.
    if (os.name != 'nt' and 'microsoft' in platform.release().lower()
            and pointer.is_file()
            and re.match(r'gitdir: [A-Za-z]:[/\\]', pointer.read_text(encoding='utf-8'))):
        executable = shutil.which('git.exe')
        if not executable:
            raise RuntimeError('Windows-created policy worktree requires git.exe in WSL PATH')

        def windows_path(value):
            return subprocess.run(['wslpath', '-w', value], check=True, capture_output=True,
                                  text=True, encoding='utf-8', timeout=10).stdout.strip().replace('\\', '/')

        directory = windows_path(str(root))
        args = tuple(windows_path(str(arg)) if str(arg).startswith('/') else arg for arg in args)
    result = subprocess.run([executable, '-c', 'safe.directory=' + directory,
                             '-c', 'core.hooksPath=', '-c', 'commit.gpgsign=false',
                             '-c', 'user.name=RSI', '-c', 'user.email=rsi@localhost',
                             '-C', directory, *args], capture_output=True, text=True,
                            encoding='utf-8', errors='strict', timeout=60)
    if result.returncode:
        raise RuntimeError('Git operation failed: ' + result.stderr.strip()[:1500])
    return result.stdout.strip() if strip else result.stdout


class PolicyStore:
    def __init__(self, project, root, baseline=BASELINE):
        self.project, self.root = Path(project).resolve(), Path(root).resolve()
        self.baseline = git(self.project, 'rev-parse', baseline + '^{commit}')

    def path(self, index):
        return self.root / f'{index:05d}'

    def sources(self, revision):
        return {name + '.py': git(self.project, 'show', f'{revision}:{COMPONENT_DIR}/{name}.py', strip=False)
                for name in COMPONENTS}

    def verify(self, index, revision):
        path = self.path(index)
        if git(path, 'rev-parse', 'HEAD') != revision or git(path, 'status', '--porcelain', '--untracked-files=all'):
            raise ValueError('Policy worktree changed outside the recorded update')
        return self.sources(revision)

    def create(self, index, parent, files=None):
        """Replay-safe creation; after interruption only the exact saved proposal is allowed."""
        files = files or {}
        allowed = {name + '.py' for name in COMPONENTS}
        if not set(files) <= allowed:
            raise ValueError('Only the six component files may be changed')
        expected = {**self.sources(parent), **files}
        path = self.path(index)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            git(self.project, 'worktree', 'add', '--detach', str(path), parent)
        head = git(path, 'rev-parse', 'HEAD')
        if head != parent:
            if git(path, 'rev-parse', 'HEAD^') != parent or self.sources(head) != expected:
                raise ValueError('Interrupted policy commit differs from the saved proposal')
            self.verify(index, head)
            return head
        changed = git(path, 'diff', '--name-only', 'HEAD').splitlines()
        if any(p not in {f'{COMPONENT_DIR}/{name}' for name in files} for p in changed):
            raise ValueError('Unexpected edits in policy worktree')
        if git(path, 'ls-files', '--others', '--exclude-standard'):
            raise ValueError('Unexpected untracked policy files')
        for name, source in files.items():
            target = path / COMPONENT_DIR / name
            old = self.sources(parent)[name]
            current = target.read_text(encoding='utf-8')
            if current not in (old, source):
                raise ValueError('Policy source differs from saved proposal')
            target.write_text(source, encoding='utf-8', newline='\n')
        if files and git(path, 'diff', '--name-only', 'HEAD'):
            git(path, 'add', '--', *(f'{COMPONENT_DIR}/{name}' for name in files))
            git(path, 'commit', '-m', f'Online policy update {index}')
        revision = git(path, 'rev-parse', 'HEAD')
        self.verify(index, revision)
        return revision
