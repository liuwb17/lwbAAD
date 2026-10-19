"""Online DeepSeek strategy updates, with immutable v5.2 starting policies."""

from pathlib import Path
import subprocess

VERSION = '5.2.r.0.online'
# Resolve the release tag instead of embedding a commit from the preparation repo.
# This allows the release itself to contain exactly one initial commit.
_project = Path(__file__).resolve().parents[2].as_posix()
BASELINE = subprocess.check_output(
    ['git', '-c', 'safe.directory=' + _project, '-C', _project,
     'rev-parse', '--verify', 'refs/tags/rsi-source-baseline^{commit}'],
    text=True, encoding='utf-8').strip()
COMPONENTS = ('parallelism', 'parents', 'operators', 'operator_selection', 'memory', 'population')
COMPONENT_DIR = 'aad/search_policy/components'
