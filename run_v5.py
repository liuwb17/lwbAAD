"""Run the isolated AAD v5 command-line interface."""
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    entry = Path(__file__).resolve().parent / 'v5/scripts/search.py'
    raise SystemExit(subprocess.call([sys.executable, str(entry), *sys.argv[1:]]))
