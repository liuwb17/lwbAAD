"""Run the persistent load sampler before starting evaluation controllers."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aad.cpu_monitor import serve

if __name__ == '__main__':
    serve(json.loads(Path(sys.argv[1]).read_text()))
