"""Small function proxies: generated source runs only in the restricted worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from . import COMPONENTS

WORKER = Path(__file__).with_name('policy_worker.py')
FUNCTIONS = {'parallelism': ('select_para',), 'parents': ('select_parents',),
             'operators': ('define_operator',), 'operator_selection': ('select_operators',),
             'memory': ('build_context', 'update_memory'), 'population': ('update_population',)}


class PolicyError(ValueError):
    pass


def invoke(sources, component, function, args, seed=None):
    payload = json.dumps({'sources': sources, 'component': component, 'function': function,
                          'args': args, 'rng_seed': seed}, ensure_ascii=False, allow_nan=False)
    # No AAD_API_KEY or other environment credentials reach generated code.
    env = {k: v for k, v in os.environ.items() if k.upper() in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP')}
    try:
        with tempfile.TemporaryDirectory(prefix='rsi-policy-') as cwd:
            completed = subprocess.run([sys.executable, '-I', '-S', '-B', str(WORKER)], input=payload,
                                       capture_output=True, text=True, encoding='utf-8', cwd=cwd,
                                       env=env, timeout=12)
        result = json.loads(completed.stdout)
        if completed.returncode or 'error' in result:
            raise PolicyError(result.get('error', 'Policy worker failed'))
        return result['result']
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as error:
        raise PolicyError('Policy worker timed out or returned an invalid result') from error


def components(sources):
    def proxy(component, function):
        def call(*args):
            seed = None
            if function in ('select_parents', 'select_operators'):
                args, rng = args[:-1], args[-1]
                seed = rng.getrandbits(64)
            result = invoke(sources, component, function, args, seed)
            if function == 'select_para' and (type(result) is not int or not 1 <= result <= 32):
                raise PolicyError('Round para must be 1..32')
            if function in ('build_context', 'update_memory', 'define_operator') and not isinstance(result, dict):
                raise PolicyError(function + ' must return a JSON object')
            if function in ('select_parents', 'select_operators', 'update_population'):
                if not isinstance(result, list) or any(not isinstance(v, str) for v in result):
                    raise PolicyError(function + ' must return a list of strings')
            return result
        return call
    return SimpleNamespace(**{name: SimpleNamespace(**{f: proxy(name, f) for f in FUNCTIONS[name]})
                              for name in COMPONENTS})


def validate(sources):
    invoke(sources, '', 'validate', [])
