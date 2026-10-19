"""Restricted, tool-free Python policy worker. Invoked with python -I -S.

No user imports, files, network, shell, reflection or native extensions. This
is a deliberately small Python interface, not a general Python/OS sandbox.
"""
import ast
import builtins
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import random
import statistics
import sys
from types import SimpleNamespace

NAMES = ('operators', 'parallelism', 'parents', 'operator_selection', 'memory', 'population')
BUILTINS = {name: getattr(builtins, name) for name in (
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 'int', 'isinstance',
    'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'ord', 'range', 'reversed', 'round',
    'set', 'sorted', 'str', 'sum', 'tuple', 'zip', 'ValueError', 'KeyError', 'IndexError',
    'RuntimeError', 'TypeError', 'AttributeError', 'Exception')}
NODES = {getattr(ast, name) for name in (
    'Module', 'FunctionDef', 'arguments', 'arg', 'Return', 'Assign', 'AnnAssign', 'AugAssign',
    'For', 'While', 'If', 'Break', 'Continue', 'Pass', 'Expr', 'Raise', 'Try', 'ExceptHandler',
    'ImportFrom', 'alias', 'Name', 'Load', 'Store', 'Del', 'Delete', 'Constant', 'List', 'Tuple',
    'Set', 'Dict', 'ListComp', 'SetComp', 'DictComp', 'GeneratorExp', 'comprehension', 'Lambda',
    'Call', 'keyword', 'Attribute', 'Subscript', 'Slice', 'Starred', 'IfExp', 'BoolOp', 'BinOp',
    'UnaryOp', 'Compare', 'JoinedStr', 'FormattedValue', 'And', 'Or', 'Add', 'Sub', 'Mult',
    'Div', 'FloorDiv', 'Mod', 'Pow', 'USub', 'UAdd', 'Not', 'Eq', 'NotEq', 'Lt', 'LtE',
    'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn', 'BitAnd', 'BitOr', 'BitXor', 'Invert',
    'LShift', 'RShift')}
SAFE_IMPORTS = {
    (0, 'copy'): SimpleNamespace(deepcopy=deepcopy),
    (0, 'math'): SimpleNamespace(**{n: getattr(math, n) for n in (
        'ceil', 'floor', 'sqrt', 'log', 'log2', 'log10', 'exp', 'isfinite', 'inf', 'pi')}),
    (0, 'statistics'): SimpleNamespace(mean=statistics.mean, median=statistics.median,
                                       pstdev=statistics.pstdev),
    (0, 'collections'): SimpleNamespace(Counter=Counter, defaultdict=defaultdict),
    (3, 'candidate_policy'): SimpleNamespace(valid_for_archive=lambda r: bool(r.get('full', {}).get('valid'))
                                     and all(x['valid'] for x in r.get('repeats', []))),
}


def validate(source, name):
    if not isinstance(source, str) or len(source.encode()) > 64000:
        raise ValueError('Component must be UTF-8 Python under 64 KB')
    tree = ast.parse(source, filename=name)
    for node in ast.walk(tree):
        if type(node) not in NODES:
            raise ValueError('Unsupported policy syntax: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith('_') and node.id != '_':
            raise ValueError('Private names/reflection are forbidden')
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_') or node.attr in ('format', 'format_map') or not isinstance(node.ctx, ast.Load):
                raise ValueError('Private/reflection attributes and attribute writes are forbidden')
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith('_') or node.decorator_list or node.returns:
                raise ValueError('Unsupported function declaration')
        if isinstance(node, ast.arg) and (node.annotation or node.arg.startswith('__')):
            raise ValueError('Unsupported function argument')
        if isinstance(node, ast.ImportFrom):
            key = (node.level, node.module)
            if key not in SAFE_IMPORTS and key != (1, 'operators'):
                raise ValueError('Import is not allowed: ' + str(key))
            for alias in node.names:
                if alias.name.startswith('_') or alias.name == '*' or (alias.asname or '').startswith('_'):
                    raise ValueError('Private/wildcard imports are forbidden')
    return tree


def load(sources):
    if set(sources) != {n + '.py' for n in NAMES}:
        raise ValueError('Exactly six components required')
    modules = {}

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        if (level, name) == (1, 'operators'):
            module = modules['operators']
        else:
            module = SAFE_IMPORTS[(level, name)]
        if any(not hasattr(module, field) for field in fromlist):
            raise ValueError('Unsupported imported symbol')
        return module

    for name in NAMES:
        tree = validate(sources[name + '.py'], name)
        scope = {'__builtins__': {**BUILTINS, '__import__': importer}}
        exec(compile(tree, name + '.py', 'exec'), scope)
        modules[name] = SimpleNamespace(**{k: v for k, v in scope.items() if not k.startswith('_')})
    return modules


def limits():
    """Bound memory/CPU in addition to the controller's wall timeout."""
    if sys.platform != 'win32':
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    else:
        import ctypes
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', w.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                        ('processes', w.DWORD), ('affinity', ctypes.c_size_t),
                        ('priority', w.DWORD), ('scheduling', w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ('ro', 'wo', 'oo', 'rt', 'wt', 'ot')]

        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.GetCurrentProcess.restype = w.HANDLE
        job = kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.basic.flags = 0x100 | 0x2  # process memory + per-process CPU time
        info.basic.process_time = 8 * 10_000_000
        info.process_memory = 512 * 1024**2
        if (not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
                or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())):
            raise OSError('Could not establish policy worker resource limits')
        # Keep the handle open for this process's lifetime.


def main():
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    limits()
    payload = json.load(sys.stdin)
    # All modules have been imported before the audit hook; no external tools or I/O below.
    def audit(event, args):
        if (event == 'open' or event == 'import' or event.startswith(('socket.', 'subprocess.', 'ctypes.',
                'os.', 'winreg.', 'urllib.', 'http.', 'shutil.'))):
            raise PermissionError('Policy workers cannot access files, processes or the network')
    sys.addaudithook(audit)
    modules = load(payload['sources'])
    if payload['function'] == 'validate':
        result = True
    else:
        function = getattr(modules[payload['component']], payload['function'])
        args = payload['args']
        if payload.get('rng_seed') is not None:
            args.append(random.Random(payload['rng_seed']))
        result = function(*args)
    encoded = json.dumps({'result': result}, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > 2_000_000:
        raise ValueError('Policy result too large')
    sys.stdout.write(encoded)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        sys.stdout.write(json.dumps({'error': type(error).__name__ + ': ' + str(error)[:1500]}))
        sys.exit(1)
