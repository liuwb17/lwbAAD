"""Official ALE batch tools without constructing a private-loading Session."""
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import zipfile
from pathlib import Path, PurePosixPath

from .io import digest, read_json, save_json
from .problem import Case
from .candidate_evaluation import (EVALUATION_PROTOCOL, aggregate_evaluations,
                              evaluation_seeds, seeded_source)
from .container_execution import EXECUTION_PROTOCOL

ALE_COMMIT = '3da9b12fb5d112dabb3af693d1a42031c95142bc'
DATA_REVISION = '0f42617'
STATUS = {'ACCEPTED': 'AC', 'COMPILATION_ERROR': 'CE', 'WRONG_ANSWER': 'WA',
          'TIME_LIMIT_EXCEEDED': 'TLE', 'RUNTIME_ERROR': 'RE',
          'MEMORY_LIMIT_EXCEEDED': 'MLE', 'OUTPUT_LIMIT_EXCEEDED': 'OLE',
          'INTERNAL_ERROR': 'IE'}


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_archive(problem_id):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', problem_id):
        raise ValueError('Invalid problem ID')
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id='SakanaAI/ALE-Bench', repo_type='dataset',
                               revision=DATA_REVISION, filename=f'{problem_id}.zip'))


def build_tools(root):
    from ale_bench.data import build_rust_tools
    build_rust_tools(Path(root) / 'tools')


def generate(seeds, root):
    from ale_bench.tool_wrappers import generate_inputs
    return generate_inputs(seeds, {}, Path(root))


def archive_files(source):
    """Canonicalize official ZIP separators without changing archive bytes."""
    files = {}
    for member in source.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
            raise ValueError('Unsafe archive member')
        if member.is_dir():
            continue
        name = path.as_posix()
        if name in files:
            raise ValueError('Ambiguous archive member: ' + name)
        files[name] = member
    return files


def read_metadata(archive, problem_id, *, supported_types=('batch',)):
    with zipfile.ZipFile(archive) as source:
        data = json.loads(source.read(archive_files(source)[f'{problem_id}/data.json']))
    if data['metadata']['problem_id'] != problem_id:
        raise ValueError('Problem ID mismatch')
    if data['metadata']['problem_type'] not in supported_types:
        raise ValueError('Unsupported problem type; legacy entry supports non-interactive (batch) problems only')
    if data['metadata']['score_type'] not in ('maximize', 'minimize'):
        raise ValueError('Unsupported score direction')
    return data


def prepare(archive, problem_id, destination, seeds=None, *, supported_types=('batch',)):
    """Only public seeds, statement and trusted tools enter the training bundle."""
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Task destination must be empty')
    data = read_metadata(archive, problem_id, supported_types=supported_types)
    seeds = list(range(50)) if seeds is None else list(seeds)
    if (not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
            or not set(seeds).issubset(data['seeds']['public'])):
        raise ValueError('Training seeds must be a nonempty unique subset of official public seeds')
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(archive) as source:
        members = archive_files(source)
        statement = source.read(members[f'{problem_id}/statement_en.md']).decode('utf-8')
        for name, member in members.items():
            if not name.startswith(f'{problem_id}/tools/'):
                continue
            relative = PurePosixPath(name).relative_to(problem_id)
            target = (destination / str(relative)).resolve()
            if not target.is_relative_to(destination) or '..' in relative.parts:
                raise ValueError('Unsafe archive member')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(member))
            files[relative.as_posix()] = sha256_file(target)
    if not files:
        raise ValueError('Official tools missing from archive')
    build_tools(destination)
    inputs = generate(seeds, destination)
    if len(inputs) != len(seeds):
        raise ValueError('Generator returned wrong case count')
    cases = []
    for seed, text in zip(seeds, inputs, strict=True):
        filename = f'public/{seed}.txt'
        path = destination / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding='utf-8')
        cases.append({'name': f'public_{seed}', 'seed': seed, 'file': filename, 'sha256': digest(text)})
    manifest = {'version': '5.1', 'problem_id': problem_id, 'problem_type': data['metadata']['problem_type'],
                'direction': data['metadata']['score_type'], 'statement': statement,
                'time_limit': data['constraints']['time_limit'],
                'memory_limit': data['constraints']['memory_limit'],
                'archive_sha256': sha256_file(archive), 'dataset_revision': DATA_REVISION,
                'ale_commit': ALE_COMMIT, 'tool_sources': files, 'cases': cases,
                'statement_has_images': bool(data.get('image_files'))}
    save_json(destination / 'task.json', manifest)
    return manifest


def load_task(root, *, supported_types=('batch',)):
    root = Path(root).resolve()
    task = read_json(root / 'task.json')
    if task.get('version') != '5.1' or task.get('problem_type') not in supported_types:
        raise ValueError('Expected a v5.1 task bundle of supported type: ' + ', '.join(supported_types))
    if task['direction'] not in ('maximize', 'minimize'):
        raise ValueError('Invalid score direction')
    for key in ('time_limit', 'memory_limit'):
        if not math.isfinite(task[key]) or task[key] <= 0:
            raise ValueError(f'Invalid {key}')
    def file_path(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Task path escapes bundle')
        return path
    for name, expected in task['tool_sources'].items():
        if sha256_file(file_path(name)) != expected:
            raise ValueError('Official tool source changed')
    cases = []
    for row in task['cases']:
        text = file_path(row['file']).read_text(encoding='utf-8')
        if digest(text) != row['sha256']:
            raise ValueError('Training input changed')
        cases.append(Case(row['name'], text))
    if not cases or len({c.name for c in cases}) != len(cases):
        raise ValueError('Training cases must be nonempty and uniquely named')
    return task, cases


def statement_for(task):
    statement = (task['statement'] + '\n\nExecution contract:\n'
            'Write a standalone C++17 program using the standard library. Read stdin and write '
            'only the required answer to stdout. No command-line time budget is supplied. '
            f'The process time limit is {task["time_limit"]} seconds including startup and I/O; '
            f'memory limit is {task["memory_limit"]} bytes. Finish with a safety margin. '
            f'Optimize the arithmetic mean of raw per-case scores ({task["direction"]}). '
            'The framework defines the C++ macro AAD_RANDOM_SEED as 0, 1 or 2; initialize all '
            'PRNGs from it, never redefine it or seed from time/random_device. '
            'No files, network, subprocesses, evaluator inspection '
            'or known-input lookup. No initial solution is supplied.')
    if task.get('problem_type') == 'reactive':
        statement += ('\nThis is an interactive (reactive) problem. The official tester starts your program '
                      'and exchanges messages over stdin/stdout. Follow the stated turn order, flush each '
                      'required output before reading the next response, and handle EOF/termination. '
                      'The generator file is private to the tester, not your stdin: do not read future '
                      'responses, hidden state or the tester seed from it. Keep diagnostics on stderr.')
    return statement


def summarize(items, cases, direction):
    if not cases or len(items) != len(cases):
        raise ValueError('Evaluator case count mismatch')
    sign = 1 if direction == 'maximize' else -1
    rows = []
    for case, item in zip(cases, items, strict=True):
        status = STATUS.get(item.judge_result.value)
        if status is None or status == 'IE':
            raise OSError(f'ALE infrastructure failure: {item.message}')
        score = item.absolute_score if status == 'AC' else 0
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Nonfinite or missing official score')
        if not math.isfinite(item.execution_time) or item.execution_time < 0:
            raise ValueError('Invalid measured duration')
        rows.append({'name': case.name, 'input_sha256': case.fingerprint,
                     'status': status, 'raw_score': score, 'score': sign * score,
                     'seconds': item.execution_time, 'stdout': item.output_str or '',
                     'message': item.message or '', 'stderr': item.error_str or ''})
    return {'valid': all(r['status'] == 'AC' for r in rows),
            'raw_mean': statistics.mean(r['raw_score'] for r in rows),
            'mean': statistics.mean(r['score'] for r in rows), 'cases': rows}


def runtime_identity(root):
    """Pin SDK code, built tools and actual Docker image IDs in each run."""
    import ale_bench
    from ale_bench import constants
    from ale_bench.code_language import CodeLanguage, JudgeVersion, get_docker_image_name
    from ale_bench.utils import docker_client
    from .cpu_pool import detect_policy
    sdk_root = Path(ale_bench.__file__).parent
    sdk_hashes = {p.relative_to(sdk_root).as_posix(): sha256_file(p)
                  for p in sorted(sdk_root.rglob('*.py'))}
    names = {get_docker_image_name(CodeLanguage.CPP17, JudgeVersion.V202301),
             constants.RUST_TOOL_DOCKER_IMAGE}
    with docker_client() as client:
        images = {name: client.images.get(name).id for name in sorted(names)}
        cpu_policy = detect_policy(client, constants.RUST_TOOL_DOCKER_IMAGE)
    binaries = {name: sha256_file(Path(root) / 'tools/target/release' / name)
                for name in ('gen', 'tester', 'vis')
                if (Path(root) / 'tools/target/release' / name).is_file()}
    if 'gen' not in binaries or 'tester' not in binaries:
        raise ValueError('Build the official gen and tester tools before running')
    return {'ale_version': importlib.metadata.version('ale-bench'),
            'execution_protocol': EXECUTION_PROTOCOL,
            'sdk_sha256': digest(json.dumps(sdk_hashes, sort_keys=True)),
            'binaries': binaries, 'docker_images': images, 'judge_version': '202301',
            'cpu_policy': cpu_policy}


class AleBatchEvaluator:
    def __init__(self, task_root, cache_root):
        self.task_root, self.cache_root = Path(task_root).resolve(), Path(cache_root)
        self.task, _ = load_task(self.task_root)
        self.runtime = runtime_identity(self.task_root)
        from .cpu_pool import make_runner
        self.run_cases = make_runner(self.runtime['cpu_policy'], self.cache_root / 'cpu_leases')
        self.signature = digest(json.dumps({'task': self.task, 'runtime': self.runtime}, sort_keys=True))

    def __call__(self, code, cases, seconds, replica, *, seed=0):
        if seconds != self.task['time_limit']:
            raise ValueError('Use the official task time limit')
        executable_source = seeded_source(code, seed)
        key = digest(json.dumps([EVALUATION_PROTOCOL, self.signature, digest(executable_source), seconds, replica,
                                 [(c.name, c.fingerprint) for c in cases]]))
        path = self.cache_root / f'{key}.json'
        if path.exists():
            return read_json(path)
        from ale_bench.code_language import CodeLanguage, JudgeVersion
        from ale_bench.data import ProblemType
        items = self.run_cases(inputs=[c.text for c in cases], code=executable_source,
                          code_language=CodeLanguage.CPP17, judge_version=JudgeVersion.V202301,
                          time_limit=seconds, memory_limit=self.task['memory_limit'],
                          problem_id=self.task['problem_id'], problem_type=(ProblemType.BATCH
                              if self.task['problem_type'] == 'batch' else ProblemType.REACTIVE),
                          tool_dir=self.task_root, return_details=True,
                          skip_local_visualization=True)
        result = summarize(items, cases, self.task['direction'])
        result.update(seed=seed, source_sha256=digest(code), executable_source_sha256=digest(executable_source))
        for row in result['cases']:
            row['seed'] = seed
        save_json(path, result)
        return result


def private_test(run, task_root, archive, *, evaluator_factory=None, supported_types=('batch',)):
    """Read private seed field only after verifying a frozen v5.2 source and protocol."""
    run = Path(run)
    protocol, state = read_json(run / 'protocol.json'), read_json(run / 'state.json')
    if (protocol.get('version') != '5.2' or not state.get('frozen')
            or not state.get('development_closed') or state.get('pending')):
        raise ValueError('Freeze v5.2 training before private testing')
    if digest(json.dumps(protocol, sort_keys=True)) != state['signature']:
        raise ValueError('Protocol changed')
    champion = state['champion']
    record = state['records'][champion]
    seeds_for_algorithm = evaluation_seeds(record.get('is_randomized'))
    if (protocol['config'].get('evaluation_protocol') != EVALUATION_PROTOCOL
            or len(record.get('evaluations', [])) != len(seeds_for_algorithm)
            or not all(r['valid'] for r in record['evaluations'])):
        raise ValueError('Incomplete or incompatible frozen evaluations')
    code = (run / 'best.cpp').read_text(encoding='utf-8')
    if digest(code) != champion or digest((run / 'candidates' / champion / 'solution.cpp').read_text(
            encoding='utf-8')) != champion:
        raise ValueError('Frozen source changed')
    task, _ = load_task(task_root, supported_types=supported_types)
    if digest(json.dumps(task, sort_keys=True)) != protocol['config']['task_sha256']:
        raise ValueError('Task changed')
    if sha256_file(archive) != task['archive_sha256']:
        raise ValueError('Official archive changed')
    report_path, receipt_path = run / 'private_report.json', run / 'private_receipt.json'
    if report_path.exists() or receipt_path.exists():
        raise ValueError('Private evaluation already started; inspect existing receipt')
    evaluator = (evaluator_factory or AleBatchEvaluator)(task_root, run / 'private_measurements')
    if evaluator.runtime != protocol['config']['runtime']:
        raise ValueError('Evaluation environment changed')
    # Exclusive creation also prevents two private writers for the same run.
    with receipt_path.open('x', encoding='utf-8') as receipt:
        json.dump({'status': 'started', 'champion': champion}, receipt)
    data = read_metadata(archive, task['problem_id'], supported_types=supported_types)
    seeds = data['seeds']['private']
    if not seeds or set(seeds) & {r['seed'] for r in task['cases']}:
        raise ValueError('Invalid or overlapping private seeds')
    texts = generate(seeds, task_root)
    cases = [Case(f'private_{i}', text) for i, text in enumerate(texts)]
    if len(cases) != len(seeds):
        raise ValueError('Private generator case count mismatch')
    if {c.fingerprint for c in cases} & set(protocol['inputs'].values()):
        raise ValueError('Private inputs overlap training')
    results = []
    for index, seed in enumerate(seeds_for_algorithm):
        results.append(evaluator(code, cases, task['time_limit'], f'private-r{index}', seed=seed))
        if not results[-1]['valid']:
            break
    result = aggregate_evaluations(results)
    report = {'version': '5.2', 'champion': champion, 'problem_id': task['problem_id'],
              'direction': task['direction'], 'private_count': len(cases),
              'private_mean': result['raw_mean'], 'valid': result['valid'],
              'is_randomized': record['is_randomized'], 'evaluation_count': len(results),
              'algorithm_seeds': list(seeds_for_algorithm[:len(results)]),
              'valid_count': sum(r['status'] == 'AC' for r in result['cases']),
              'seed_sha256': digest(json.dumps(seeds)),
              'scope': 'Official ALE private seeds and tools; raw mean, not leaderboard rank'}
    save_json(report_path, report)
    save_json(receipt_path, {'status': 'finished', 'champion': champion})
    summary = read_json(run / 'report.json')
    summary.update(private_evaluated=True, private_report='private_report.json')
    save_json(run / 'report.json', summary)
    return report
