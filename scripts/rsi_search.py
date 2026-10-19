"""RSI search: evolve, freeze, then official private evaluation.

Prepare uses official public seeds 0..19 by default.
This entry never calls Plan or generates initial programs.
"""
import argparse
from contextlib import contextmanager
import getpass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aad.io import digest, read_json, save_json
from aad.ale_batch import download_archive, sha256_file
from aad.ale_evaluation import (AleBatchEvaluator, private_test, prepare, DEFAULT_PUBLIC_SEEDS,
                         load_task, statement_for)
from aad.idle_cpu_pool import WINDOWS_PROBE
from aad.parallel_engine import V51Provider
from aad.fixed_initials import FixedInitialEngine, READER_PATH
from aad.search_policy.engine import R0Engine
from aad.meta_online.engine import OnlineEngine
from aad.meta_online.provider import DEFAULT_META_MODEL, DEFAULT_META_REASONING_EFFORT
from aad.meta_online.dsh import DshProvider as OnlineProvider, DSH_PROTOCOL, SDK_VERSION
from aad.meta_online import VERSION as ONLINE_VERSION


ENGINES = {'5.2': FixedInitialEngine, '5.2.r.0': R0Engine, ONLINE_VERSION: OnlineEngine}


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'parallel_engine.lock'
    with path.open('x', encoding='utf-8') as stream:
        stream.write('fixed-bank single writer\n')
    try:
        yield
    finally:
        path.unlink()


def implementation_hashes():
    project = Path(__file__).resolve().parents[1]
    paths = sorted((project / 'aad').rglob('*.py')) + [Path(__file__).resolve(), WINDOWS_PROBE]
    return {**{p.relative_to(project).as_posix(): sha256_file(p) for p in paths},
            '../tools/fixed_initials.py': sha256_file(READER_PATH)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare', help='Official generator, public 0..19 by default')
    prep.add_argument('--problem', required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--archive', type=Path)
    prep.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_PUBLIC_SEEDS))
    for action in ('run', 'freeze', 'private'):
        sub = commands.add_parser(action)
        sub.add_argument('--run-dir', type=Path, required=True)
        sub.add_argument('--task', type=Path, required=True)
        sub.add_argument('--bank', type=Path, help='Canonical fixed bank; manifest identity is always checked')
        if action == 'run':
            sub.add_argument('--private-after-run', action=argparse.BooleanOptionalAction, default=True,
                             help='Freeze and evaluate official private seeds on normal budget completion (default: on)')
            sub.add_argument('--public-seeds', type=int, nargs='+', default=list(DEFAULT_PUBLIC_SEEDS),
                             help='Expected public seeds in the prepared bundle (default: 0..19)')
            sub.add_argument('--method', choices=ENGINES, default='5.2.r.0')
            sub.add_argument('--calls', type=int, required=True,
                             help='Shared solver + meta request boundary; finish last round and one-shot repairs')
            sub.add_argument('--token-budget', type=int, default=0)
            sub.add_argument('--model', default='deepseek-flash', help='Inner solver model; low reasoning')
            sub.add_argument('--meta-model', default=DEFAULT_META_MODEL,
                             help='Outer DeepSeek harness model, independent of --model')
            sub.add_argument('--meta-reasoning-effort', default=DEFAULT_META_REASONING_EFFORT,
                             help='Outer harness reasoning effort (default: max)')
            sub.add_argument('--meta-interval', type=int, default=4,
                             help='Online meta review interval in completed rounds (default: 4)')
            sub.add_argument('--initial-policy', choices=('v52', 'minimal', 'greedy', 'diversity'), default='v52',
                             help='Online starting harness; fixed solver initials are unchanged')
            sub.add_argument('--failure-experience', action=argparse.BooleanOptionalAction, default=False)
            sub.add_argument('--promotion-margin', type=float, default=0)
            sub.add_argument('--smoke-count', type=int, default=6)
            sub.add_argument('--allow-invalid-initials', action='store_true',
                             help='Retain failed original slots; evolve only valid originals without replacement')
            sub.add_argument('--resume', action='store_true')
            sub.add_argument('--ask-key', action='store_true')
        if action in ('run', 'private'):
            sub.add_argument('--archive', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        task = prepare(args.archive or download_archive(args.problem), args.problem, args.output, args.seeds)
        print({'problem': task['problem_id'], 'public_seeds': [r['seed'] for r in task['cases']]})
        return
    root = args.run_dir.resolve()
    if args.action != 'run' and not (root / 'state.json').exists():
        raise ValueError('Existing fixed-bank run required')
    with run_lock(root):
        task, train = load_task(args.task)
        task_hash = digest(json.dumps(task, sort_keys=True))
        if args.action != 'run':
            protocol = read_json(root / 'protocol.json')
            config = protocol['config']
            if (protocol['version'] != '5.2' or config.get('harness_version') not in ENGINES
                    or 'fixed_initials' not in config or task_hash != config['task_sha256']):
                raise ValueError('Not this fixed-bank run/task')
            if implementation_hashes() != config['implementation_hashes']:
                raise ValueError('Implementation changed; use the pinned run source')
            engine = ENGINES[config['harness_version']](root, None, None, train, config, bank=args.bank)
            if args.action == 'freeze':
                print({'frozen': engine.finalize()})
            else:
                if not engine.s['frozen']:
                    raise ValueError('Freeze training before private testing')
                report = private_test(root, args.task, args.archive or download_archive(task['problem_id']))
                report['harness_version'] = config['harness_version']
                save_json(root / 'private_report.json', report)
                print(report)
            return
        exists = (root / 'state.json').exists()
        if exists != args.resume:
            raise ValueError('Use a new run directory, or --resume for an existing run')
        if not exists and any(p.name != 'parallel_engine.lock' for p in root.iterdir()):
            raise ValueError('New run directory must be empty')
        if exists and read_json(root / 'state.json').get('development_closed'):
            raise ValueError('Development is closed')
        if args.calls < 1 or args.token_budget < 0:
            raise ValueError('Invalid request/token budget')
        if args.meta_interval < 1:
            raise ValueError('meta-interval must be positive')
        if args.method != ONLINE_VERSION and (args.meta_interval != 4 or args.initial_policy != 'v52'):
            raise ValueError('Initial policy and meta interval options require the online method')
        if [row['seed'] for row in task['cases']] != args.public_seeds:
            raise ValueError('Task public seeds differ; prepare a 0..19 bundle or explicitly set --public-seeds')
        if args.method == ONLINE_VERSION:
            OnlineProvider.check_dependencies()
        evaluator = AleBatchEvaluator(args.task, root / 'measurements')
        config = {'problem_id': task['problem_id'], 'problem_type': task['problem_type'],
                  'direction': task['direction'],
                  'time_limit': task['time_limit'], 'task_sha256': task_hash,
                  'runtime': evaluator.runtime, 'implementation_hashes': implementation_hashes(),
                  'model': args.model, 'calls': args.calls, 'token_budget': args.token_budget,
                  'failure_experience': args.failure_experience, 'promotion_margin': args.promotion_margin,
                  'smoke_count': args.smoke_count, 'allow_invalid_initials': args.allow_invalid_initials}
        config['public_seeds'] = args.public_seeds
        config['private_after_run'] = args.private_after_run
        meta_options = {}
        if args.method == ONLINE_VERSION:
            meta_options = {'meta_model': args.meta_model, 'meta_reasoning_effort': args.meta_reasoning_effort}
            config.update(meta_options)
            config.update(meta_backend='dsh', dsh_protocol=DSH_PROTOCOL, dsh_sdk_version=SDK_VERSION)
        if args.method == ONLINE_VERSION:
            config.update(meta_interval=args.meta_interval)
            if args.initial_policy != 'v52':
                config['meta_initial_policy'] = ('minimal' if args.initial_policy == 'diversity'
                                                  else args.initial_policy)
        engine = ENGINES[args.method](root, None, evaluator, train, config, bank=args.bank)
        if (root / 'STOP_AFTER_CURRENT').exists():
            return
        engine.initialize()  # Verify/evaluate originals before touching a model ledger.
        key = getpass.getpass('AAD API key (hidden): ') if args.ask_key else None
        provider_cls = OnlineProvider if args.method == ONLINE_VERSION else V51Provider
        engine.provider = provider_cls(root, args.model, args.calls, args.token_budget,
                                       key=key, problem=statement_for(task), **meta_options)
        engine.run()
        if (root / 'STOP_AFTER_CURRENT').exists():
            print('Training paused; no freeze or private evaluation.')
        elif args.private_after_run:
            engine.finalize()
            report = private_test(root, args.task, args.archive or download_archive(task['problem_id']))
            report['harness_version'] = args.method
            save_json(root / 'private_report.json', report)
            print(report)
        else:
            print('Training stopped. Use freeze, then private, for terminal evaluation.')


if __name__ == '__main__':
    main()
