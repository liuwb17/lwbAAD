"""Prepare, evolve, freeze and test ALE-Bench batch problems with AAD v5."""
import argparse
from contextlib import contextmanager
import getpass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aad.io import digest, read_json
from aad.ale_batch import (AleBatchEvaluator, download_archive, load_task, prepare,
                        private_test, sha256_file, statement_for)
from aad.parallel_engine import V51Engine, V51Provider


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'parallel_engine.lock'
    try:
        stream = path.open('x', encoding='utf-8')
    except FileExistsError:
        raise ValueError('Run is locked; inspect the other process before removing a stale lock') from None
    try:
        with stream:
            stream.write('v5.2 single writer\n')
        yield
    finally:
        path.unlink()


def implementation_hashes():
    project = Path(__file__).resolve().parents[1]
    paths = sorted((project / 'aad').rglob('*.py')) + [Path(__file__).resolve()]
    return {p.relative_to(project).as_posix(): sha256_file(p) for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare', help='Generate only official public training cases')
    prep.add_argument('--problem', required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--archive', type=Path, help='Official problem ZIP; otherwise download pinned revision')
    prep.add_argument('--seed-start', type=int, default=0)
    prep.add_argument('--count', type=int, default=50)
    prep.add_argument('--seeds', type=int, nargs='+', help='Explicit public seeds, overrides start/count')
    for action in ('run', 'freeze', 'private'):
        sub = commands.add_parser(action)
        sub.add_argument('--run-dir', type=Path, required=True)
        sub.add_argument('--task', type=Path, required=True)
        if action == 'run':
            sub.add_argument('--calls', type=int, default=256)
            sub.add_argument('--token-budget', type=int, default=0)
            sub.add_argument('--model', default='deepseek-flash')
            sub.add_argument('--para', type=int, choices=[4], default=None,
                             help='Four concurrent LLM tasks per round; omitted retains serial v5 scheduling')
            sub.add_argument('--failure-experience', action=argparse.BooleanOptionalAction, default=False,
                             help='Provide up to eight failed attempts specific to the main parent')
            sub.add_argument('--plan', action=argparse.BooleanOptionalAction, default=False,
                             help='One separately budgeted Plan call for four cold-start hints (default: disabled)')
            sub.add_argument('--promotion-margin', type=float, default=0)
            sub.add_argument('--smoke-count', type=int, default=6)
            sub.add_argument('--resume', action='store_true')
            sub.add_argument('--ask-key', action='store_true')
        elif action == 'private':
            sub.add_argument('--archive', type=Path,
                             help='Same official ZIP as training; otherwise download pinned revision')
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        if args.count < 1:
            parser.error('--count must be positive')
        seeds = args.seeds if args.seeds is not None else list(range(args.seed_start, args.seed_start + args.count))
        archive = args.archive or download_archive(args.problem)
        task = prepare(archive, args.problem, args.output, seeds)
        print({'problem': task['problem_id'], 'train_count': len(task['cases']),
               'statement_has_images': task['statement_has_images']})
        return
    root = args.run_dir.resolve()
    if args.action != 'run' and not (root / 'state.json').exists():
        raise ValueError('Existing run required')
    with run_lock(root):
        task, train = load_task(args.task)
        task_hash = digest(json.dumps(task, sort_keys=True))
        if args.action in ('freeze', 'private'):
            protocol = read_json(root / 'protocol.json')
            config = protocol['config']
            if protocol.get('version') != '5.2' or task_hash != config['task_sha256']:
                raise ValueError('Not this v5.2 run/task')
            if implementation_hashes() != config['implementation_hashes']:
                raise ValueError('Implementation changed; use the pinned run source')
            if args.action == 'freeze':
                engine = V51Engine(root, None, None, train, config)
                print({'frozen': engine.finalize()})
            else:
                state = read_json(root / 'state.json')
                if not state.get('frozen'):
                    raise ValueError('Freeze training before private testing')
                archive = args.archive or download_archive(task['problem_id'])
                print(private_test(root, args.task, archive))
            return
        exists = (root / 'state.json').exists()
        if exists != args.resume:
            raise ValueError('Use a new run directory, or --resume for an existing run')
        if not exists and any(p.name != 'parallel_engine.lock' for p in root.iterdir()):
            raise ValueError('New run directory must be empty')
        if exists and read_json(root / 'state.json').get('frozen'):
            raise ValueError('Development is closed')
        if args.calls < 1 or args.token_budget < 0:
            raise ValueError('Invalid request/token budget')
        evaluator = AleBatchEvaluator(args.task, root / 'measurements')
        config = {'problem_id': task['problem_id'], 'direction': task['direction'],
                  'time_limit': task['time_limit'], 'task_sha256': task_hash,
                  'runtime': evaluator.runtime,
                  'implementation_hashes': implementation_hashes(),
                  'model': args.model, 'calls': args.calls, 'token_budget': args.token_budget,
                  'plan': args.plan, 'para': args.para, 'failure_experience': args.failure_experience,
                  'promotion_margin': args.promotion_margin,
                  'smoke_count': args.smoke_count}
        # Check the protocol before opening a provider or touching its ledger.
        engine = V51Engine(root, None, evaluator, train, config)
        key = getpass.getpass('AAD API key (hidden): ') if args.ask_key else None
        engine.provider = V51Provider(root, args.model, args.calls, args.token_budget,
                                    key=key, problem=statement_for(task))
        engine.run()
        print('Training stopped. Use freeze, then private, for terminal evaluation.')


if __name__ == '__main__':
    main()
