"""RSI with an explicit greedy or diversity starting policy."""
import argparse

from aad.meta_online import VERSION
from scripts.meta_recovery import run_with_recovery


def build_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from', dest='origin', required=True, choices=('greedy', 'diversity'),
                        help='Starting policy; subsequent meta updates may change all six components')
    args, remaining = parser.parse_known_args(argv)
    for option in ('--method', '--initial-policy'):
        if any(value == option or value.startswith(option + '=') for value in remaining):
            parser.error(f'{option} is controlled by --from')
    defaults = []
    for option, value in (('--meta-interval', '12'), ('--meta-reasoning-effort', 'high')):
        if not any(arg == option or arg.startswith(option + '=') for arg in remaining):
            defaults.extend((option, value))
    return ['run', '--method', VERSION, '--initial-policy', args.origin, *defaults, *remaining]


if __name__ == '__main__':
    run_with_recovery(build_args())
