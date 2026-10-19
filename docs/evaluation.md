# Evaluation setup

Official evaluation uses ALE-Bench's generators, Docker-based execution, time limits, and scoring. Candidate programs should not be executed directly on the host.

## Dependencies and hardware

Install the `ale-v5-1` extra from the root package. The dependency pins an ALE-Bench commit; the `dsh-meta` extra pins the SDK used by RSI. Install Docker in Linux or WSL and make the official tool images available according to ALE-Bench's setup instructions.

The current CPU policy expects 32 online logical CPUs in 16 consistent SMT pairs. It reserves two complete physical-core groups for services and leases the remaining groups for solvers. The topology check fails on unsupported layouts; adapting it requires preserving sibling exclusion and official timing boundaries.

RSI adds load-based admission through a shared monitor. On Windows/WSL it also checks host CPU load. Stale or missing samples fail closed.

## Public task bundles

Use the appropriate preparation entry to build a task bundle from an official problem archive:

```bash
python run_v5.py prepare --help
python scripts/rsi_search.py prepare --help
```

RSI defaults to public instance seeds 0–19; the task bundle and `--public-seeds` must agree. Its fixed initial bank must be supplied separately and pass `tools/fixed_initials.py verify`. The current reader accepts its known, hashed bank formats; an arbitrary folder of source files is not a compatible bank.

## RSI CPU monitor

Create a policy file using the same topology and admission functions as the evaluator, then start one monitor on the evaluation host:

```bash
python -c "import json,docker; from ale_bench.constants import RUST_TOOL_DOCKER_IMAGE; from aad.cpu_pool import detect_policy; from aad.idle_cpu_pool import idle_policy; p=idle_policy(detect_policy(docker.from_env(),RUST_TOOL_DOCKER_IMAGE)); open('cpu-policy.json','w').write(json.dumps(p,indent=2))"
python scripts/cpu_monitor.py cpu-policy.json
```

Keep the monitor running while RSI evaluation controllers use that host. Do not start multiple monitors for the same shared state. Generated `cpu-policy.json` describes the local machine and is excluded from Git.

## Scope

The repository contains source code and offline tests. It does not include official benchmark archives, private instances, fixed initial-bank data, API credentials, or generated experiments. Unit tests do not exercise a paid model session or certify Docker readiness.
