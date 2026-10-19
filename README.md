# lwbAAD

Algorithm search with LLM-generated programs and an optional self-improving search policy.

| Method | Description | Entry point |
|---|---|---|
| **AAD v5** | A fixed evolutionary search policy with parallel candidate generation, mutation, crossover, and repair. | `python run_v5.py` |
| **RSI** | An outer agent updates the search policy using public evaluation evidence. | `python run_rsi.py` |

RSI supports two starting policies:

- `--from greedy`: all four children use the current best program as their parent.
- `--from diversity`: each child independently samples a parent, with replacement, from the retained population. The initial population policy retains up to eight programs.

The two starting policies otherwise share the same operators and feedback. Subsequent meta updates can change any of the six policy components: parallelism, parent selection, operator selection, operator definitions, memory, and population management.

## Requirements

- Python 3.11+, Git, and a DeepSeek API key.
- Linux or WSL with Linux Docker for official ALE-Bench evaluation.
- The pinned ALE-Bench and DeepSeek Harness SDK dependencies shown below.
- For RSI, a compatible fixed initial-program bank supplied separately.

The evaluator currently validates a **16-core / 32-thread CPU topology**. Other topologies require an evaluator adaptation. RSI evaluation also requires the shared CPU load monitor. See [evaluation setup](docs/evaluation.md) before running a search.

## Install

Clone the repository with its tags. The `rsi-source-baseline` tag identifies the initial policy used to create isolated Git worktrees; downloading a ZIP alone is insufficient for RSI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dsh-meta,ale-v5-1]"
```

Set the API key in your shell, or pass `--ask-key` for a hidden prompt:

```bash
export AAD_API_KEY="your-api-key"
```

In PowerShell, use `$env:AAD_API_KEY = "your-api-key"`. Keys are read at runtime and should not be saved in source files.

## AAD v5

Inspect the preparation and search options:

```bash
python run_v5.py prepare --help
python run_v5.py run --help
```

For a prepared public task bundle:

```bash
python run_v5.py run --task /path/to/public-task --run-dir runs/v5 --calls 100 --para 4 --token-budget 10000000
```

AAD v5 has a fixed search policy. Its implementation is isolated in [`v5/`](v5/README.md).

## RSI

Check the available options and validate your initial bank:

```bash
python run_rsi.py --help
python scripts/rsi_search.py run --help
python tools/fixed_initials.py verify --bank /path/to/initial-bank
```

Choose a starting policy and use a new output directory for each run:

```bash
python run_rsi.py --from greedy --task /path/to/public-task --bank /path/to/initial-bank --run-dir runs/rsi-greedy --calls 100 --token-budget 10000000
python run_rsi.py --from diversity --task /path/to/public-task --bank /path/to/initial-bank --run-dir runs/rsi-diversity --calls 100 --token-budget 10000000
```

The wrapper defaults to `deepseek-flash`, inner reasoning effort `low`, outer reasoning effort `high`, and a meta update every 12 completed rounds. Use `--meta-interval` and `--meta-reasoning-effort` to override the outer defaults. A model alias does not pin a backend revision.

The fixed initial bank is required and is not included in this repository. The reader verifies its manifest and program hashes; it does not silently generate replacement initials. Initialization evaluates the imported programs. See [method and budget semantics](docs/methods.md) for details.

## Tests

```bash
python -m unittest discover -s tests -q
cd v5
python -m unittest discover -s tests -q
```

Tests use mock model calls and evaluation results. Checks that need separately installed benchmark assets are skipped when those assets are absent. Passing unit tests does not establish that a machine is ready for official evaluation.

## Layout

```text
aad/            RSI and shared search/evaluation implementation
  search_policy/ Six editable search components and their engine
  meta_online/  DSH agent and online policy updates
v5/             Isolated AAD v5 implementation
scripts/        RSI search, recovery, and CPU monitor entry points
tools/          Fixed initial-bank validation
tests/          Offline regression tests
docs/           Method and evaluation documentation
```

Generated runs, benchmark data, credentials, and initial-program banks are excluded from Git. Modules are named by function; internal protocol identifiers retain their original values for compatibility.
