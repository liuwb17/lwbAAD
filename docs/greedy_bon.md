# Greedy sequential best-of-N

This is the parent-only greedy baseline used in the October 2026 simple/complex
comparison, released separately from RSI. `N=4` reproduces its search/prompt
mechanism. `N=1` makes candidate generation sequential as well.

“Sequential” refers to **rounds**, not necessarily individual requests:

1. Generate N independent initial programs with identical from-scratch prompts.
2. Select the highest-scoring valid program.
3. Freeze that parent and generate N children concurrently.
4. Wait for the complete batch, then retain the best valid program seen so far.
5. Repeat until the token budget or batch limit is reached.

The implementation retains up to eight source-distinct programs for compatibility
with the experimental archive, but always chooses its single best entry as every
child's parent. Ties are resolved by insertion/slot order, never response speed.
Invalid children do not become parents. If initialization produces no valid
program, the next charged batch starts from scratch again.

## Prompt

The exact system guidance and message construction are in
[`aad/greedy_bon/prompts.py`](../aad/greedy_bon/prompts.py). The user message contains
the problem/execution constraints and, after initialization, the parent code
with “Improve the following parent algorithm.”

System guidance explicitly permits structural changes or replacing the approach.
There is no history, score feedback, donor code, reflection, task-specific
knowledge injection, repair loop, meta update, or fixed initial bank.

## Run

Use the repository's prepared official public task format and
[evaluation setup](evaluation.md). For example:

```bash
python run_v5.py prepare --problem ahc024 --output tasks/ahc024 --seed-start 0 --count 20
export DEEPSEEK_API_KEY="your-api-key"
python run_greedy.py --task tasks/ahc024 --run-dir runs/greedy-024-r1 \
  --n 4 --token-budget 7500000
```

Each repeat needs a new output directory and generates its own initials. No
private split is evaluated. The default is `deepseek-flash`, LOW thinking effort,
393216 maximum output tokens. A model alias does not pin a backend version.
Use `--api-key-env AAD_API_KEY` to reuse the other entry points' variable.

The CLI uses the existing official ALE evaluator and its existing CPU admission
policy, **not** the private ten-core experiment broker. Its machine/topology
requirements remain those documented for this repository. Evaluation is
serialized per evaluator instance; model generation is concurrent within each
batch. The experiment's shared broker can instead be integrated through the
Python `evaluate(code)` callback, without changing the search loop.

## Budget and recovery

- Count all input and output tokens, including cached input and initialization.
- No minimum-call requirement. `--rounds` includes the initial batch.
- Stop allocating requests once known usage reaches the budget. Already issued
  requests drain; this is a soft boundary, and the last batch may be partial.
- Missing usage or infrastructure errors stop the run for inspection. Sibling
  calls finish and retain their artifacts; a failed partial batch is not adopted.
- Existing run directories are rejected. There is no automatic retry/resume that
  could replay paid calls or uncertain solver executions.

Outputs include request/response receipts, `llm/ledger.sqlite`, per-batch prompts
and evaluations, `state.json`, `champion.cpp`, `frozen.json`, and `run_status.json`.
Credentials are read from the environment and are not persisted.

## Python integration

```python
from aad.greedy_bon.engine import run_greedy

best = await run_greedy(
    problem, generate, evaluate, output_dir, n=4, rounds=100
)
```

`generate(messages, role)` is an async callable. `evaluate(code)` is synchronous
and returns `{"valid": bool, "combined_score": float}`; scores must be maximization
utilities. Custom evaluators must be thread-safe or serialized by the caller.
Evaluation seed and case selection are the evaluator's responsibility.
Tests are offline and do not call models or official solvers.
