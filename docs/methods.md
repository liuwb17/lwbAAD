# Methods

## AAD v5

AAD v5 generates candidate programs, evaluates them on a public task set, and retains the best observed solution. The fixed policy supports parallel proposals, lineage-based parent selection, mutation, redesign, crossover, and one-shot execution repair. Planning and parent-specific failure feedback are optional.

## RSI

RSI separates program search from policy updates. The inner model proposes programs. The outer DeepSeek Harness agent reviews public search evidence and proposes edits to six Python policy components. Updates are validated before adoption and stored in isolated Git worktrees. Private evaluation evidence is excluded from policy development.

The `greedy` and `diversity` initial policies differ only in parent selection. Greedy selects the incumbent for each child. Diversity samples each parent independently from the population, with replacement. Both start with four requests per round and the same mutation and repair instructions. These are initial policies, not restrictions on later meta updates.

There is no score-based trial period or automatic rollback after a valid policy update. Invalid proposals may receive bounded, charged correction requests; exhausting those retries retains the current validated policy. Unknown request usage and infrastructure failures are not treated as free successful work.

## Budgets and results

`--calls` and `--token-budget` use the underlying search engine's request accounting and stopping rules. Allocated batches and one-shot repairs may finish at a request boundary; the token gate reserves budget before issuing a request. These flags do not mean "spend the entire token budget and then make at least this many generations."

DSH meta usage is recorded separately and is excluded from the default inner token budget. Preserve the configuration and ledger when comparing runs.

RSI imports a verified fixed initial bank and evaluates the imported programs for the selected task. Bank sources, public cases, evaluation settings, and initial-policy choice all affect comparability. LLM sampling and backend changes mean that matching settings do not guarantee identical scores.

## Checkpoints and private evaluation

Use a new run directory for each independent search. Resume an existing search only with `--resume`, the original parameters, and matching source. A `STOP_AFTER_CURRENT` marker requests a checkpoint pause; it does not freeze or privately evaluate a run.

The RSI entry defaults to freezing and evaluating the public incumbent on private cases after normal completion. Use `--no-private-after-run` to stop before terminal evaluation. AAD v5 exposes separate `freeze` and `private` commands. Private results must never guide search or policy updates.
