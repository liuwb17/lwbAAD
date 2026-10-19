# AAD v5

This directory contains the fixed-policy AAD v5 implementation. Run it from the repository root with `python run_v5.py`, or use `python scripts/search.py` from this directory.

The search supports four parallel candidate requests (`--para 4`), mutation, crossover, one-shot repair, optional planning, and optional parent-specific failure feedback. Use `run --help` for available options.

Prepare public cases before searching. Freeze the selected public incumbent before evaluating private cases:

```bash
python scripts/search.py prepare --help
python scripts/search.py run --help
python scripts/search.py freeze --help
python scripts/search.py private --help
```

See the root [README](../README.md) for dependencies and [evaluation setup](../docs/evaluation.md) for hardware requirements. Tests run with `python -m unittest discover -s tests -q`.

Modules are named by function. Internal protocol identifiers are retained for compatibility; the public method name is AAD v5.
