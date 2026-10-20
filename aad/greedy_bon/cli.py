"""Run greedy sequential best-of-N on a prepared official public task."""
import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import threading

from .accounting import AuditedModel, StopRun
from .engine import run_greedy
from .storage import write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=4, help="Children per round, including initialization (default: 4)")
    p.add_argument("--rounds", type=int, default=10**9, help="Maximum batches, including initialization")
    p.add_argument("--token-budget", type=int, default=7_500_000)
    p.add_argument("--model", default="deepseek-flash")
    p.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--max-output-tokens", type=int, default=393216)
    p.add_argument("--timeout", type=float, default=3600)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.n, args.rounds, args.token_budget, args.max_output_tokens, args.timeout) <= 0:
        raise ValueError("Counts, token limits and timeout must be positive")
    if not os.environ.get(args.api_key_env):
        raise ValueError(f"Set {args.api_key_env} in your environment")
    root = args.run_dir.resolve()
    # Exclusive creation also prevents two controllers from using the same run.
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"run_status.json", {"status": "starting"})
    model = None
    try:
        # Lazy import keeps --help and core offline tests independent of ALE assets.
        from aad.ale_evaluation import AleBatchEvaluator, load_task
        task, cases = load_task(args.task)
        evaluator = AleBatchEvaluator(args.task, root/"measurements")
        write_json(root/"evaluation_runtime.json", evaluator.runtime)
        model_config = {"name": args.model, "endpoint": args.endpoint, "api_key_env": args.api_key_env,
                        "max_tokens": args.max_output_tokens, "timeout": args.timeout,
                        "request_options": {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}}
        spec = {"output_dir": str(root), "models": {"solution": model_config},
                "budget": {"tokens": args.token_budget, "min_calls": 0, "scope": "all"},
                "task": str(args.task.resolve()), "n": args.n, "rounds": args.rounds,
                "public_cases": [c.name for c in cases], "algorithm_seed": 0, "private": "disabled"}
        write_json(root/"spec.json", spec)
        model = AuditedModel("solution", spec)
        # Keep this evaluator instance single-writer; LLM calls remain concurrent.
        evaluation_lock = threading.Lock()

        def evaluate(code):
            with evaluation_lock:
                result = evaluator(code, cases, task["time_limit"], "greedy-bon-public", seed=0)
            if any(c["status"] in ("IE", "INTERNAL_ERROR") for c in result["cases"]):
                raise StopRun("Evaluator infrastructure error; inspect before any recovery")
            return {"combined_score": result["mean"], "raw_mean": result["raw_mean"], "valid": result["valid"]}

        problem = task["statement"] + (
            f"\nExecution: C++17, time limit {task['time_limit']} seconds; "
            f"memory limit {task['memory_limit']}; algorithm seed is AAD_RANDOM_SEED=0. "
            "Use the environment seed for randomness, not wall-clock entropy. "
            "Follow the problem input/output and interaction protocol.")
        write_json(root/"run_status.json", {"status": "search_running"})
        winner = asyncio.run(run_greedy(problem, model.request, evaluate, root, n=args.n, rounds=args.rounds))
        usage = model.ledger.summary()
        if usage["unknown_calls"]:
            raise StopRun("Unsettled usage; inspect before recovery")
        if winner:
            code = winner.code.encode("utf-8")
            (root/"champion.cpp").write_bytes(code)
            write_json(root/"frozen.json", {"source_sha256": hashlib.sha256(code).hexdigest(),
                "public_score": winner.score*(1 if task["direction"]=="maximize" else -1),
                "usage": usage, "private_evaluated": False})
        write_json(root/"run_status.json", {"status": "complete_public_only" if winner else "no_valid_solution",
                                          "usage": usage})
    except BaseException as error:
        write_json(root/"run_status.json", {"status": "stopped_needs_inspection", "type": type(error).__name__,
                                          "usage": model.ledger.summary() if model else None})
        raise
