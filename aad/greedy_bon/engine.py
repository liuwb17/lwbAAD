"""Sequential rounds; N concurrent children share the frozen incumbent."""
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

from .accounting import BudgetReached, InvalidModelResponse
from .prompts import extract_source, messages
from .storage import write_json


@dataclass(frozen=True)
class Candidate:
    id: str
    code: str
    score: float
    parent_id: str | None = None


async def run_greedy(problem, generate, evaluate, output, *, n=4, rounds=10**9):
    """generate(messages, role) is async; evaluate(code) returns valid/combined_score.

    Score must be a maximization utility (negate minimization objectives).
    Initialization is one batch of N independent, identical from-scratch prompts.
    Only strictly better valid candidates replace the incumbent; slot order breaks
    ties. Failed initial batches retry from scratch under the same paid budget.
    No history, score feedback, donors, repair, meta-evolution or fixed bank.
    """
    if type(n) is not int or n < 1 or type(rounds) is not int or rounds < 1:
        raise ValueError("n and rounds must be positive integers")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"state.json").exists() or (root/"batches").exists():
        raise FileExistsError("Existing search: inspect it; automatic replay is disabled")
    best = None
    archive = []
    write_json(root/"state.json", {"status": "running", "n": n, "round": -1, "incumbent": None})
    for round_id in range(rounds):
        parent = best
        batch = root/"batches"/f"{round_id:06d}"
        write_json(batch/"allocation.json", {"parents": [parent.id if parent else None]*n})
        budget_reached = False

        async def child(slot):
            nonlocal budget_reached
            ident = f"r{round_id:06d}s{slot}"
            prompt = messages(problem, parent.code if parent else None)
            write_json(batch/f"{slot}.prompt.json", prompt)
            try:
                text = await generate(prompt, "solution" if parent else "initialization")
            except BudgetReached:
                budget_reached = True
                write_json(batch/f"{slot}.not_submitted.json", {"reason": "budget_reached"})
                return None
            except InvalidModelResponse as error:
                write_json(batch/f"{slot}.result.json", {"id": ident, "valid": False, "error": str(error)})
                return None
            try:
                code = extract_source(text)
            except ValueError as error:
                write_json(batch/f"{slot}.result.json", {"id": ident, "valid": False, "error": str(error)})
                return None
            # Infrastructure errors propagate; never turn them into algorithm WA.
            result = await asyncio.to_thread(evaluate, code)
            score = float(result["combined_score"])
            valid = bool(result["valid"]) and math.isfinite(score)
            candidate = Candidate(ident, code, score, parent.id if parent else None) if valid else None
            write_json(batch/f"{slot}.result.json", {"id": ident, "code": code, "valid": valid,
                "score": score if math.isfinite(score) else None, "parent_id": parent.id if parent else None,
                "evaluation": result})
            return candidate

        # Drain every sibling before raising, and never adopt a failed partial batch.
        children = await asyncio.gather(*(child(i) for i in range(n)), return_exceptions=True)
        for result in children:
            if isinstance(result, BaseException):
                raise result
        # Preserve the experiment's source deduplication and stable top-eight
        # storage. Only its strongest entry is ever used as a greedy parent.
        seen = {hashlib.sha256(c.code.encode()).hexdigest() for c in archive}
        for candidate in children:
            if candidate is not None:
                digest = hashlib.sha256(candidate.code.encode()).hexdigest()
                if digest not in seen:
                    archive.append(candidate)
                    seen.add(digest)
        archive.sort(key=lambda c: c.score, reverse=True)
        archive = archive[:8]
        best = archive[0] if archive else None
        write_json(root/"state.json", {"status": "running", "n": n, "round": round_id,
                                      "incumbent": asdict(best) if best else None,
                                      "archive": [asdict(c) for c in archive]})
        if budget_reached:
            break
    write_json(root/"state.json", {"status": "search_complete" if best else "no_valid_solution",
        "n": n, "round": round_id, "incumbent": asdict(best) if best else None,
        "archive": [asdict(c) for c in archive],
        "stop_reason": "budget_reached" if budget_reached else "iteration_limit"})
    return best
