import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from aad.greedy_bon.accounting import AuditedModel, BudgetReached, Ledger, StopRun
from aad.greedy_bon.engine import run_greedy
from aad.greedy_bon.prompts import messages


class GreedyBoNTests(unittest.TestCase):
    def test_n_one_deduplicates_identical_code(self):
        async def scenario(root):
            seen = []
            async def generate(body, role):
                seen.append(role)
                return "```cpp\nint main(){}\n```"
            scores = iter((1, 100))
            best = await run_greedy("task", generate,
                lambda code: {"valid": True, "combined_score": next(scores)},
                root, n=1, rounds=2)
            self.assertEqual(seen, ["initialization", "solution"])
            self.assertEqual(best.score, 1)
            self.assertEqual(best.id, "r000000s0")
        with tempfile.TemporaryDirectory() as root:
            asyncio.run(scenario(root))

    def test_frozen_parent_batch_barrier_and_stable_tie(self):
        async def scenario(root):
            prompts = []
            completed = []
            async def generate(body, role):
                i = len(prompts)
                prompts.append((body, role))
                if i >= 3:
                    self.assertEqual(set(completed), {0, 1, 2})
                await asyncio.sleep((2-i % 3)*0.003)
                return f"```cpp\nint main(){{return {i};}}\n```"
            def evaluate(code):
                i = int(code.split("return ")[1].split(";")[0])
                if i < 3:
                    completed.append(i)
                return {"valid": i != 4, "combined_score": [1, 5, 5, 3, 100, 6][i]}
            best = await run_greedy("task", generate, evaluate, root, n=3, rounds=2)
            self.assertEqual(best.id, "r000001s2")
            self.assertEqual(best.parent_id, "r000000s1")
            self.assertTrue(all(p == prompts[0] for p in prompts[:3]))
            expected = messages("task", "int main(){return 1;}\n")
            self.assertTrue(all(p == (expected, "solution") for p in prompts[3:]))
            self.assertEqual([p[1] for p in prompts[:3]], ["initialization"]*3)
        with tempfile.TemporaryDirectory() as root:
            asyncio.run(scenario(root))

    def test_siblings_drain_before_infrastructure_failure_without_adoption(self):
        async def scenario(root):
            finished = []
            async def generate(body, role):
                await asyncio.sleep(.001)
                return "```cpp\nint main(){}\n```"
            def evaluate(code):
                finished.append(code)
                if len(finished) == 1:
                    raise RuntimeError("infrastructure")
                return {"valid": True, "combined_score": 1}
            with self.assertRaisesRegex(RuntimeError, "infrastructure"):
                await run_greedy("task", generate, evaluate, root, n=4, rounds=1)
            self.assertEqual(len(finished), 4)
            self.assertEqual(json.loads((Path(root)/"state.json").read_text())["round"], -1)
            with self.assertRaises(FileExistsError):
                await run_greedy("task", generate, evaluate, root)
        with tempfile.TemporaryDirectory() as root:
            asyncio.run(scenario(root))

    def test_invalid_initials_retry_and_budget_stop(self):
        async def scenario(root):
            roles = []
            async def generate(body, role):
                roles.append(role)
                if len(roles) > 4:
                    raise BudgetReached()
                if len(roles) <= 2:
                    return "malformed response"
                return "```cpp\nint main(){}\n```"
            best = await run_greedy("task", generate,
                lambda code: {"valid": True, "combined_score": 1}, root, n=2, rounds=4)
            self.assertEqual(roles[:4], ["initialization"]*4)
            self.assertEqual(roles[4:], ["solution"]*2)
            self.assertIsNotNone(best)
            self.assertEqual(json.loads((Path(root)/"state.json").read_text())["stop_reason"], "budget_reached")
        with tempfile.TemporaryDirectory() as root:
            asyncio.run(scenario(root))

    def test_token_budget_includes_cached_initialization_and_unknown_stops(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(Path(root)/"ledger.sqlite", 100, 0)
            call = ledger.allocate("initialization")
            ledger.settle(call, {"prompt_tokens": 80, "completion_tokens": 20,
                                 "prompt_cache_hit_tokens": 70})
            self.assertEqual(ledger.summary()["known_tokens"], 100)
            with self.assertRaises(BudgetReached):
                ledger.allocate("solution")
            unknown = Ledger(Path(root)/"unknown.sqlite", 100, 0)
            call = unknown.allocate("solution")
            unknown.settle(call, None)
            with self.assertRaises(StopRun):
                unknown.allocate("solution")

    def test_mock_transport_preserves_model_usage_and_excludes_key(self):
        with tempfile.TemporaryDirectory() as root:
            config = {"name": "deepseek-flash", "endpoint": "https://example.invalid",
                      "api_key_env": "UNUSED_FOR_MOCK", "max_tokens": 393216,
                      "request_options": {"reasoning_effort": "low"}}
            spec = {"output_dir": root, "models": {"solution": config},
                    "budget": {"tokens": 100, "min_calls": 0, "scope": "all"}}
            def transport(body):
                self.assertNotIn("api_key", body)
                return {"usage": {"total_tokens": 10, "prompt_tokens": 7, "completion_tokens": 3},
                        "choices": [{"finish_reason": "stop", "message": {"content": "test"}}]}
            model = AuditedModel("solution", spec, transport=transport)
            self.assertEqual(asyncio.run(model.request(messages("task"), "initialization")), "test")
            self.assertEqual(model.ledger.summary()["known_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
