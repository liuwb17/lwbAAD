"""Dependency-free Chat Completions compatible client with an auditable call ledger."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .blocks import apply_blocks, extract
from .io import read_json, save_json


class BudgetExhausted(RuntimeError):
    pass


def environment(name):
    value = os.environ.get(name)
    # Windows parent processes do not automatically see newly set user variables.
    # Read only the explicitly named variable, never enumerate credentials.
    if not value and os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value = winreg.QueryValueEx(key, name)[0]
        except OSError:
            pass
    return value


def decode_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Unclosed JSON code fence")
        text = "\n".join(lines[1:-1])
    payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("hypothesis"), str):
        raise ValueError("Expected hypothesis and blocks JSON object")
    if not isinstance(payload.get("blocks"), dict):
        raise ValueError("blocks must be a dictionary")
    return payload


class ChatProposer:
    def __init__(self, config, run_dir: Path):
        self.config = config
        self.folder = run_dir / "llm"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.folder / "ledger.json"
        self.ledger = read_json(self.ledger_path) if self.ledger_path.exists() else []
        # One controller per run: a reserved request without a completion belongs
        # to a previous interrupted process. Keep its budget charge, never invent usage.
        for entry in self.ledger:
            if entry["status"] == "started":
                entry["status"] = "interrupted_usage_unknown"
                entry["note"] = "Reserved request interrupted before a complete response; billed token usage unknown"
        if self.ledger:
            save_json(self.ledger_path, self.ledger)
        self.key = environment(config["api_key_env"])
        if not self.key:
            raise ValueError(f"Missing environment variable {config['api_key_env']}")
        self.url = config["base_url"].rstrip("/") + "/chat/completions"
        parsed = urllib.parse.urlparse(self.url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain credentials, query or fragment")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("Use HTTPS, or HTTP only for a loopback model server")
        if not config.get("model"):
            raise ValueError("A model name must be configured")

    @property
    def requests(self):
        return len(self.ledger)

    def check_connection(self):
        """Read-only model inventory; fail early on invalid credentials or model IDs."""
        request = urllib.request.Request(self.config["base_url"].rstrip("/") + "/models",
                                         headers={"Authorization": "Bearer " + self.key})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read(1_000_000))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Model inventory returned HTTP {error.code}; check API key and endpoint") from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Unable to reach model inventory endpoint") from None
        models = [item["id"] for item in payload.get("data", [])]
        if self.config["model"] not in models:
            raise ValueError(f"Configured model unavailable. Available models: {models}")
        return models

    def propose(self, parent, donor, operator, feedback, rng, remaining_requests=None):
        language = self.config.get("language", "python")
        problem = (Path(__file__).parent / "assets/problem.md").read_text(encoding="utf-8")
        if language == "cpp":
            problem = problem.replace("Standard Python library only.", "C++17 standard library only.")
        system = (f"You design optimization algorithms by evolving {language} source code. "
                  "Return only one JSON object with keys hypothesis (string) and blocks "
                  "(object mapping existing block names to complete replacement source strings). "
                  "No markdown, no marker lines inside replacement strings. Preserve signatures. "
                  "Never access files, network, environment, evaluator internals or hidden cases. "
                  "Only stdin/stdout via the supplied skeleton. Deterministic algorithmic randomness. "
                  "Whole algorithm redesign is allowed in the search block; helper functions may be added there.")
        user = {"problem": problem, "operator": operator,
                "local_time_limit_seconds": self.config["timeout"],
                "guidance": {"parameters": "Change schedules or numerical parameters with a reason.",
                             "priority": "Change the ordering or allocation priority policy.",
                             "search": "Introduce a new neighborhood, constructor or acceptance rule.",
                             "crossover": "Synthesize complementary ideas from parent and donor.",
                             "rewrite": "Redesign the solve algorithm; escape the current search family.",
                             "proposal": "Improve proposal geometry, move distribution, or spatial rearrangements.",
                             "conflict": "Improve coupled neighboring-rectangle space transfers; preserve anchors and validity.",
                             "schedule": "Improve annealing schedules, reheating or phases; preserve the passed runtime budget."}[operator],
                "editable_blocks": list(extract(parent)), "parent_source": parent,
                "inspiration_source": donor, "feedback": feedback}
        body = {"model": self.config["model"], "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            "max_tokens": self.config["max_output_tokens"]}
        body.update(self.config.get("request_options", {}))
        # Do not allow additional options to replace the intended prompt or model.
        if any(k in self.config.get("request_options", {}) for k in ("messages", "model", "stream", "max_tokens", "max_completion_tokens")):
            raise ValueError("request_options cannot override messages, model, stream or token budget")
        if self.requests >= self.config["max_requests"]:
            raise BudgetExhausted("LLM request budget exhausted")
        call_id = len(self.ledger)
        save_json(self.folder / f"{call_id:05d}.request.json", body)
        entry = {"id": call_id, "status": "started", "usage": {}}
        self.ledger.append(entry)
        # Reserve before the request: interrupted/failed requests also count.
        save_json(self.ledger_path, self.ledger)
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + self.key})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.config["request_timeout"]) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("Response exceeds 2 MB")
            payload = json.loads(raw)
            save_json(self.folder / f"{call_id:05d}.response.json", payload)
            entry["usage"] = payload.get("usage", {})
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Expected text message content")
            proposal = decode_json(content)
            code = apply_blocks(parent, proposal["blocks"], language=language)
            entry["status"] = "ok"
            return {"code": code, "hypothesis": proposal["hypothesis"],
                    "blocks": list(proposal["blocks"]), "usage": entry["usage"],
                    "requests": 1, "provider": "chat-completions", "call_id": call_id}
        except urllib.error.HTTPError as error:
            entry["status"] = f"http_{error.code}"
            # Do not log provider response bodies that may echo credentials.
            raise RuntimeError(f"Model endpoint returned HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError):
            entry["status"] = "network_error"
            raise RuntimeError("Model request failed or timed out; inspect endpoint connectivity") from None
        except (KeyError, IndexError, TypeError, ValueError, SyntaxError) as error:
            entry["status"] = "invalid_response"
            raise ValueError(f"Invalid model response: {error}") from None
        finally:
            entry["seconds"] = time.monotonic() - started
            save_json(self.ledger_path, self.ledger)
