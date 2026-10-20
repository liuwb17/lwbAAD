"""Audited cache-inclusive token accounting and HTTP transport."""
import asyncio
import json
import os
import sqlite3
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .storage import read_json, write_json


class StopRun(BaseException):
    """Must cross framework 'except Exception' retry/fallback handlers."""


class BudgetReached(StopRun):
    """Known budget boundary, distinct from infrastructure/unknown-usage stops."""


class InvalidModelResponse(ValueError):
    """A charged, settled request produced no usable complete candidate."""


class Ledger:
    def __init__(self, path, token_limit, min_calls, scope="all"):
        self.path = Path(path)
        if token_limit <= 0 or min_calls < 0 or scope not in ("all", "solution"):
            raise ValueError("Explicit positive token limit and valid scope required")
        self.limit, self.minimum, self.scope = token_limit, min_calls, scope
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, role TEXT, "
                       "status TEXT, started REAL, tokens INTEGER, usage TEXT, error TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            with db:
                yield db
        finally:
            db.close()

    def allocate(self, role):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT role,status,tokens FROM calls").fetchall()
            if any(s == "unknown" for _, s, _ in rows):
                raise StopRun("Unknown usage requires inspection; no automatic replay")
            selected = [(s, n) for r, s, n in rows
                        if self.scope == "all" or r in ("solution", "initialization")]
            # Pending calls may finish beyond the boundary. Their usage is unknown,
            # not zero; no retrospective statement of a strict cap is made.
            known = sum(n for _, n in selected if n is not None)
            if known >= self.limit and len(selected) >= self.minimum:
                raise BudgetReached("Token AND minimum-call threshold reached")
            row = db.execute("INSERT INTO calls(role,status,started) VALUES(?,?,?)",
                             (role, "started", time.time()))
            return row.lastrowid

    def exhausted(self):
        with self.connect() as db:
            rows = db.execute("SELECT role,status,tokens FROM calls").fetchall()
        if any(s == "unknown" for _, s, _ in rows):
            raise StopRun("Unknown usage requires inspection")
        selected = [(s, n) for r, s, n in rows
                    if self.scope == "all" or r in ("solution", "initialization")]
        return sum(n or 0 for _, n in selected) >= self.limit and len(selected) >= self.minimum

    def settle(self, call_id, usage, error=None):
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if total is None and isinstance(usage, dict):
            p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if isinstance(p, int) and isinstance(c, int):
                total = p + c  # Cache-hit tokens are already included in prompt_tokens.
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            total = None
        with self.connect() as db:
            db.execute("UPDATE calls SET status=?,tokens=?,usage=?,error=? WHERE id=?",
                       ("unknown" if total is None else ("failed" if error else "complete"),
                        total, json.dumps(usage), error, call_id))
        return total

    def summary(self):
        with self.connect() as db:
            rows = db.execute("SELECT role,status,tokens FROM calls ORDER BY id").fetchall()
        roles = {}
        for role, status, tokens in rows:
            r = roles.setdefault(role, {"calls": 0, "known_tokens": 0, "unknown_calls": 0})
            r["calls"] += 1
            r["known_tokens"] += tokens or 0
            r["unknown_calls"] += tokens is None
        return {"scope": self.scope, "roles": roles, "calls": len(rows),
                "known_tokens": sum(n for _, _, n in rows if n is not None),
                "unknown_calls": sum(n is None for _, _, n in rows)}


class AuditedModel:
    """One saved request/response per generation, with no automatic retries."""
    def __init__(self, role, specification=None, transport=None):
        if specification is None:
            raise ValueError("An explicit model and budget specification is required")
        self.spec = specification
        self.role, self.transport, self.last_usage = role, transport, None
        self.model = self.spec["models"]["solution"]["name"]
        self.root = Path(self.spec["output_dir"])
        budget = self.spec["budget"]
        self.ledger = Ledger(self.root / "llm/ledger.sqlite", budget["tokens"],
                             budget["min_calls"], budget["scope"])

    async def request(self, messages, role=None, **kwargs):
        return await asyncio.to_thread(self._request, messages, role or self.role, kwargs)

    def _request(self, messages, role, kwargs):
        model = self.spec["models"]["solution"]
        body = {"model": model["name"], "messages": messages,
                "max_tokens": kwargs.get("max_tokens") or model["max_tokens"],
                **model.get("request_options", {})}
        call_id = self.ledger.allocate(role)
        folder = self.root / "llm" / f"{call_id:07d}"
        write_json(folder / "request.json", body)
        usage, error = None, None
        try:
            if self.transport:
                response = self.transport(body)
            else:
                key = os.environ[model["api_key_env"]]
                req = urllib.request.Request(model["endpoint"],
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
                with urllib.request.urlopen(req, timeout=model.get("timeout", 600)) as stream:
                    response = json.load(stream)
            write_json(folder / "response.json", response)
            usage = response.get("usage")
            self.last_usage = usage
            choice = response["choices"][0]
            if choice.get("finish_reason", "stop") != "stop" and body["max_tokens"] != 1:
                raise InvalidModelResponse("Incomplete model response")
            content = choice["message"]["content"]
            if body["max_tokens"] == 1 and content is None:
                content = ""  # Preserve native connectivity-only probe semantics.
            if not isinstance(content, str) or (not content.strip() and body["max_tokens"] != 1):
                raise InvalidModelResponse("Empty model response")
            return content
        except Exception as exc:
            # Do not persist exception text: HTTP errors may include credentials.
            error = type(exc).__name__
            raise
        finally:
            total = self.ledger.settle(call_id, usage, error)
            if total is None:
                raise StopRun("Request usage unknown; saved for inspection")
