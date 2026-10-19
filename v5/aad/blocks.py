"""Strict edits to named blocks; the evaluator is never in the editable genome."""

import ast
import re

MARKER = re.compile(r"^(?:#|//) AAD-(BEGIN|END) ([a-z_]+)$", re.MULTILINE)


def spans(code: str) -> dict[str, tuple[int, int]]:
    result, opened = {}, None
    for match in MARKER.finditer(code):
        kind, name = match.groups()
        if kind == "BEGIN":
            if opened or name in result:
                raise ValueError("Nested or duplicate evolution block")
            opened = name, match.end() + 1
        else:
            if not opened or opened[0] != name:
                raise ValueError("Unmatched evolution block")
            result[name] = opened[1], match.start()
            opened = None
    if opened or not result:
        raise ValueError("Missing or unclosed evolution blocks")
    return result


def extract(code: str) -> dict[str, str]:
    return {name: code[a:b] for name, (a, b) in spans(code).items()}


def apply_blocks(code: str, replacements: dict[str, str], language: str = "python") -> str:
    positions = spans(code)
    if not replacements or set(replacements) - set(positions):
        raise ValueError("Unknown or empty replacement blocks")
    for name, body in replacements.items():
        if not isinstance(body, str) or "# AAD-" in body or "// AAD-" in body:
            raise ValueError("Replacement contains markers or is not text")
    for name, (a, b) in sorted(positions.items(), key=lambda item: item[1][0], reverse=True):
        if name in replacements:
            code = code[:a] + replacements[name].rstrip() + "\n" + code[b:]
    if len(code.encode()) > 100000:
        raise ValueError("Candidate exceeds 100 KB")
    if language == "python":
        ast.parse(code)
    elif language != "cpp":
        raise ValueError("Unsupported candidate language")
    return code
