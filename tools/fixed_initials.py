"""Offline verification and ordered loading of the frozen ALE initial bank.

This is a data reader, not a v5.2 runner integration. It never executes candidates.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

ORIGINAL_BANK_NAME = "ale_fixed_initial_flash_low_20261012"
ORIGINAL_MANIFEST_SHA256 = "d4e41564dd37bafcb05db7f957b185f8e5dd7caf7ec353b0733f54be43c6f2c9"
BANK_NAME = "ale_fixed_initial_public50_ac_20261015"
MANIFEST_SHA256 = "1c0ce69651f675894e958f96d2a79dcf567881434db3986ad280354b55b96784"
TRUSTED_MANIFESTS = {ORIGINAL_MANIFEST_SHA256, MANIFEST_SHA256}
TRUSTED_MANIFESTS.add("d78f18ee7666df5d6653eca362ca58122384fdf68b5d227fe5cb8928be0defe6")
DEFAULT_BANK = Path(__file__).resolve().parents[1] / "initial_populations" / BANK_NAME


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def contained(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe bank path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Bank path escapes root: {relative}")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_bank(bank=DEFAULT_BANK, *, expected_manifest_sha256=None):
    """Verify every manifest file and the identity/order of every selected initial.

    The optional expected hash exists for fixture tests; production uses the
    pinned canonical hash. Never fall back to the first-pass manifest.
    """
    root = Path(bank).resolve()
    digest = sha256(root / "resolved_manifest.json")
    require(digest in TRUSTED_MANIFESTS if expected_manifest_sha256 is None
            else digest == expected_manifest_sha256, "Resolved manifest identity mismatch")
    manifest = read_json(root / "resolved_manifest.json")
    completion = read_json(root / "completion.json")
    protocol = read_json(root / "protocol.json")
    require(manifest["status"] == completion["status"] == "complete", "Incomplete bank")
    require(completion["manifest_sha256"] == digest, "Completion manifest mismatch")
    require(sha256(root / "protocol.json") == manifest["protocol_sha256"], "Protocol hash mismatch")
    require(sha256(root / "manifest.json") == manifest["original_manifest_sha256"], "Original manifest hash mismatch")
    total_bytes = 0
    for relative, expected in manifest["files"].items():
        path = contained(root, relative)
        require(sha256(path) == expected, f"File hash mismatch: {relative}")
        total_bytes += path.stat().st_size
    ids = manifest["problem_ids"]
    require(len(ids) == len(set(ids)) == completion["problem_count"], "Problem count mismatch")
    require(ids == protocol["problem_ids"], "Protocol problem order mismatch")
    require(set(ids) == set(manifest["selected_sources"]) == set(completion["selected_source_hashes"]), "Selected problem set mismatch")
    require(completion["initial_count"] == 4 * len(ids), "Initial count mismatch")
    kinds = Counter()
    for pid in ids:
        task_rel = f"tasks/{pid}/task.json"
        require(manifest["files"][task_rel] == protocol["task_hashes"][pid], f"Task identity mismatch: {pid}")
        task = read_json(contained(root, task_rel))
        require(task["problem_id"] == pid, f"Task id mismatch: {pid}")
        kinds[task["problem_type"]] += 1
        plan_rel = f"tasks/{pid}/plan/artifact.json"
        require(plan_rel in manifest["files"], f"Untracked Plan: {pid}")
        directions = read_json(contained(root, plan_rel))["directions"]
        selected = manifest["selected_sources"][pid]
        require(len(selected) == len(set(selected)) == len(directions) == 4, f"Expected four directions: {pid}")
        hashes = []
        for index, relative in enumerate(selected):
            source_rel = f"{relative}/candidate.cpp"
            meta_rel = f"{relative}/artifact.json"
            require(source_rel in manifest["files"] and meta_rel in manifest["files"], f"Untracked selected source: {relative}")
            meta = read_json(contained(root, meta_rel))
            require(meta["initial_direction"] == index and meta["direction"] == directions[index], f"Direction order mismatch: {relative}")
            require(type(meta["is_randomized"]) is bool, f"Invalid randomness metadata: {relative}")
            require(meta["source_sha256"] == manifest["files"][source_rel], f"Source metadata mismatch: {relative}")
            hashes.append(meta["source_sha256"])
        require(hashes == completion["selected_source_hashes"][pid], f"Completion selection mismatch: {pid}")
    return {"status": "verified", "manifest_sha256": digest, "problem_count": len(ids),
            "initial_count": 4 * len(ids), "problem_types": dict(kinds),
            "manifest_files": len(manifest["files"]), "manifest_bytes": total_bytes,
            "official_evaluation": completion["official_evaluation"]}


def load_initials(problem_id, bank=DEFAULT_BANK):
    """Return all four selected individuals in direction order after full checking.

    Preserve source_bytes for copying/hashing; code is UTF-8 decoded for prompts.
    Callers must record provenance and evaluate under their own fixed protocol.
    """
    root = Path(bank).resolve()
    verify_bank(root)
    manifest = read_json(root / "resolved_manifest.json")
    if problem_id not in manifest["selected_sources"]:
        raise ValueError(f"Problem absent from fixed bank: {problem_id}")
    result = []
    for relative in manifest["selected_sources"][problem_id]:
        meta = read_json(contained(root, f"{relative}/artifact.json"))
        raw = contained(root, f"{relative}/candidate.cpp").read_bytes()
        result.append({**meta, "code": raw.decode("utf-8"), "source_bytes": raw,
                       "bank_relative_path": relative,
                       "bank_manifest_sha256": sha256(root / "resolved_manifest.json")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "export"])
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--problem")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_bank(args.bank), indent=2))
        return
    if not args.problem or args.destination is None:
        parser.error("export requires --problem and --destination")
    if args.destination.resolve().is_relative_to(args.bank.resolve()):
        parser.error("export destination must be outside the immutable bank")
    initials = load_initials(args.problem, args.bank)
    args.destination.mkdir(parents=True, exist_ok=False)
    selection = []
    for index, item in enumerate(initials):
        folder = args.destination / f"initial_{index}"
        folder.mkdir()
        (folder / "candidate.cpp").write_bytes(item["source_bytes"])
        original = contained(args.bank.resolve(), f"{item['bank_relative_path']}/artifact.json")
        (folder / "artifact.json").write_bytes(original.read_bytes())
        selection.append({key: item[key] for key in ("initial_direction", "source_sha256", "bank_relative_path")})
    plan = contained(args.bank.resolve(), f"tasks/{args.problem}/plan/artifact.json")
    (args.destination / "plan.json").write_bytes(plan.read_bytes())
    record = {"problem_id": args.problem, "bank_manifest_sha256": initials[0]['bank_manifest_sha256'],
              "selected_sources": selection, "evaluation_status": "not_evaluated"}
    (args.destination / "selection.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
