"""AHC001 trusted validator; never imports candidate code."""

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .io import digest, read_json, save_json

W = 10000
INTEGER = re.compile(r"[+-]?[0-9]+\Z")


@dataclass(frozen=True)
class Case:
    name: str
    text: str

    @property
    def fingerprint(self):
        return digest(self.text)


def parse_input(text: str) -> list[tuple[int, int, int]]:
    values = list(map(int, text.split()))
    if not values or not 1 <= values[0] <= 200 or len(values) != 1 + values[0] * 3:
        raise ValueError("Malformed AHC001 input")
    points = [tuple(values[i:i + 3]) for i in range(1, len(values), 3)]
    if any(not (0 <= x < W and 0 <= y < W and r > 0) for x, y, r in points):
        raise ValueError("Invalid coordinates or target area")
    if len({(x, y) for x, y, _ in points}) != len(points):
        raise ValueError("Duplicate requested locations")
    return points


def judge(input_text: str, output_text: str) -> dict:
    points = parse_input(input_text)
    tokens = output_text.split()
    if len(tokens) != 4 * len(points) or any(not INTEGER.fullmatch(t) or len(t) > 12 for t in tokens):
        return {"status": "WA", "score": 0, "message": "Expected exactly n integer rectangles"}
    values = list(map(int, tokens))
    rects = [tuple(values[i:i + 4]) for i in range(0, len(values), 4)]
    scores, ratios, aspects = [], [], []
    area_total = 0
    for i, ((x, y, target), (a, b, c, d)) in enumerate(zip(points, rects)):
        if not (0 <= a < c <= W and 0 <= b < d <= W):
            return {"status": "WA", "score": 0, "message": f"Rectangle {i}: bounds or positive area"}
        # Check every overlap, even when a rectangle misses its anchor. The supplied
        # Rust scorer skips part of this check for missing anchors; we enforce the statement.
        for j, (e, f, g, h) in enumerate(rects[:i]):
            if max(a, e) < min(c, g) and max(b, f) < min(d, h):
                return {"status": "WA", "score": 0, "message": f"Rectangles {j}, {i} overlap"}
        area = (c - a) * (d - b)
        area_total += area
        ratio = min(area, target) / max(area, target)
        scores.append(1 - (1 - ratio) ** 2 if a <= x < c and b <= y < d else 0.0)
        ratios.append(ratio)
        aspects.append(max(c - a, d - b) / min(c - a, d - b))
    score = math.floor(1e9 * sum(scores) / len(points) + 0.5)
    return {"status": "AC", "score": score, "message": "",
            "coverage": area_total / W ** 2, "mean_aspect": sum(aspects) / len(points),
            "weak_fraction": sum(s < 0.8 for s in scores) / len(points),
            "worst_ads": sorted(range(len(points)), key=lambda i: scores[i])[:5]}


def synthetic_case(seed: int, n: int | None = None) -> str:
    """Statement distribution; Python RNG, NOT the official Rust seed mapping."""
    rng = random.Random(seed)
    n = n or math.floor(50 * 4 ** rng.random() + 0.5)
    xy, seen = [], set()
    while len(xy) < n:
        pair = (rng.randrange(W), rng.randrange(W))
        if pair not in seen:
            seen.add(pair)
            xy.append(pair)
    cuts = [0] + sorted(rng.sample(range(1, W * W), n - 1)) + [W * W]
    return str(n) + "\n" + "".join(f"{x} {y} {cuts[i+1]-cuts[i]}\n" for i, (x, y) in enumerate(xy))


def prepare_dataset(destination: Path, count: int = 12, sample: Path | None = None) -> dict:
    if count < 2:
        raise ValueError("Need at least 2 cases per split")
    if (destination / "manifest.json").exists():
        raise ValueError("Dataset already exists; choose a new destination")
    manifest = {"problem": "ahc001", "kind": "synthetic-statement-distribution",
                "generator": "Python random; not ALE-Bench public/private seeds", "splits": {}}
    for split, offset in [("train", 0), ("validation", 10000), ("holdout", 20000)]:
        folder = destination / split
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for i in range(count):
            text = synthetic_case(offset + i)
            name = f"synthetic_{offset+i:05d}.txt"
            (folder / name).write_text(text, encoding="utf-8")
            records.append({"file": f"{split}/{name}", "sha256": digest(text), "seed": offset + i})
        if split == "train" and sample:
            text = sample.read_text(encoding="utf-8")
            parse_input(text)
            (folder / "official_sample.txt").write_text(text, encoding="utf-8")
            records.append({"file": f"{split}/official_sample.txt", "sha256": digest(text),
                            "source": "ALE-Bench ahc001/example_input.txt"})
        manifest["splits"][split] = records
    save_json(destination / "manifest.json", manifest)
    return manifest


def load_split(root: Path, split: str) -> list[Case]:
    manifest = read_json(root / "manifest.json")
    all_hashes = [r["sha256"] for records in manifest["splits"].values() for r in records]
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("Duplicate input across splits (data leakage)")
    result = []
    for record in manifest["splits"][split]:
        path = (root / record["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Case escapes dataset root")
        text = path.read_text(encoding="utf-8")
        if digest(text) != record["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {path}")
        parse_input(text)
        result.append(Case(record["file"], text))
    if not result:
        raise ValueError("Empty split")
    return result
