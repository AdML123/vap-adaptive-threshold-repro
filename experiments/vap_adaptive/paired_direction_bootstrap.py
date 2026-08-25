"""Paired conversation-cluster bootstrap for forward versus reverse labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def f1(rows: list[dict]) -> float:
    counts = {key: sum(int(row.get(key, 0)) for row in rows) for key in ("tp", "fp", "fn")}
    denom = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 2 * counts["tp"] / denom if denom else 0.0


def paired(forward: dict, reverse: dict, replicates: int, seed: int) -> dict:
    f_rows = {str(row["conversation_id"]): row for row in forward["conversation_counts"]}
    r_rows = {str(row["conversation_id"]): row for row in reverse["conversation_counts"]}
    ids = sorted(set(f_rows) & set(r_rows))
    rng = random.Random(seed)
    diffs = []
    for _ in range(replicates):
        sample = rng.choices(ids, k=len(ids))
        diffs.append(f1([r_rows[i] for i in sample]) - f1([f_rows[i] for i in sample]))
    estimate = f1(list(r_rows.values())) - f1(list(f_rows.values()))
    low, high = np.percentile(np.asarray(diffs), [2.5, 97.5])
    return {"estimate_reverse_minus_forward": estimate, "low": float(low), "high": float(high), "replicates": replicates, "conversation_count": len(ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    rows = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text)
        forward = json.loads((root / "adaptive_forward" / str(seed) / "test_metrics.json").read_text(encoding="utf-8"))
        reverse = json.loads((root / "adaptive_reverse" / str(seed) / "test_metrics.json").read_text(encoding="utf-8"))
        rows.append({"seed": seed, **paired(forward, reverse, args.replicates, seed)})
    output = {"comparison": "reverse_minus_forward", "rows": rows, "status": "pass"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
