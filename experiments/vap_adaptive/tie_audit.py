"""Streaming exact-half ratio audit for pre-experiment event tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def count_ties(rows):
    counts = defaultdict(lambda: defaultdict(lambda: {"ties": 0, "instances": 0, "strict_erased": 0}))
    for row in rows:
        variant = str(row["variant"])
        bin_k = str(row["bin_k"])
        counts[variant][bin_k]["instances"] += 1
        counts[variant][bin_k]["strict_erased"] += int(str(row.get("label_tau_0.5", "0")) == "0")
        if float(row["voiced_ratio"]) == 0.5:
            counts[variant][bin_k]["ties"] += 1
    report = {}
    for variant, bins in counts.items():
        report[variant] = {}
        for bin_k, values in bins.items():
            values = dict(values)
            values["fraction"] = values["ties"] / values["instances"] if values["instances"] else None
            values["strict_erasure"] = values["strict_erased"] / values["instances"] if values["instances"] else None
            values["ge_erased"] = values["strict_erased"] - values["ties"]
            values["ge_erasure"] = values["ge_erased"] / values["instances"] if values["instances"] else None
            values["warning_over_1pct"] = bool(values["fraction"] is not None and values["fraction"] > 0.01)
            report[variant][bin_k] = values
    return report


def audit_csv(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return count_ties(csv.DictReader(handle))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_csv(args.csv_path)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
