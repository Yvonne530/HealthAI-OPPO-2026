"""
benchmark_flatness_audit.py — verify that the committed latency evidence is
un-throttled.

A sustained CPU loop on a laptop can silently start measuring the cooling
solution rather than the model: on the host used for this repository the same
pipeline code measured 2.264 ms mean in one session and ramped 3.62 -> 12.5 ms
in another, with no change to the models or harness. An aggregate mean cannot
distinguish those two cases. This script replays the *stored* per-iteration
latencies from results/*_latency.csv and compares block medians, so the
committed numbers can be shown to be flat (drift ~1x) rather than assumed.

Usage:
    python benchmarks/benchmark_flatness_audit.py
Exit code is non-zero if any committed run shows significant drift.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_utils import RESULTS_DIR, flatness_diag  # noqa: E402

COMPONENTS = ["stgcn", "fno", "risk", "pipeline"]


def load_csv(label):
    path = os.path.join(RESULTS_DIR, f"{label}_latency.csv")
    with open(path, newline="") as f:
        return [float(row["latency_ms"]) for row in csv.DictReader(f)]


def main():
    print(f"{'component':10s} {'iters':>6s} {'drift':>7s}  verdict")
    worst = 0.0
    for label in COMPONENTS:
        lats = load_csv(label)
        d = flatness_diag(lats)
        worst = max(worst, d["drift_ratio"])
        print(f"{label:10s} {len(lats):6d} {d['drift_ratio']:6.2f}x  {d['verdict']}")
        if not d["flat"]:
            print(f"           block p50 (ms): {d['block_p50_ms']}")
    print(f"\nworst drift across committed evidence: {worst:.2f}x")
    if worst > 1.5:
        print("FAIL: at least one committed run drifted; its mean is not a model latency.")
        return 1
    print("PASS: every committed run is flat, so its mean is a valid model latency.")
    return 0


if __name__ == "__main__":
    sys.exit(main())