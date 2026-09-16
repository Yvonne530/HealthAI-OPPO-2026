"""
benchmark_stgcn.py — ST-GCN (stgcn_phaseA.mnn) latency benchmark.
Input : visual_seq [1,5,33,3]   Output: joint_angles [1,23]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import (make_session, run_once, measure,
                             save_results, print_stats)

WARMUP = 100
# 90 s idle before warm-up + 30 s after it: on this (Balanced-plan) host the
# boost window is narrow and the warm-up itself is load. The flatness audit
# below asserts the measured blocks agree; if the host throttles the run
# aborts without writing results.
COOLDOWN = 90
POST_WARMUP_GAP = 30
# 1,000 iterations = 4 blocks x 250 with a 20 s idle gap between blocks
# (benchmark_utils.measure). Measured on this host: a *continuous* loop stops
# measuring the model once the package power limit engages (ST-GCN blocks ramp
# ~3x after a few thousand back-to-back iterations). measure() keeps every
# block inside the boost window and flatness-audits the finished run; the
# scripts assert the audit passed, so a throttled run cannot silently produce
# committed numbers. See reports/BENCHMARK_REPORT.md.
ITERS = 1000

if __name__ == "__main__":
    net, sess = make_session("stgcn_phaseA.mnn")
    x = np.random.RandomState(42).randn(1, 5, 33, 3).astype(np.float32)

    # correctness smoke test
    out = run_once(net, sess, "visual_seq", x, ["joint_angles"])
    ja = out["joint_angles"].ravel()
    assert ja.shape[0] == 23, f"unexpected output shape {out['joint_angles'].shape}"
    assert np.isfinite(ja).all(), "NaN/Inf in STGCN output"
    print(f"[STGCN] output[0:5] = {np.round(ja[:5], 4)}")

    lats, diag = measure(lambda: run_once(net, sess, "visual_seq", x, ["joint_angles"]),
                         "STGCN", WARMUP, ITERS,
                         cooldown_s=COOLDOWN, post_warmup_gap_s=POST_WARMUP_GAP)
    fl = diag["flatness"]
    assert fl["flat"], f"host throttled ({fl['drift_ratio']}x) -- re-run cooled/quiet; not committed"
    s = save_results("stgcn", lats, extra={"protocol": diag["protocol"], "flatness": fl})
    print_stats("ST-GCN (stgcn_phaseA.mnn, MNN CPU 4-thread)", s)
    print(f"  Protocol      : {diag['protocol']}")
    print(f"  Flatness      : drift {fl['drift_ratio']}x  ({fl['verdict']})")
