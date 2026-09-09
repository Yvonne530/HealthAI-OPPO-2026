"""
benchmark_stgcn.py — ST-GCN (stgcn_phaseA.mnn) latency benchmark.
Input : visual_seq [1,5,33,3]   Output: joint_angles [1,23]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import make_session, run_once, bench, save_results, print_stats

WARMUP = 100
ITERS = 5000

if __name__ == "__main__":
    net, sess = make_session("stgcn_phaseA.mnn")
    x = np.random.randn(1, 5, 33, 3).astype(np.float32)

    # correctness smoke test
    out = run_once(net, sess, "visual_seq", x, ["joint_angles"])
    ja = out["joint_angles"].ravel()
    assert ja.shape[0] == 23, f"unexpected output shape {out['joint_angles'].shape}"
    assert np.isfinite(ja).all(), "NaN/Inf in STGCN output"
    print(f"[STGCN] output[0:5] = {np.round(ja[:5], 4)}")

    lats = bench(lambda: run_once(net, sess, "visual_seq", x, ["joint_angles"]),
                 WARMUP, ITERS, "STGCN")
    s = save_results("stgcn", lats)
    print_stats("ST-GCN (stgcn_phaseA.mnn, MNN CPU 4-thread)", s)
