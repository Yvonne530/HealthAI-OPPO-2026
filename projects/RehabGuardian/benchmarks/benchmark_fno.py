"""
benchmark_fno.py — FNO-LSTM (fno_lstm_phaseA.mnn) latency benchmark.
Input : bio_seq [1,20,72]   Output: grf_seq [1,10,12]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import (make_session, run_once, measure,
                             save_results, print_stats)

WARMUP = 100
# See benchmark_stgcn.py for the COOLDOWN / POST_WARMUP_GAP rationale.
COOLDOWN = 90
POST_WARMUP_GAP = 30
# 1,000 iterations = 4 blocks x 250 with a 20 s idle gap (see benchmark_stgcn.py
# and benchmark_utils.measure for why the loop is block-separated). The run is
# flatness-audited and asserted flat before anything is written to results/.
ITERS = 1000

if __name__ == "__main__":
    net, sess = make_session("fno_lstm_phaseA.mnn")
    x = np.random.RandomState(42).randn(1, 20, 72).astype(np.float32)

    out = run_once(net, sess, "bio_seq", x, ["grf_seq"])
    g = out["grf_seq"].ravel()
    assert g.shape[0] == 120, f"unexpected output shape {out['grf_seq'].shape}"
    assert np.isfinite(g).all(), "NaN/Inf in FNO output"
    print(f"[FNO] grf0[0:5] = {np.round(g[:5], 4)}")

    lats, diag = measure(lambda: run_once(net, sess, "bio_seq", x, ["grf_seq"]),
                         "FNO", WARMUP, ITERS,
                         cooldown_s=COOLDOWN, post_warmup_gap_s=POST_WARMUP_GAP)
    fl = diag["flatness"]
    assert fl["flat"], f"host throttled ({fl['drift_ratio']}x) -- re-run cooled/quiet; not committed"
    s = save_results("fno", lats, extra={"protocol": diag["protocol"], "flatness": fl})
    print_stats("FNO-LSTM (fno_lstm_phaseA.mnn, MNN CPU 4-thread)", s)
    print(f"  Protocol      : {diag['protocol']}")
    print(f"  Flatness      : drift {fl['drift_ratio']}x  ({fl['verdict']})")
