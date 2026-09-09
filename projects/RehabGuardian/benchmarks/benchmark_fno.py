"""
benchmark_fno.py — FNO-LSTM (fno_lstm_phaseA.mnn) latency benchmark.
Input : bio_seq [1,20,72]   Output: grf_seq [1,10,12]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import make_session, run_once, bench, save_results, print_stats

WARMUP = 100
ITERS = 5000

if __name__ == "__main__":
    net, sess = make_session("fno_lstm_phaseA.mnn")
    x = np.random.randn(1, 20, 72).astype(np.float32)

    out = run_once(net, sess, "bio_seq", x, ["grf_seq"])
    g = out["grf_seq"].ravel()
    assert g.shape[0] == 120, f"unexpected output shape {out['grf_seq'].shape}"
    assert np.isfinite(g).all(), "NaN/Inf in FNO output"
    print(f"[FNO] grf0[0:5] = {np.round(g[:5], 4)}")

    lats = bench(lambda: run_once(net, sess, "bio_seq", x, ["grf_seq"]),
                 WARMUP, ITERS, "FNO")
    s = save_results("fno", lats)
    print_stats("FNO-LSTM (fno_lstm_phaseA.mnn, MNN CPU 4-thread)", s)
