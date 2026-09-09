"""
benchmark_risk.py — RiskMLP (risk_phaseA.mnn) latency benchmark.
Input : risk_seq [1,20,35]   Outputs: risk_logits [1,3], risk_confidence [1,1]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import make_session, run_once, bench, save_results, print_stats

WARMUP = 100
ITERS = 5000

if __name__ == "__main__":
    net, sess = make_session("risk_phaseA.mnn")
    x = np.random.randn(1, 20, 35).astype(np.float32)

    out = run_once(net, sess, "risk_seq", x, ["risk_logits", "risk_confidence"])
    lg = out["risk_logits"].ravel()
    assert lg.shape[0] == 3, f"unexpected logits shape {out['risk_logits'].shape}"
    assert np.isfinite(lg).all(), "NaN/Inf in Risk output"
    print(f"[Risk] logits = {np.round(lg, 4)}  conf = {float(out['risk_confidence'].ravel()[0]):.4f}")

    lats = bench(lambda: run_once(net, sess, "risk_seq", x, ["risk_logits", "risk_confidence"]),
                 WARMUP, ITERS, "Risk")
    s = save_results("risk", lats)
    print_stats("RiskMLP (risk_phaseA.mnn, MNN CPU 4-thread)", s)
