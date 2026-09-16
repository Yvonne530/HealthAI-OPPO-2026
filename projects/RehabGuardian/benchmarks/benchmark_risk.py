"""
benchmark_risk.py — RiskMLP (risk_phaseA.mnn) latency benchmark.
Input : risk_seq [1,20,35]   Outputs: risk_logits [1,3], risk_confidence [1,1]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import (make_session, run_once, measure,
                             save_results, print_stats)

WARMUP = 100
# RiskMLP is ~0.05 ms -- 100 warm-up iterations are only ~5 ms of load, far too
# little to spin the CPU governor up after an idle period. That inverts the
# throttle problem: measured blocks *speed up* (0.399 -> 0.049 ms = 8x drift)
# as the clock ramps. The fix is the opposite of the heavy models: no cooldown
# and a fat warm-up (2000 iters ~ 0.1 s of load) so the clock is boosted when
# timing starts, plus short gaps -- 250 x 0.05 ms bursts cannot trip the
# package power limit, so the gaps only need to keep the clock up, not shed heat.
WARMUP_ITERS = 2000
BLOCK_GAP_S = 5
# 1,000 iterations = 4 blocks x 250 with a 20 s idle gap (see benchmark_stgcn.py
# and benchmark_utils.measure for why the loop is block-separated). The run is
# flatness-audited and asserted flat before anything is written to results/.
ITERS = 1000

if __name__ == "__main__":
    net, sess = make_session("risk_phaseA.mnn")
    x = np.random.RandomState(42).randn(1, 20, 35).astype(np.float32)

    out = run_once(net, sess, "risk_seq", x, ["risk_logits", "risk_confidence"])
    lg = out["risk_logits"].ravel()
    assert lg.shape[0] == 3, f"unexpected logits shape {out['risk_logits'].shape}"
    assert np.isfinite(lg).all(), "NaN/Inf in Risk output"
    print(f"[Risk] logits = {np.round(lg, 4)}  conf = {float(out['risk_confidence'].ravel()[0]):.4f}")

    lats, diag = measure(
        lambda: run_once(net, sess, "risk_seq", x, ["risk_logits", "risk_confidence"]),
        "Risk", WARMUP_ITERS, ITERS, gap_s=BLOCK_GAP_S)
    fl = diag["flatness"]
    assert fl["flat"], f"host throttled ({fl['drift_ratio']}x) -- re-run cooled/quiet; not committed"
    s = save_results("risk", lats, extra={"protocol": diag["protocol"], "flatness": fl})
    print_stats("RiskMLP (risk_phaseA.mnn, MNN CPU 4-thread)", s)
    print(f"  Protocol      : {diag['protocol']}")
    print(f"  Flatness      : drift {fl['drift_ratio']}x  ({fl['verdict']})")
