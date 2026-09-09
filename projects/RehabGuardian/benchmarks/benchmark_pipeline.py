"""
benchmark_pipeline.py — End-to-end three-model pipeline benchmark (ST-GCN -> FNO -> RiskMLP),
mirroring RGPhaseAEngine.infer() hot path on MNN CPU (4 threads), including the
feature-building steps (bio_seq, risk_seq) as done in RGFeatureBuilder.

Warm-up 100 -> 5000 timed iterations -> P50/P95/P99 -> CSV + summary.json
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import make_session, run_once, bench, save_results, print_stats, RESULTS_DIR, stats

WARMUP = 100
ITERS = 5000

if __name__ == "__main__":
    stgcn_net, stgcn_sess = make_session("stgcn_phaseA.mnn")
    fno_net, fno_sess = make_session("fno_lstm_phaseA.mnn")
    risk_net, risk_sess = make_session("risk_phaseA.mnn")

    visual_seq = np.random.randn(1, 5, 33, 3).astype(np.float32)

    def pipeline():
        # §3.2 STGCN → joint_angles [1,23]  (1 chunk representative run)
        stgcn_out = run_once(stgcn_net, stgcn_sess, "visual_seq", visual_seq, ["joint_angles"])
        ja = stgcn_out["joint_angles"].reshape(23)
        # §3.3 FeatureBuilder: bio_seq [1,20,72] = [ja(23) | vel(23) | acc(23) | com_vel(3)]
        bio_seq = np.zeros((1, 20, 72), dtype=np.float32)
        bio_seq[:, :, :23] = ja
        bio_seq[:, :, 23:26] = 0.1  # com_vel placeholder as engine does
        # §3.4 FNO → grf_seq [1,10,12]
        fno_out = run_once(fno_net, fno_sess, "bio_seq", bio_seq, ["grf_seq"])
        grf0 = fno_out["grf_seq"].reshape(10, 12)[0]
        # §3.5 risk_seq [1,20,35] = [ja_norm(23) | grf0(12)]
        risk_seq = np.zeros((1, 20, 35), dtype=np.float32)
        risk_seq[:, :, :23] = ja
        risk_seq[:, :, 23:35] = grf0
        # §3.6 Risk → logits + confidence
        return run_once(risk_net, risk_sess, "risk_seq", risk_seq, ["risk_logits", "risk_confidence"])

    out = pipeline()
    lg = out["risk_logits"].ravel()
    assert np.isfinite(lg).all()
    print(f"[Pipeline] risk_logits = {np.round(lg, 4)}")

    t0 = time.perf_counter()
    lats = bench(pipeline, WARMUP, ITERS, "Pipeline")
    s = stats(lats)
    save_results("pipeline", lats)

    print_stats("Full Pipeline (STGCN -> bio_seq -> FNO -> risk_seq -> Risk, MNN CPU 4-thread)", s)
