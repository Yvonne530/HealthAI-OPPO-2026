"""
benchmark_pipeline.py — End-to-end three-model pipeline benchmark (ST-GCN -> FNO -> RiskMLP),
mirroring RGPhaseAEngine.infer() hot path on MNN CPU (4 threads), including the
feature-building steps (ja_seq, bio_seq, risk_seq) as done in RGFeatureBuilder.

Mirrored engine steps (RGPhaseAEngine.runPipeline + RGFeatureBuilder):
  §3.2  4 x STGCN, 5-frame chunks      -> chunks[4][23] -> jaSeq[20][23] (repeated per chunk)
  §3.3  bio_seq [1,20,72]              = [ja(23) | vel(23) | acc(23) | com_vel(3)]
  §3.4  FNO -> grf_seq [1,10,12], grf0 = first 12 values
  §3.5  risk_seq [1,20,35]             = [ja_normalized(23) | grf0(12)]
  §3.6  Risk -> risk_logits [1,3] + risk_confidence [1,1]

Note: the engine runs STGCN **four** times per inference (one 5-frame chunk each),
and normalizes joint angles with assets/norm_stats.json before the risk head.
Both are reproduced here; an earlier revision of this script ran STGCN once and
fed raw (unnormalized) angles to the risk model, understating the engine's cost.

Warm-up 100 -> 5000 timed iterations -> P50/P95/P99 -> CSV + summary.json
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import (make_session, run_once, measure,
                             save_results, print_stats, load_norm_stats)

WARMUP = 100
# See benchmark_stgcn.py for the COOLDOWN / POST_WARMUP_GAP rationale.
COOLDOWN = 90
POST_WARMUP_GAP = 30
# 1,000 iterations = 4 blocks x 250 with a 20 s idle gap (see benchmark_stgcn.py
# and benchmark_utils.measure). The pipeline is ~6x heavier per iteration than
# ST-GCN alone, so its un-throttled window is even shorter -- a continuous loop
# is unusable here (measured: 4.24 ms for the first 500 iterations, then 11-13 ms
# for every iteration after). The run is flatness-audited and asserted flat.
ITERS = 1000

# RGMarkerMapper indices used by RGFeatureBuilder.comPosition()
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

if __name__ == "__main__":
    stgcn_net, stgcn_sess = make_session("stgcn_phaseA.mnn")
    fno_net,   fno_sess   = make_session("fno_lstm_phaseA.mnn")
    risk_net,  risk_sess  = make_session("risk_phaseA.mnn")

    ja_mean, ja_std = load_norm_stats()

    # ── pre-allocated buffers (engine pre-allocates all I/O outside the hot path) ──
    rng = np.random.RandomState(1234)
    # Synthetic pose stream: scaled so STGCN joint-angle outputs stay in a
    # plausible radian range (|ja| <= ~0.9 rad) and do not trigger degenerate
    # FP behaviour. The inputs are NOT physiological motion, so the risk head's
    # output on them is out-of-distribution and is not a prediction.
    frames20 = (rng.randn(20, 33, 3) * 0.05).astype(np.float32)   # MediaPipe-style [20][33][3]
    chunks_in = np.ascontiguousarray(
        np.stack([frames20[c * 5:(c + 1) * 5] for c in range(4)]).reshape(4, 1, 5, 33, 3)
    )                                                            # [4,1,5,33,3]
    ja_seq   = np.zeros((20, 23), dtype=np.float32)
    vel      = np.zeros((20, 23), dtype=np.float32)
    acc      = np.zeros((20, 23), dtype=np.float32)
    bio_seq  = np.zeros((1, 20, 72), dtype=np.float32)
    risk_seq = np.zeros((1, 20, 35), dtype=np.float32)

    com = frames20[:, [L_HIP, R_HIP, L_SHOULDER, R_SHOULDER], :].mean(axis=1)   # [20,3]
    com_vel = np.zeros((20, 3), dtype=np.float32)
    com_vel[1:] = com[1:] - com[:-1]                             # §3.3 computeComVel

    def pipeline():
        # §3.2  4 x STGCN (5-frame chunks) -> chunks[c][23] -> jaSeq[20][23]
        for c in range(4):
            out = run_once(stgcn_net, stgcn_sess, "visual_seq", chunks_in[c], ["joint_angles"])
            ja_seq[c * 5:(c + 1) * 5] = out["joint_angles"].reshape(23)   # buildJaSeq
        # §3.3  bio_seq = [ja | vel | acc | com_vel]  (in-place: engine does not
        # allocate on the hot path; only the output dicts of run_once are new)
        np.subtract(ja_seq[1:], ja_seq[:-1], out=vel[1:])
        vel[0] = 0.0
        np.subtract(vel[1:], vel[:-1], out=acc[1:])
        acc[0] = 0.0
        bio_seq[0, :, :23]   = ja_seq
        bio_seq[0, :, 23:46] = vel
        bio_seq[0, :, 46:69] = acc
        bio_seq[0, :, 69:72] = com_vel
        # §3.4  FNO -> grf_seq [1,10,12]; grf0 = t=0 slice
        fno_out = run_once(fno_net, fno_sess, "bio_seq", bio_seq, ["grf_seq"])
        grf0 = fno_out["grf_seq"].reshape(10, 12)[0]
        # §3.5  risk_seq = [ja_normalized | grf0]  (RGNormStats.normalizeJointAngle)
        ja_norm = risk_seq[0, :, :23]
        np.subtract(ja_seq, ja_mean, out=ja_norm)
        np.divide(ja_norm, ja_std, out=ja_norm)
        risk_seq[0, :, 23:35] = grf0
        # §3.6  Risk -> logits + confidence
        return run_once(risk_net, risk_sess, "risk_seq", risk_seq, ["risk_logits", "risk_confidence"])

    out = pipeline()
    lg = out["risk_logits"].ravel()
    conf = float(out["risk_confidence"].ravel()[0])
    assert np.isfinite(lg).all()
    print(f"[Pipeline] risk_logits = {np.round(lg, 4)}  conf = {conf:.4f}")

    lats, diag = measure(pipeline, "Pipeline", WARMUP, ITERS,
                         cooldown_s=COOLDOWN, post_warmup_gap_s=POST_WARMUP_GAP)
    fl = diag["flatness"]
    assert fl["flat"], f"host throttled ({fl['drift_ratio']}x) -- re-run cooled/quiet; not committed"
    s = save_results("pipeline", lats,
                     extra={"protocol": diag["protocol"], "flatness": fl})

    print_stats("Full Pipeline (4x STGCN -> bio_seq -> FNO -> risk_seq -> Risk, MNN CPU 4-thread)", s)
    print(f"  Flatness      : drift {fl['drift_ratio']}x  ({fl['verdict']})")
