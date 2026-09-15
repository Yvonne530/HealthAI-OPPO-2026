#!/usr/bin/env python3
"""
generate_validator_baseline.py
===============================
Generates the Python-side baseline values for RGInferenceValidator.kt.

Run from the Python project root:
  python scripts/generate_validator_baseline.py \
    --mnn_dir export/mnn_phasea_android_opt

Output: prints Kotlin constants you paste into RGInferenceValidator.kt

Requirements:
  pip install MNN numpy
"""
import argparse, sys, numpy as np

def lcg_seed_input():
    """
    Matches RGInferenceValidator.buildSeedInput() LCG exactly.
    LCG: s = (s * 6364136223846793005 + 1442695040888963407) & 0x7fffffffffffffff
    """
    s = 42
    MAX = 0x7FFFFFFFFFFFFFFF
    MASK = 0x7FFFFFFFFFFFFFFF
    def next_f():
        nonlocal s
        s = (s * 6364136223846793005 + 1442695040888963407) & MASK
        return max(0.01, min(0.99, s / MAX))
    return np.array([[[next_f() for _ in range(3)] for _ in range(33)] for _ in range(20)], dtype=np.float32)

def run_mnn_pipeline(mnn_dir, seed_input):
    import MNN
    import os

    def load(name):
        path = os.path.join(mnn_dir, name)
        net  = MNN.Interpreter(path)
        sess = net.createSession()
        return net, sess

    stgcn_net, stgcn_sess = load("stgcn_phaseA.mnn")
    fno_net,   fno_sess   = load("fno_lstm_phaseA.mnn")
    risk_net,  risk_sess  = load("risk_phaseA.mnn")

    # STGCN x4 chunks
    ja_seq = np.zeros((20, 23), dtype=np.float32)
    for c in range(4):
        chunk = seed_input[c*5:(c+1)*5]                    # [5,33,3]
        t_in  = stgcn_net.getSessionInput(stgcn_sess, "visual_seq")
        inp   = MNN.Tensor([1,5,33,3], MNN.Halide_Type_Float,
                           chunk.reshape(1,5,33,3).flatten().tolist(),
                           MNN.Tensor_DimensionType_Caffe)
        t_in.copyFromHostTensor(inp)
        stgcn_net.runSession(stgcn_sess)
        t_out = stgcn_net.getSessionOutput(stgcn_sess, "joint_angles")
        ja    = np.array(t_out.getData()).reshape(23)
        for t in range(c*5, (c+1)*5):
            ja_seq[t] = ja

    # bio_seq
    bio_seq = np.zeros((1,20,72), dtype=np.float32)
    prev_vel = np.zeros(23, dtype=np.float32)
    for t in range(20):
        ja  = ja_seq[t]
        vel = ja - ja_seq[t-1] if t>0 else np.zeros(23)
        acc = vel - prev_vel   if t>0 else np.zeros(23)
        lhip  = seed_input[t][23]
        rhip  = seed_input[t][24]
        lsho  = seed_input[t][11]
        rsho  = seed_input[t][12]
        com   = ((lhip+rhip+lsho+rsho)/4)
        com_vel = com - ((seed_input[t-1][23]+seed_input[t-1][24]+
                          seed_input[t-1][11]+seed_input[t-1][12])/4) if t>0 else np.zeros(3)
        bio_seq[0,t,:23]    = ja
        bio_seq[0,t,23:46]  = vel
        bio_seq[0,t,46:69]  = acc
        bio_seq[0,t,69:72]  = com_vel
        prev_vel = vel

    t_in = fno_net.getSessionInput(fno_sess, "bio_seq")
    inp  = MNN.Tensor([1,20,72], MNN.Halide_Type_Float,
                      bio_seq.flatten().tolist(), MNN.Tensor_DimensionType_Caffe)
    t_in.copyFromHostTensor(inp)
    fno_net.runSession(fno_sess)
    t_out = fno_net.getSessionOutput(fno_sess, "grf_seq")
    grf_seq = np.array(t_out.getData()).reshape(10,12)
    grf0    = grf_seq[0]

    # risk_seq (identity norm — using mean=0 std=1 as placeholder)
    risk_seq = np.zeros((1,20,35), dtype=np.float32)
    for t in range(20):
        risk_seq[0,t,:23] = ja_seq[t]          # already normalized in real engine
        risk_seq[0,t,23:] = grf0

    t_in = risk_net.getSessionInput(risk_sess, "risk_seq")
    inp  = MNN.Tensor([1,20,35], MNN.Halide_Type_Float,
                      risk_seq.flatten().tolist(), MNN.Tensor_DimensionType_Caffe)
    t_in.copyFromHostTensor(inp)
    risk_net.runSession(risk_sess)

    logits     = np.array(risk_net.getSessionOutput(risk_sess, "risk_logits").getData()).reshape(3)
    confidence = np.array(risk_net.getSessionOutput(risk_sess, "risk_confidence").getData()).reshape(1)

    e     = np.exp(logits - logits.max())
    probs = e / e.sum()
    risk_score = float(probs[2])

    for net, sess in [(stgcn_net,stgcn_sess),(fno_net,fno_sess),(risk_net,risk_sess)]:
        net.releaseSession(sess)

    return risk_score, grf0[:6].tolist(), grf_seq.flatten().tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnn_dir", required=True)
    args = ap.parse_args()

    print("Generating seed input with LCG (seed=42)...")
    seed = lcg_seed_input()
    print(f"Seed input shape: {seed.shape}, min={seed.min():.4f}, max={seed.max():.4f}")

    print("Running MNN pipeline...")
    risk_score, grf0, _ = run_mnn_pipeline(args.mnn_dir, seed)

    print("\n" + "="*60)
    print("Paste these into RGInferenceValidator.kt companion object:")
    print("="*60)
    print(f"private val BASELINE_RISK_SCORE: Float? = {risk_score:.6f}f")
    grf0_str = ", ".join(f"{v:.6f}f" for v in grf0)
    print(f"private val BASELINE_GRF0: FloatArray? = floatArrayOf({grf0_str})")
    print("="*60)

if __name__ == "__main__":
    main()
