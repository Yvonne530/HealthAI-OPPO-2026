"""
benchmark_validate.py — Model contract & NaN/Inf safety validation.
Loads each production MNN model, verifies the exact tensor contract used by
RGPhaseAEngine.kt (input/output names & shapes), and checks outputs are finite.
Writes real results to results/model_contract.json.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from benchmark_utils import make_session, run_once, RESULTS_DIR

MODELS = [
    ("stgcn_phaseA.mnn", "visual_seq", [1, 5, 33, 3], ["joint_angles"], [[1, 23]]),
    ("fno_lstm_phaseA.mnn", "bio_seq", [1, 20, 72], ["grf_seq"], [[1, 10, 12]]),
    ("risk_phaseA.mnn", "risk_seq", [1, 20, 35], ["risk_logits", "risk_confidence"], [[1, 3], [1, 1]]),
]

results = {"backend": "MNN 3.6.1 CPU 4-thread", "checks": [], "all_pass": True}
for fname, in_name, in_shape, out_names, out_shapes in MODELS:
    net, sess = make_session(fname)
    x = np.random.RandomState(42).randn(*in_shape).astype(np.float32)  # deterministic input
    try:
        out = run_once(net, sess, in_name, x, out_names)
    except Exception as e:
        results["checks"].append({"model": fname, "status": "FAIL", "error": str(e)})
        results["all_pass"] = False
        continue
    shape_ok = all(list(out[n].shape) == s for n, s in zip(out_names, out_shapes))
    finite_ok = all(bool(np.isfinite(out[n]).all()) for n in out_names)
    status = "PASS" if (shape_ok and finite_ok) else "FAIL"
    results["all_pass"] &= (status == "PASS")
    results["checks"].append({
        "model": fname,
        "input_name": in_name, "input_shape": in_shape,
        "output_shapes": {n: list(out[n].shape) for n in out_names},
        "shape_match": shape_ok, "nan_inf_count": int(sum(np.size(out[n]) - np.isfinite(out[n]).sum() for n in out_names)),
        "deterministic_input_seed": 42,
        "sample_output": {n: np.round(out[n].ravel()[:4], 5).tolist() for n in out_names},
        "status": status,
    })
    net.releaseSession(sess) if hasattr(net, "releaseSession") else None

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "model_contract.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps({c["model"]: c["status"] for c in results["checks"]}, indent=2))
print("ALL PASS" if results["all_pass"] else "SOME FAIL")
