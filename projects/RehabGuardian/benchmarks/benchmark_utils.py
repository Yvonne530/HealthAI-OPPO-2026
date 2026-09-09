"""
benchmark_utils.py
Shared helpers for RehabGuardian PC benchmark (MNN CPU backend, mirrors Android config:
4 threads, CPU forward, session reused across runs, pre-allocated I/O tensors).
"""
import os
import time
import json
import statistics
import numpy as np

import MNN

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
NUM_THREADS = 4

# Tensor contract (must match RGPhaseAEngine.kt / ONNX export)
SHAPES = {
    "visual_seq": [1, 5, 33, 3],
    "bio_seq": [1, 20, 72],
    "risk_seq": [1, 20, 35],
}


def make_session(model_name):
    """Create an MNN Interpreter + Session for <MODELS_DIR>/<model_name>.mnn."""
    path = os.path.join(MODELS_DIR, model_name)
    net = MNN.Interpreter(path)
    session = net.createSession({"backend": {"type": "CPU"}, "numThread": NUM_THREADS})
    return net, session


def run_once(net, session, input_name, input_arr, output_names):
    """Single inference: copy input, run session, fetch outputs. Returns dict of outputs."""
    inp = net.getSessionInput(session, input_name)
    t = MNN.Tensor(SHAPES[input_name], MNN.Halide_Type_Float, input_arr, MNN.Tensor_DimensionType_Caffe)
    inp.copyFromHostTensor(t)
    net.runSession(session)
    outs = {}
    for name in output_names:
        ot = net.getSessionOutput(session, name)
        shape = ot.getShape()
        host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.zeros(shape, dtype=np.float32), MNN.Tensor_DimensionType_Caffe)
        ot.copyToHostTensor(host)
        outs[name] = np.asarray(host.getNumpyData()).reshape(shape)
    return outs


def bench(fn, warmup=100, iters=5000, label=""):
    """Warm-up then timed loop. Returns list of per-iteration latencies in ms."""
    for _ in range(warmup):
        fn()
    lats = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


def stats(lats):
    a = np.asarray(lats)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "min": float(a.min()),
    }


def save_results(label, lats, extra=None):
    """Append per-iteration latencies to results/latency.csv and update summary.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "latency.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        if new_file:
            f.write("component,iteration,latency_ms\n")
        for i, v in enumerate(lats):
            f.write(f"{label},{i},{v:.4f}\n")

    s = stats(lats)
    if extra:
        s.update(extra)
    jp = os.path.join(RESULTS_DIR, "summary.json")
    data = json.load(open(jp)) if os.path.exists(jp) else {}
    data[label] = s
    json.dump(data, open(jp, "w"), indent=2)
    return s


def print_stats(label, s):
    print(f"\n{label}")
    print(f"  n   : {s['n']}")
    print(f"  Mean: {s['mean']:.3f} ms   Std: {s['std']:.3f}")
    print(f"  P50 : {s['p50']:.3f} ms")
    print(f"  P95 : {s['p95']:.3f} ms")
    print(f"  P99 : {s['p99']:.3f} ms")
    print(f"  Min : {s['min']:.3f} ms   Max: {s['max']:.3f} ms")
