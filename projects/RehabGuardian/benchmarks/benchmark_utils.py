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
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
NUM_THREADS = 4

# Tensor contract (must match RGPhaseAEngine.kt / ONNX export)
SHAPES = {
    "visual_seq": [1, 5, 33, 3],
    "bio_seq": [1, 20, 72],
    "risk_seq": [1, 20, 35],
}


def _safe_float(v, fallback):
    """Mirror RGNormStats.parseSafeFloat: "NaN"/Inf/unknown -> fallback."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fallback
    return f if np.isfinite(f) else fallback


def load_norm_stats():
    """Load joint_angles mean/std from assets/norm_stats.json.

    Mirrors RGNormStats.kt: missing/NaN entries fall back (mean 0, std 1) and
    std is clamped to >= 1e-6 (RGNormStats.clampStd) to avoid divide-by-zero.
    Returns (mean[23], std[23]) as float32 arrays.
    """
    with open(os.path.join(ASSETS_DIR, "norm_stats.json")) as f:
        d = json.load(f)
    mean = np.array([_safe_float(v, 0.0) for v in d.get("joint_angles_mean", [])][:23],
                    dtype=np.float32)
    std = np.array([_safe_float(v, 1.0) for v in d.get("joint_angles_std", [])][:23],
                   dtype=np.float32)
    assert mean.size == 23 and std.size == 23, "norm_stats.json joint_angles arrays must be [23]"
    std = np.maximum(std, 1e-6, dtype=np.float32)
    return mean, std


def make_session(model_name):
    """Create an MNN Interpreter + Session for <MODELS_DIR>/<model_name>.mnn."""
    path = os.path.join(MODELS_DIR, model_name)
    net = MNN.Interpreter(path)
    session = net.createSession({"backend": {"type": "CPU"}, "numThread": NUM_THREADS})
    return net, session


def run_once(net, session, input_name, input_arr, output_names):
    """Single inference: copy input, run session, fetch outputs. Returns dict of outputs.

    NOTE (memory-safety): the destination numpy buffer is bound to a local
    variable before being wrapped in ``MNN.Tensor``. Passing a *temporary*
    ``np.zeros(...)`` straight into ``MNN.Tensor`` lets CPython free it before
    ``getNumpyData()`` materialises the copy -- MNN does not hold a Python
    reference to the buffer -- so the caller read freed memory and got
    non-reproducible values (``-0.0`` / ``1.5e25`` / ``8.2e33`` / ``NaN``)
    depending on GC timing, instead of the real tensor contents. The buffer must
    stay alive for the whole call, and the data must be read back through
    ``getNumpyData()``: reading the raw buffer directly returns zeros because the
    copy is materialised inside ``getNumpyData()``.
    See benchmarks/reports/VALIDATION_REPORT.md.
    """
    inp = net.getSessionInput(session, input_name)
    t = MNN.Tensor(list(SHAPES[input_name]), MNN.Halide_Type_Float, input_arr,
                   MNN.Tensor_DimensionType_Caffe)
    inp.copyFromHostTensor(t)
    net.runSession(session)
    outs = {}
    for name in output_names:
        ot = net.getSessionOutput(session, name)
        shape = list(ot.getShape())
        buf = np.zeros(shape, dtype=np.float32)   # must outlive getNumpyData()
        host = MNN.Tensor(shape, MNN.Halide_Type_Float, buf,
                          MNN.Tensor_DimensionType_Caffe)
        ot.copyToHostTensor(host)
        outs[name] = np.asarray(host.getNumpyData()).reshape(shape).copy()
    return outs


# ---------------------------------------------------------------------------
# Measurement protocol + thermal audit
# ---------------------------------------------------------------------------
# Measured on the Lenovo i7-14700HX host (Windows "Balanced" power plan, on AC):
# the *same* three-model pipeline code ran 2.264 ms mean when the recorded
# evidence was produced, and 3.62 -> 12.5 ms (a 3.3x ramp, first-500 vs
# last-500 p50) when re-run while the host was in a degraded power state -- with
# no change to the models, shapes, or harness. A sustained loop therefore
# reports the host whenever the host is throttling, and the aggregate mean hides
# it completely. flatness_diag() makes that visible: it splits a finished run
# into blocks and compares their medians.
FLATNESS_BLOCK = 250


def flatness_diag(lats, block=FLATNESS_BLOCK, warn_ratio=1.5):
    """Audit a finished run for thermal drift. Returns a diagnostics dict.

    ``drift_ratio`` is the slowest block median over the fastest. A run whose
    blocks disagree by more than ``warn_ratio`` did not measure the model for its
    whole duration and its mean must not be quoted as a model latency.
    """
    a = np.asarray(lats, dtype=np.float64)
    blocks = [float(np.median(a[s:s + block])) for s in range(0, a.size, block)]
    fastest = min(blocks)
    drift = max(blocks) / fastest if fastest > 0 else float("inf")
    return {
        "block_size": int(block),
        "block_p50_ms": [round(x, 3) for x in blocks],
        "fastest_block_p50_ms": round(fastest, 3),
        "slowest_block_p50_ms": round(max(blocks), 3),
        "drift_ratio": round(drift, 2),
        "flat": bool(drift <= warn_ratio),
        "verdict": ("un-throttled: mean is a valid model latency" if drift <= warn_ratio
                    else f"THROTTLED {drift:.1f}x -- mean reflects the host, not the model"),
    }


def measure(fn, label, warmup=100, iters=1000, block=250, gap_s=20.0,
            cooldown_s=0.0, post_warmup_gap_s=0.0):
    """The single measurement protocol behind every committed number.

    Optional ``cooldown_s`` idle before warm-up (lets a heat-soaked host recover
    its boost state; 0 = off). Optional ``post_warmup_gap_s`` idle after warm-up
    -- the warm-up itself is load and consumes the boost budget, so without a gap
    the first timed block can already be throttled. Then ``iters`` timed
    iterations measured as short blocks of ``block`` iterations separated by
    ``gap_s`` idle seconds. Blocks are short enough that the package power limit
    never engages -- measured on this host, a *continuous* pipeline loop runs
    4.24 ms for its first 500 iterations and then 11-13 ms for every iteration
    after (the limit engages within seconds), so a continuous mean reports the
    laptop, not the models. The gaps let the CPU return to its boost state
    between blocks while the work per iteration is unchanged.

    Returns ``(latencies, diagnostics)``; ``diagnostics['flatness']`` is the
    thermal audit of the finished run -- callers must assert it is flat before
    quoting the mean as a model latency.
    """
    if cooldown_s > 0:
        print(f"  [{label}] cooling down {cooldown_s:g}s ...", flush=True)
        time.sleep(cooldown_s)
    for _ in range(warmup):
        fn()
    if post_warmup_gap_s > 0:
        time.sleep(post_warmup_gap_s)
    lats, blocks = [], []
    nblocks = max(1, -(-iters // block))
    for b in range(nblocks):
        n = min(block, iters - len(lats))
        if n <= 0:
            break
        t_blk = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t_blk.append((time.perf_counter() - t0) * 1000.0)
        lats.extend(t_blk)
        blocks.append(float(np.median(t_blk)))
        print(f"  [{label}] block {b + 1}/{nblocks}: p50={blocks[-1]:.3f} ms", flush=True)
        if b != nblocks - 1:
            time.sleep(gap_s)
    flat = flatness_diag(lats, block=block)
    return lats, {"protocol": f"{nblocks} blocks x <= {block} iters, {gap_s:g}s idle gap",
                  "flatness": flat}


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
    """Append per-iteration latencies to results/<label>_latency.csv and update summary.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{label}_latency.csv")
    with open(csv_path, "w", newline="") as f:
        f.write('"component","iteration","latency_ms"\n')
        for i, v in enumerate(lats):
            f.write(f'"{label}","{i}","{v:.4f}"\n')

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
