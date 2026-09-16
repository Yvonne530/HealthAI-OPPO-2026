# RehabGuardian PC Benchmark Report

## 1. Hardware / Software

| Item | Value |
|---|---|
| Host PC | Lenovo laptop (Windows 11) |
| CPU backend | MNN 3.6.1 (Python binding), CPU, **4 threads** — mirrors Android engine config |
| GPU reference | PyTorch 2.10 + CUDA available on host (not used in this run; see Limitations) |
| Models | `stgcn_phaseA.mnn`, `fno_lstm_phaseA.mnn`, `risk_phaseA.mnn` (FP32 STGCN / FP16 FNO+Risk mixed precision) |
| Methodology | **1,000 timed iterations per component, measured as 4 blocks × ≤250 iterations with an idle gap between blocks** (20 s for the heavy models; 5 s for RiskMLP, see below), per-iteration `perf_counter` latency, single session, no re-init. Heavy models idle 90 s before warm-up and 30 s after it (boost recovery); RiskMLP (~0.05 ms/iter) instead gets no cooldown and a fat 2,000-iteration warm-up, because after an idle period its blocks would *speed up* as the clock ramps (measured 0.399 → 0.049 ms, an 8× inverted drift). Every run is audited by `flatness_diag()` — the run is split into 250-iteration blocks and their medians compared — and each script **asserts the run is flat before writing any evidence**. A single sustained loop on this laptop silently starts measuring the cooling solution instead of the model (observed: 3.62 → 12.5 ms over 5,000 back-to-back pipeline iterations, a 3.3× ramp, §4). **All four runs recorded in `results/` are flat — drift ≤ 1.34× — so their means are valid un-throttled model latencies.** |

## 2. Model-level latency (MNN CPU, 4 threads — 1,000 iterations each, block-measured)

| Model | Input | Output | Mean (ms) | P50 | P95 | P99 | Max | Drift |
|---|---|---|---|---|---|---|---|---:|
| ST-GCN | visual_seq [1,5,33,3] | joint_angles [1,23] | **0.885** | 0.695 | 1.802 | 2.419 | 3.976 | 1.08× |
| FNO-LSTM | bio_seq [1,20,72] | grf_seq [1,10,12] | **0.579** | 0.426 | 1.499 | 2.352 | 4.216 | 1.34× |
| RiskMLP | risk_seq [1,20,35] | risk_logits [1,3] + confidence [1,1] | **0.084** | 0.050 | 0.176 | 0.787 | 2.612 | 1.07× |

"Drift" is `flatness.block_p50_ms` slowest block / fastest block, with a 250-iteration
block size. All three are flat (≤ 1.34×), so these are un-throttled measurements and
the means can be quoted directly. Block P50 series are stored per component in
`results/summary.json`.

## 3. Full Pipeline latency (4 × ST-GCN → bio_seq → FNO → risk_seq → Risk, incl. feature building)

| Metric | Value (ms) |
|---|---:|
| Mean | **3.965** |
| P50 | 3.781 |
| P95 | 5.574 |
| P99 | 6.382 |
| Min / Max | 2.823 / 7.894 |

(1,000 iterations in 4 × 250-iteration blocks, `flatness.drift_ratio = 1.21×` → un-throttled.)

**Scope correction.** This pipeline mirrors `RGPhaseAEngine.runPipeline()`: **4 × ST-GCN** (one 5-frame chunk each → `jaSeq[20][23]`), then `bio_seq = [ja|vel|acc|com_vel]`, FNO → `grf0`, `risk_seq = [ja_normalized|grf0]` using `assets/norm_stats.json`, then the risk head. An earlier revision of `benchmark_pipeline.py` ran **one** ST-GCN pass, injected a placeholder `com_vel`, and fed **raw (unnormalized)** joint angles to the risk head. That understated the engine's real per-inference cost, and it is the origin of the previously published 2.26 ms. The corrected numbers are internally consistent with the per-model table above:

```
4 x ST-GCN  +  FNO  +  Risk  =  4(0.885) + 0.579 + 0.084  =  4.203 ms   (sum of component means)
measured pipeline mean                                    =  3.965 ms   (5.6% lower: amortised setup)
```

The superseded 1 × ST-GCN pipeline could not have been consistent with its own per-model numbers (4 × 1.28 + 0.73 + 0.13 = 5.99 ms against a reported 2.26 ms), which is the inconsistency that prompted the correction.

**Answer to the key question: the three-model inference pipeline costs ≈ 4.0 ms mean (P95 ≈ 5.6 ms, P99 ≈ 6.4 ms) on this host, un-throttled, with the same MNN CPU backend / 4-thread configuration used on the phone — including all four ST-GCN passes and the feature-building steps.** Everything remains well inside a 33 ms (30 fps) frame budget, so the on-device cost is dominated by MediaPipe pose detection and preprocessing, not by this pipeline.

## 4. Host power-limit / thermal audit

This host is a Lenovo laptop on the Windows **Balanced** power plan. While re-running the suite it was caught in a degraded power state, and the *same* pipeline code that measured ~4 ms produced this over a *continuous* loop:

```
iter     0-  499 : mean 3.620 ms
iter   500-  999 : mean 9.200 ms      <- no code, model or shape change
iter  3000- 3499 : mean 12.484 ms
```

i.e. a 3.3× ramp purely from sustained load, after only a few seconds. A single
mean over that run would have been reported as "model latency" while actually
measuring the cooling solution. The inverse was also observed for the tiny
RiskMLP after an idle period: measured blocks *speed up* (0.399 → 0.049 ms,
8×) as the CPU clock ramps up from its idle state. Guards added:

1. `flatness_diag()` splits every recorded run into 250-iteration blocks and
   compares their medians. Every benchmark script **asserts the run is flat
   before writing any evidence** — a throttled or clock-ramping run aborts
   instead of committing misleading numbers. The audit is stored under
   `flatness` in `results/summary.json` and can be re-run offline with
   `benchmark_flatness_audit.py`.
2. `measure()` in `benchmark_utils.py` implements the block protocol actually
   used for the committed numbers: 4 blocks × ≤250 timed iterations with idle
   gaps (20 s for the heavy models, which need to shed power-limit debt;
   5 s for RiskMLP, whose bursts cannot trip the limit but whose clock must
   stay up), plus a 90 s cooldown + 30 s post-warm-up gap for the heavy
   models, and a fat 2,000-iteration warm-up for RiskMLP.

The committed evidence is the flat case: drift 1.08× / 1.34× / 1.07× / 1.21×.
The four CSV files are the raw per-iteration data, so this audit can be re-run on
them at any time.

**Note for anyone re-running this suite:** the scripts refuse to commit a
throttled run (assertion aborts before any file is written). If that happens,
give the host time to cool and re-run.

## 5. Harness correctness fix (memory safety in `run_once`)

`benchmark_utils.run_once()` originally wrapped a **temporary** numpy array in the
destination `MNN.Tensor`:

```python
host = MNN.Tensor(shape, MNN.Halide_Type_Float,
                  np.zeros(shape, dtype=np.float32),   # temporary -- freed early
                  MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(host)                             # writes into freed memory
outs[name] = np.asarray(host.getNumpyData()).reshape(shape)
```

MNN does not keep a Python reference to that buffer, so CPython could free it
between the `MNN.Tensor(...)` call and `getNumpyData()`. The output read back was
then uninitialised heap. Symptoms observed with **identical seed-42 inputs**:
`-0.0`, `1.5e25`, `8.2e33`, occasionally `NaN`, and occasionally the correct value —
the result varied with GC timing, and a single warm-up run could look "finite" and
pass a NaN check while being pure garbage.

Fixed by binding the buffer to a local variable so it outlives `getNumpyData()`
(`benchmarks/benchmark_utils.py`). `benchmark_validate.py` now also **repeats the
inference 4× and requires bit-identical output**, so this class of bug fails the
contract check instead of silently passing it.

Verified after the fix: all three models return bit-identical output over 20
consecutive runs, and `max_abs_drift == 0.0` is recorded per model in
`results/model_contract.json`.

## 6. Correctness checks (performed in the same runs)

- ST-GCN output: 23 dims, all finite ✅
- FNO output: [1,10,12] = 120 dims, all finite ✅
- Risk output: 3 logits + 1 confidence, all finite ✅
- I/O shapes exactly match `RGPhaseAEngine.kt` tensor contract ✅
- Repeatability: 4 repeat inferences per model, bit-identical (`max_abs_drift = 0.0`) ✅

## 7. APK build verification

`gradlew assembleDebug` (JDK 17, AGP 8.3.2, Gradle 8.13) **BUILD SUCCESSFUL**; per-ABI debug APKs produced:

```
app/build/outputs/apk/debug/app-arm64-v8a-debug.apk      ← target for Reno15 Pro  (40.9 MB)
app/build/outputs/apk/debug/app-armeabi-v7a-debug.apk                             (34.1 MB)
```

`abiFilters` is restricted to `arm64-v8a` + `armeabi-v7a` because the MNN `jniLibs`
only ship those two ABIs — x86/x86_64 slices would crash at runtime, so they are not
emitted. Current build: `applicationId com.healthai.ankle`, `versionCode 3`,
`versionName 1.0.2`, published as
[release v1.0.2](https://github.com/Yvonne530/HealthAI-OPPO-2026/releases/tag/v1.0.2).

## 8. Limitations

- PC CPU latency is **not** a direct predictor of on-device latency. Cross-device values are reported for reference only; they are not intended as a direct hardware performance comparison.
- This run benchmarks the model-level pipeline (an earlier PC benchmark on this repo quoted 2.26 ms, which pre-dated the 4×ST-GCN scope correction — see §3). The end-to-end camera-to-UI latency (previously observed 40–110 ms) is dominated by MediaPipe pose detection, preprocessing, and UI — measured separately on device.
- GPU (CUDA) reference was not run in this pass; benchmark scripts can be extended.
- The numbers above come from a single un-throttled session on one host. They characterise the models and the pipeline, not a hardware class.

## 9. Reproducibility

```bash
cd benchmarks
pip install MNN numpy
python benchmark_stgcn.py
python benchmark_fno.py
python benchmark_risk.py
python benchmark_pipeline.py
python benchmark_validate.py
```

Raw data: `results/{stgcn,fno,risk,pipeline}_latency.csv` · Aggregates: `results/summary.json` · Contract check: `results/model_contract.json`

## 10. Contract & NaN/Inf validation (same session as benchmarks)

Validated by `benchmark_validate.py` (deterministic seed-42 inputs, production models):
all 3 models PASS — I/O names/shapes exactly match `RGPhaseAEngine.kt`; `nan_inf_count = 0`
everywhere; and each model's output is bit-identical over 4 repeat inferences
(`max_abs_drift = 0.0`).
Evidence: `results/model_contract.json`.
