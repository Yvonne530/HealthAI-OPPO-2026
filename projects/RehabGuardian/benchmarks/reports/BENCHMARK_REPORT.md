# RehabGuardian PC Benchmark Report

## 1. Hardware / Software

| Item | Value |
|---|---|
| Host PC | Lenovo laptop (Windows 11) |
| CPU backend | MNN 3.6.1 (Python binding), CPU, **4 threads** — mirrors Android engine config |
| GPU reference | PyTorch 2.10 + CUDA available on host (not used in this run; see Limitations) |
| Models | `stgcn_phaseA.mnn`, `fno_lstm_phaseA.mnn`, `risk_phaseA.mnn` (FP32 STGCN / FP16 FNO+Risk mixed precision) |
| Methodology | Warm-up 100 → 5000 timed iterations each; per-iteration `perf_counter` latency |

## 2. Model-level latency (MNN CPU, 4 threads)

| Model | Input | Output | Mean (ms) | P50 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|
| ST-GCN | visual_seq [1,5,33,3] | joint_angles [1,23] | **1.28** | 1.13 | 2.40 | 3.31 | 7.43 |
| FNO-LSTM | bio_seq [1,20,72] | grf_seq [1,10,12] | **0.73** | 0.64 | 1.14 | 2.30 | 4.29 |
| RiskMLP | risk_seq [1,20,35] | risk_logits [1,3] + confidence [1,1] | **0.13** | 0.09 | 0.20 | 0.96 | 3.14 |

## 3. Full Pipeline latency (ST-GCN → bio_seq → FNO → risk_seq → Risk, incl. feature building)

| Metric | Value (ms) |
|---|---|
| Mean | **2.26** |
| P50 | 2.01 |
| P95 | 3.72 |
| P99 | 5.00 |
| Min / Max | 1.71 / 7.71 |

**Answer to the key question: the three-model inference pipeline latency is ≈ 2.3 ms mean (P95 ≈ 3.7 ms, P99 ≈ 5.0 ms) on PC CPU with the same MNN CPU backend / 4-thread configuration used on the phone.**

## 4. Correctness checks (performed in the same runs)

- ST-GCN output: 23 dims, all finite ✅
- FNO output: [1,10,12] = 120 dims, all finite ✅
- Risk output: 3 logits + 1 confidence, all finite ✅
- I/O shapes exactly match `RGPhaseAEngine.kt` tensor contract ✅

## 5. APK build verification

`gradlew assembleDebug` (JDK 17, AGP 8.3.2, Gradle 8.13) **BUILD SUCCESSFUL**; per-ABI debug APKs produced:

```
app/build/outputs/apk/debug/app-arm64-v8a-debug.apk      ← target for Reno15 Pro
app/build/outputs/apk/debug/app-armeabi-v7a-debug.apk
app/build/outputs/apk/debug/app-x86-debug.apk
app/build/outputs/apk/debug/app-x86_64-debug.apk
```

## 6. Limitations

- PC CPU latency is **not** a direct predictor of on-device latency. Cross-device values are reported for reference only; they are not intended as a direct hardware performance comparison.
- This run benchmarks the model-level pipeline (as the user's earlier 1.34 ms / 2.26 ms Python benchmark did). The end-to-end camera-to-UI latency (previously observed 40–110 ms) is dominated by MediaPipe pose detection, preprocessing, and UI — measured separately on device.
- GPU (CUDA) reference was not run in this pass; benchmark scripts can be extended.

## 7. Reproducibility

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

## 8. Contract & NaN/Inf validation (same session as benchmarks)

Validated by `benchmark_validate.py` (deterministic seed-42 inputs, production models):
all 3 models PASS — I/O names/shapes exactly match `RGPhaseAEngine.kt`; `nan_inf_count = 0` everywhere.
Evidence: `results/model_contract.json`.
