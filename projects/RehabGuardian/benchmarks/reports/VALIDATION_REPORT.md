# RehabGuardian — Engineering Validation Report

Evidence-level status of every engineering claim. Items marked ✅ have committed,
reproducible evidence in this repository; nothing on this list is asserted without
a script or artifact behind it.

| Validation | Evidence | Status |
|---|---|---|
| Build reproducibility | `gradlew assembleDebug` → per-ABI debug APKs `arm64-v8a` + `armeabi-v7a` (JDK 17 / Gradle 8.13 / AGP 8.3.2; `abiFilters` limited to the ABIs MNN ships) | ✅ |
| Model-level inference latency | 1,000-iteration block-measured MNN benchmark × 4 components, each run flatness-audited (drift ≤ 1.34×); raw CSV + stats in `benchmarks/results/` | ✅ |
| Tensor contract (I/O names & shapes match `RGPhaseAEngine.kt`) | `benchmark_validate.py` → `results/model_contract.json` (3/3 PASS) | ✅ |
| NaN / Inf safety check | Same script; `nan_inf_count = 0` for all 3 models | ✅ |
| PyTorch vs ONNX vs MNN numerical consistency | Original training checkpoints not in this repository — **not yet measured** | ⏳ pending |
| Ground-truth accuracy (MAE/RMSE/Precision/Recall/F1) | Labeled dataset unavailable in this repo | ⚠️ dataset unavailable |
| Real-device benchmark (Reno15 Pro) | Planned — APK is instrumented-ready; ADB-based run pending | 🔄 planned |

## What "PASS" means here

- **Tensor contract:** the exact production `.mnn` models were loaded with MNN 3.6.1
  and executed; input/output tensor names and shapes matched the constants in
  `RGPhaseAEngine.kt` (`visual_seq [1,5,33,3]`, `bio_seq [1,20,72]`,
  `risk_seq [1,20,35]`; outputs `joint_angles [1,23]`, `grf_seq [1,10,12]`,
  `risk_logits [1,3]`, `risk_confidence [1,1]`).
- **NaN/Inf:** every output element finite (`nan_inf_count == 0`), inputs seeded
  deterministically (`RandomState(42)`).

## Honest limitations

1. PC CPU latency does not predict on-device latency; it validates the models and
   pipeline cost under the same MNN CPU 4-thread configuration, not the phone.
2. Numerical-consistency numbers will be published only after re-running the export
   script (`export_stgcn_fp32.py`) with the original checkpoints — no invented `1e-5`.
3. End-to-end camera→UI latency (pose detection + preprocessing + UI) is a separate
   on-device measurement and is not yet reported.
