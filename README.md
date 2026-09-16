# HealthAI · Mobile On-Device AI Portfolio

> Android · Edge AI · Multimodal AI · On-Device Inference

A real-world development repository containing two mobile AI projects,
preserving their original Git history and development process.

**Developer:** Jianyi Jian (简健怡)  
**GitHub:** [@YvonnePotter](https://github.com/Yvonne530)

🏅 **Award / 奖状:**

![奖状](奖状.jpg)

---

## Projects

### 🏃 Project A — RehabGuardian · Real-Time On-Device ACL Injury Risk Monitoring

An Android app that estimates human pose from a single RGB camera and runs a
complete on-device inference chain offline:
**joint angle estimation → ground reaction force (GRF) prediction → 3-level ACL injury risk classification**.
No cloud, no wearables.

- **Stack:** PyTorch · ST-GCN · FNO (+ learnable temporal lag) · RiskMLP/FSM · ONNX · MNN · Kotlin · CameraX · MediaPipe
- **Audited model benchmark (this repo, main branch)**: MNN 3.6.1 CPU 4-thread
  (same backend/threading as the Android engine), 1,000 measured iterations per
  component in 4 flatness-audited blocks — full engine pipeline (4× ST-GCN +
  feature building + FNO + risk head) **3.97 ms mean / P50 3.78 / P95 5.57 /
  P99 6.38 ms**, un-throttled. Raw per-iteration CSVs, aggregate stats,
  flatness audits and tensor-contract validation (3/3 PASS, zero NaN/Inf) are
  committed under
  [projects/RehabGuardian/benchmarks/](projects/RehabGuardian/benchmarks/README.md).

| | |
|---|---|
| 📱 Verified source on main | [projects/RehabGuardian](projects/RehabGuardian/readme.md) — builds per-ABI debug APKs (`arm64-v8a`, `armeabi-v7a`) with gradlew assembleDebug |
|  Latest release **v1.0.2** | [Download APKs](https://github.com/Yvonne530/HealthAI-OPPO-2026/releases/tag/v1.0.2) — real PhaseA normalization stats + bundled MNN models & MediaPipe pose landmarker (versionCode 3) |
| 📉 Benchmark audit trail | [benchmarks/](projects/RehabGuardian/benchmarks/README.md) - [report](projects/RehabGuardian/benchmarks/reports/BENCHMARK_REPORT.md) - [validation](projects/RehabGuardian/benchmarks/reports/VALIDATION_REPORT.md) - PC pipeline **3.97 ms mean / P95 5.57 / P99 6.38** (MNN CPU 4-thread, 1,000 block-measured iters, flatness-audited, raw CSVs committed) |
| 📖 Project README | [PROJECT_A_REHABGUARDIAN.md](PROJECT_A_REHABGUARDIAN.md) |
| 💻 Source / Branches | [feat/st-gcn](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/feat/st-gcn) (training) - [ABtest](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/ABtest) (MNN deployment) - [android-app](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/android-app) / [feat/android_app_two](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/feat/android_app_two) (Android apps) |
| 📊 Legacy reports (incl. historical on-device run: 1.34 ms avg on Reno15 Pro, 1000 forward passes, scope = three-model forward pass only) | [ONNX export consistency report](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/feat/st-gcn/ONNX_TEST_REPORT.md) - [MNN integration guide](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/ABtest/MNN_ANDROID_INTEGRATION_GUIDE.md) |

🎬 **Demo video / 演示视频:**

![演示视频](演示视频.mp4)


---

### 🤖 Project B — ActuAware (动悟) · Offline Multimodal AI Assistant for Android

A fully offline Android AI assistant powered by **llama.cpp**: streaming chat,
image analysis with pose/skeleton detection, web-search RAG, conversation
history management, and device-adaptive performance tuning.

- **Stack:** llama.cpp (mtmd) · C++ / JNI · GGUF · OpenCL · ML Kit Pose Detection · CameraX · Markwon
- An Android multimodal AI project developed collaboratively,
  with its original development history preserved in the `Android-Debug` branch.
  See the project documentation and Git history for implementation details.

| | |
|---|---|
| 📖 Project README | [PROJECT_B_ACTUAWARE.md](PROJECT_B_ACTUAWARE.md) |
| 💻 Source / Branch | [`Android-Debug`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/Android-Debug) |

---

## Why are two projects in one repository?

This repository began as a long-running mobile health-AI development workspace.
The two projects evolved as separate codebases and are therefore preserved on
separate branches while sharing the same repository history.

This is not an accidental mix of projects — the branch structure is kept as-is
to preserve the real, verifiable development process (103 commits,
2026-01 → present).

## Repository Navigation

| Project | Main docs | Code | Documentation |
|---|---|---|---|
| RehabGuardian | `main` + feature branches | [`feat/st-gcn`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/feat/st-gcn) · [`ABtest`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/ABtest) · [`android-app`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/android-app) · [`feat/android_app_two`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/feat/android_app_two) | [Project A README](PROJECT_A_REHABGUARDIAN.md) |
| ActuAware | [`Android-Debug`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/Android-Debug) | [`Android-Debug`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/Android-Debug) | [Project B README](PROJECT_B_ACTUAWARE.md) |

Additional development-process branches (data preparation, early experiments,
design documents) are linked from each project README.

## Development Timeline

```
2026-01   Early data / model experiments (seed datasets, Kaggle training, LoRA v0.1)
2026-03   ST-GCN / FNO training pipeline → ONNX/MNN deployment →
          Android applications (RehabGuardian v1/v2) ‖ ActuAware development
2026+     Technical documentation and refinement
```

## Engineering Highlights

- On-device AI inference with measured real-device latency budgets
- Android native integration (CameraX, MediaPipe, JNI/C++)
- Full deployment chain: PyTorch → ONNX → MNN, with per-layer accuracy checks
- llama.cpp / C++ / JNI integration with GPU-acceleration exploration (Vulkan → OpenCL)
- Real-device performance benchmarking with published methodology and checksums
- Reproducible engineering workflow: data → training → export → integration → testing

## Developer

**Jianyi Jian (简健怡)**  
GitHub: [@YvonnePotter](https://github.com/Yvonne530)

Early-career software engineer focused on:

- Java / Spring Boot backend engineering
- Android development
- Edge AI
- On-device multimodal AI
- Model deployment and optimization