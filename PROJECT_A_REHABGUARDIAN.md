# Project A · RehabGuardian — Real-Time On-Device ACL Injury Risk Monitoring System

> 端侧 ACL 损伤风险实时监测系统 · 第十九届全国大学生软件创新大赛（SWC2026）· OPPO 手机端侧 AI 参赛项目
> ← [Back to repository home](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/main/README.md)

Developed by Jianyi Jian ([@YvonnePotter](https://github.com/Yvonne530)).
All training, deployment and app commits of this project are preserved under
the Git history of this repository.

## 1. Overview

A single RGB camera estimates human pose in real time, and the phone runs a
complete offline inference chain:

**joint angle estimation → ground reaction force (GRF) prediction → 3-level ACL injury risk classification**

No cloud services. No wearables.

## 2. Problem

ACL (anterior cruciate ligament) injuries are among the most serious sports
injuries. Conventional risk assessment requires laboratory force plates or
wearable sensors — expensive and inaccessible. Goal: real-time risk monitoring
using only a smartphone.

## 3. Architecture

```
Camera2 (60fps) ──► MediaPipe Pose (33 keypoints)
                          │
                          ▼
              ┌───────────────────────┐
              │ ST-GCN                │  (B,5,33,3) → joint_angles(23) + markers(28×3)
              └──────────┬────────────┘
                         ▼
              ┌───────────────────────┐
              │ FNO (+ Learnable Lag) │  (B,20,72) → GRF, next 10 frames (~167ms), ×12 ch
              │ FFT main path /       │
              │ LSTM fallback on NPU  │
              └──────────┬────────────┘
                         ▼
              ┌───────────────────────┐
              │ RiskMLP + FSM         │  (B,20,35) → 3-level risk + confidence + reasons
              │ hysteresis/EWMA/HR    │
              └───────────────────────┘
```

Key design decisions:

- **Learnable temporal lag** — GRF is a force-response signal that lags joint
  motion by tens of milliseconds. A differentiable sigmoid-parameterized lag
  (`lag = sigmoid(lag_raw) × max_lag`) lets the model learn the physical delay
  between visual input and GRF response during training.
- **FNO on-device fallback** — FFT operators have poor mobile-NPU support;
  export switches to an LSTM-compatible branch (`--useOriginRNNImpl`) so all
  supported devices can run the pipeline.
- **ST-GCN structural prior** — joints grouped per the OpenSim gait2392 model,
  with a left/right mirror-symmetry loss (L_sym).

## 4. Models & Training

| Item | Detail |
|---|---|
| Dataset | Camargo2021 (AddBiomechanics open dataset), 20 subjects / 120 `.b3d` trials |
| Features | 23 joint angle/velocity/acceleration dims, bilateral GRF (Fx,Fy,Fz,Mx,My,Mz), CoM ×3 |
| Labels | 3-level risk labels from a physics-rule-based TeacherLabeler |
| Pipeline | `preprocess.py` (.b3d → HDF5) → `train_v2.py` (three-model joint training) → ONNX export |

Training code: [`feat/st-gcn`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/feat/st-gcn)
(`models/stgcn.py`, `models/fno.py`, `models/risk_model.py`, `train_v2.py`,
`preprocess.py`, `export/`, checkpoints).

## 5. On-Device Deployment

Conversion chain: **PyTorch (.pth) → ONNX (opset 11) → MNNConvert**, with
per-layer numerical consistency checks. Models ship inside the Android app's
assets (`.mnn`, 2.80 MB total). App v1 uses a Kotlin inference-engine wrapper;
App v2 connects to the MNN C++ API directly via JNI.

Deployment package: [`ABtest`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/ABtest) —
benchmark scripts, Kotlin reference implementation,
[file checksums](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/ABtest/MNN_CHECKSUMS.json),
and the [MNN Android integration guide](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/ABtest/MNN_ANDROID_INTEGRATION_GUIDE.md).

## 6. Benchmark（实测）

Device: OPPO Reno15 Pro (arm64-v8a) · MNN 2.9.0 · 1000 consecutive inference runs.

| Model | Avg latency | P95 | Max |
|---|---|---|---|
| STGCN | 0.77 ms | 1.38 ms | 3.64 ms |
| FNO (LSTM path) | 0.48 ms | 0.73 ms | 2.03 ms |
| Risk | 0.09 ms | 0.15 ms | 2.56 ms |
| **Three-model serial** | **1.34 ms** | **2.26 ms** | 8.23 ms |

Scope — important:

- These figures measure **three-model forward-pass latency only**.
- They **exclude** Camera2 capture and MediaPipe pre-processing, and are
  therefore **not** end-to-end app latency.
- Device and MNN version are documented in
  [测试.md](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/main/%E6%B5%8B%E8%AF%95.md);
  the latency table is as recorded in this repository's project documentation.
  <!-- TODO: attach raw benchmark logs / scripts output -->

Model sizes (per project documentation): STGCN ≈0.95 MB · FNO LSTM ≈1.75 MB ·
Risk ≈0.10 MB · total ≈2.80 MB. Note: the checksummed MNN delivery package on
the [`ABtest`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/ABtest)
branch records an earlier Phase-A build
([MNN_CHECKSUMS.json](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/ABtest/MNN_CHECKSUMS.json):
`fno_lstm_phaseA` ≈1.66 MB, `risk_phaseA` ≈0.014 MB), so exact sizes vary by build.

## 7. Testing

- App-layer unit tests: **66/66 passed** — see
  [测试.md](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/main/%E6%B5%8B%E8%AF%95.md);
  development log: [开发.md](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/main/%E5%BC%80%E5%8F%91.md)
- [ONNX export consistency test report](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/feat/st-gcn/ONNX_TEST_REPORT.md)

## 8. Repository / Branch Navigation

| Branch | Content |
|---|---|
| [`main`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/main) | Portfolio docs (this README, dev/test documents) |
| [`feat/st-gcn`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/feat/st-gcn) | Full training pipeline: models, trainers, preprocessing, export, checkpoints |
| [`ABtest`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/ABtest) | MNN delivery package: benchmarks, integration guide, checksums |
| [`android-app`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/android-app) | Android Demo App v1 (MNNInferenceEngine.kt, sliding-window buffer, CameraX + MediaPipe rendering) |
| [`feat/android_app_two`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/feat/android_app_two) | App v2 (JNI-direct MNN, feature pipeline, explainable risk output, GRF curve, PDF report, Room persistence) |
| [`Yvonne530-upload-1`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/Yvonne530-upload-1) | Technical research report ([Technical documentation.md](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/Yvonne530-upload-1/Technical%20documentation.md)) |
| `data/raw-screenshots`, `feat/refined-seed-data`, `fix/remove-background-distractions` | Early data-preparation iterations (Jan 2026) |
| `initial_release`, `feat/kaggle-training-pipeline`, `test/smoke-t4-config` | Early model experiments (LoRA v0.1 → superseded by the ST-GCN route), Kaggle T4 training |
| `feat/prompt-engineer`, `技术方案迭代` | Design documents and OpenSim gait2392 model |

The early-experiment branches are kept intentionally: they show the real
iteration from a LoRA vision baseline (Jan 2026) to the final ST-GCN/FNO
biomechanics pipeline (Mar 2026).

## 9. Quick Start

```bash
# Training (PC / Kaggle)
git checkout feat/st-gcn
pip install -r requirements.txt   # torch, h5py, nimblephysics ...
python preprocess.py              # .b3d → HDF5
python train_v2.py                # three-model joint training
python export/export_onnx.py      # ONNX export + consistency checks

# On-device integration (Android)
git checkout android-app          # or feat/android_app_two
./gradlew assembleDebug           # assets include stgcn.mnn / fno_lstm.mnn / risk.mnn
```

## 10. Demo & Screenshots

Screenshots / demo video: coming soon.

## 11. Documentation

- [Technical documentation.md](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/Yvonne530-upload-1/Technical%20documentation.md)
- [开发.md](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/main/%E5%BC%80%E5%8F%91.md) /
  [测试.md](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/main/%E6%B5%8B%E8%AF%95.md)
- [MNN_ANDROID_INTEGRATION_GUIDE.md](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/ABtest/MNN_ANDROID_INTEGRATION_GUIDE.md)
