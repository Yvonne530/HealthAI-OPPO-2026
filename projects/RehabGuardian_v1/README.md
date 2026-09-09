# RehabGuardian v1 — Real-Time On-Device ACL Injury Risk Monitoring

An Android app that estimates human pose from a single RGB camera and runs a complete on-device inference chain offline: **joint angle estimation → ground reaction force (GRF) prediction → 3-level ACL injury risk classification**. No cloud, no wearables.

## Architecture

```
CameraX (640×480) → MediaPipe Pose (33 keypoints) → ST-GCN (joint angles)
  → FNO (GRF prediction) → RiskMLP (SAFE / WARNING / DANGER)
```

- **Pose**: MediaPipe Pose Landmarker Lite (33 keypoints, live streaming mode)
- **ST-GCN**: Temporal graph convolution for 23 joint angles (T=5 window)
- **FNO**: Fourier Neural Operator for GRF left/right (T=20 window)
- **RiskMLP**: 3-class injury risk classifier (SAFE/WARNING/DANGER, T=20 window)
- **Risk FSM**: 10-frame sliding window state machine with hard override rules

## Build & Run

### Prerequisites

- Android Studio Arctic Fox or later
- JDK 17+
- Android SDK 34+
- Target device: Android 8.0 (API 26) or above

### Steps

```bash
# Clone the repository
git clone https://github.com/Yvonne530/HealthAI-OPPO-2026.git
cd HealthAI-OPPO-2026/projects/RehabGuardian_v1

# Build debug APK
./gradlew assembleDebug

# Install on connected device
./gradlew installDebug
```

The APK is located at `app/build/outputs/apk/debug/app-debug.apk`.

### Model Files

The app bundles three MNN models in `app/src/main/assets/`:

| Model | File | Purpose |
|-------|------|---------|
| ST-GCN | `stgcn.mnn` | Joint angle estimation (23 angles) |
| FNO | `fno.mnn` | Ground reaction force prediction (12 outputs) |
| RiskMLP | `risk.mnn` | 3-level ACL injury risk classification |

Normalization stats are loaded from `norm_stats.json` at startup.

## Risk Thresholds

All risk thresholds are centralized in `RiskThresholds.kt`:

| State | Condition |
|-------|-----------|
| SAFE | avg risk ≤ 0.4 |
| WARNING | 0.4 < avg risk ≤ 0.7 |
| DANGER | avg risk > 0.7 |

Hard override rules (safetyCheck) in `MNNInferenceEngine`:

- Knee hyperextension < -5° → DANGER (confidence 0.92)
- GRF > 2.5× body weight → DANGER (confidence 0.85)
- GRF > 1.5× body weight → WARNING (confidence 0.55)

## Key Features

- **Real-time skeleton rendering** on camera preview
- **Risk indicator** with progress bar and color-coded status
- **Latency monitoring** per inference frame
- **Session recording** with frame-by-frame data capture
- **Post-session report** with risk level, anomalies, and recommendations
- **Anatomical validation** — limb length checks, angle clamping [10°, 190°]
- **Per-keypoint confidence filtering** — rejects frames with low-visibility lower-body landmarks

## Optimization History (v1.1)

See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for details on the latest optimization cycle:

1. Unified risk thresholds across all modules (was inconsistent: 0.35 vs 0.4)
2. Added per-keypoint visibility filtering in PoseProcessor
3. Removed simplified GRF proxy, using FNO model's real output
4. Added anatomical validation (limb length, angle clamping)

## Repository Structure

```
app/
  src/main/java/com/erlab/actuaware/
    camera/         — CameraX pipeline, PoseProcessor
    inference/      — MNN models, sliding window, normalization
    analysis/       — Biomechanics, risk state machine, thresholds, report
    ui/             — Session management, risk overlay
    data/           — FrameData, Session, HistoryRecord
    MainActivity.kt — App entry point
  src/main/assets/  — MNN models + normalization stats
```

## License

Internal development project — see parent repository for licensing.
