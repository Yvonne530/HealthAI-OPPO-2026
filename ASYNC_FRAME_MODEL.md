# Revised Asynchronous Frame Model

## Problem with Current Design

Current design had:
- `frameId = onPoseResult loop variable` (forbidden)
- No reliable frame-level correlation between camera and pose results

## Revised Frame Tracking Model

### Frame Flow

```
CameraManager (ImageAnalysis)
    │
    ├─ [1] bitmap, timestamp → PoseProcessor.processFrame()
    │
    │   (STRATEGY_KEEP_ONLY_LATEST - frames may be dropped here)
    │
PoseProcessor (LIVE_STREAM)
    │
    ├─ [2] PoseLandmarker.detectAsync(mpImage, timestamp)
    │
    │   (Timestamp preserved through MediaPipe pipeline)
    │
    └─ [3] onResult(landmarks, timestamp) → RehabGuardianActivity
```

### Frame Counters

```kotlin
data class FrameMetrics(
    val cameraFramesReceived: Int,      // [1] increment each onFrame call
    val poseFramesSubmitted: Int,      // [2] increment each processFrame call
    val poseResultsReceived: Int,      // [3] increment each onResult call
    val cameraTimestamp: Long?,         // Current frame timestamp from ImageInfo
    val poseResultTimestamp: Long?      // Same timestamp from MediaPipe result
)
```

### What NOT to Report

| Field | Reason for Exclusion |
|-------|---------------------|
| droppedFrames | STRATEGY_KEEP_ONLY_LATEST makes reliable counting impossible |
| frameId | MediaPipe LIVE_STREAM doesn't support custom frame IDs |

### Timestamp Correlation

The MediaPipe LIVE_STREAM mode preserves the timestamp through the pipeline:

```kotlin
// CameraManager.kt
onFrame(bitmap, proxy.imageInfo.timestamp / 1_000_000)

// PoseProcessor.kt
poseLandmarker?.detectAsync(mpImage, timestampMs)

// onResult
val timestampFromResult = result.timestampNs() / 1_000_000
```

This allows correlation without needing explicit frame IDs.