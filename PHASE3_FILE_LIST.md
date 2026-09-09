# Revised Phase 3 Exact File List

## Problem with Current Design

Current Phase 3 file list included:
- `CameraXManager.kt` - Does not exist
- `InferenceView.kt` - Does not exist

These files were assumed to exist but were not verified.

## Verified File List

### Existing Files (AndroidApp Branch)

| File Path | Purpose | Instrumentation Point | New File Required? |
|-----------|---------|----------------------|-------------------|
| `MainActivity.kt` | Main activity | Add diagnostic callbacks | NO |
| `CameraManager.kt` | Camera management | Add frame counters | NO |
| `PoseProcessor.kt` | Pose detection | Add timestamp tracking | NO |
| `RiskStateMachine.kt` | Risk state machine | Add diagnostic state | NO |
| `FrameData.kt` | Frame data model | Add diagnostic fields | NO |
| `RiskRenderer.kt` | Risk rendering | Add latency display | NO |
| `SkeletonRenderer.kt` | Skeleton rendering | Existing | NO |
| `RiskOverlayView.kt` | Risk overlay | Existing | NO |

### Files NOT to Create

| File | Reason |
|------|--------|
| CameraXManager.kt | CameraManager.kt already exists |
| InferenceView.kt | No need for separate view |
| DiagnosticManager.kt | Minimize new architecture |
| DiagnosticState.kt | Minimize new architecture |

## Minimal Instrumentation Points

### 1. CameraManager.kt

```kotlin
class CameraManager(
    // ... existing constructor
) {
    private var cameraFramesReceived = 0
    
    private val onFrame: (Bitmap, Long) -> Unit = { bitmap, timestamp ->
        cameraFramesReceived++
        // existing logic
    }
}
```

### 2. PoseProcessor.kt

```kotlin
class PoseProcessor(
    // ... existing constructor
) {
    private var poseFramesSubmitted = 0
    private var poseResultsReceived = 0
    
    fun processFrame(bitmap: Bitmap, timestampMs: Long) {
        poseFramesSubmitted++
        // existing logic
    }
    
    private val onResult: (PoseLandmarkerResult) -> Unit = { result ->
        poseResultsReceived++
        // existing logic
    }
}
```

### 3. RiskStateMachine.kt

```kotlin
class RiskStateMachine {
    private val latencyHistory = mutableListOf<Double>()
    private var lastUpdateTime = 0L
    
    fun update(/* existing params */, latencyMs: Float) {
        lastUpdateTime = System.currentTimeMillis()
        latencyHistory.add(latencyMs.toDouble())
        
        // Trim to last 100 samples
        if (latencyHistory.size > 100) {
            latencyHistory.removeAt(0)
        }
    }
    
    fun getDiagnosticState(): DiagnosticState {
        val sorted = latencyHistory.sorted()
        val p95Index = (sorted.size * 0.95).toInt().coerceIn(0, sorted.size - 1)
        
        return DiagnosticState(
            diagnosticStatus = currentStatus,
            avgLatencyMs = latencyHistory.average(),
            p95LatencyMs = sorted[p95Index],
            maxLatencyMs = sorted.maxOrNull() ?: 0.0,
            fps = calculateFps()
        )
    }
}
```

## Summary

| Category | Count |
|----------|-------|
| Existing files used | 8 |
| New files required | 0 |
| New architecture layers | 0 |

This approach minimizes changes and only adds instrumentation to existing classes.