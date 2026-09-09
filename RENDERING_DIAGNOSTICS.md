# Revised Rendering Diagnostics

## Problem with Current Design

Current design included:
- `visibilityStats` - Not available from MediaPipe PoseLandmarker
- `occlusionRate` - Not available from MediaPipe PoseLandmarker
- `blurRatio` - Not available from MediaPipe PoseLandmarker

These fields were fabricated and should be removed.

## Available Rendering Diagnostics

### From PoseLandmarkerResult

| Field | MediaPipe API | Available? |
|-------|--------------|-----------|
| landmarks | result.landmarks() | YES |
| worldLandmarks | result.worldLandmarks() | YES |
| timestamp | result.timestampNs() | YES |

### From RiskStateMachine

| Field | Available? |
|-------|-----------|
| riskScore | YES |
| riskLabel | YES |
| kneeAngleL | YES |
| kneeAngleR | YES |
| latencyMs | YES |

### Available Visibility Information

The MediaPipe PoseLandmarker provides per-landmark visibility:

```kotlin
fun visible(i: Int) = (lms.getOrNull(i)?.visibility() ?: 0f) > 0.5f
```

This can be used for per-landmark diagnostics, but NOT for aggregate occlusionRate.

## What to Include

### Minimal Rendering Diagnostics

```json
{
  "rendering": {
    "poseResultExists": true,
    "poseListEmpty": false,
    "firstPoseExists": true,
    "landmarkCount": 33,
    "expectedLandmarkCount": 33,
    "visibleLandmarkCount": 25,
    "latencyMs": 12.5
  }
}
```

### Forbidden Fields

| Field | Reason |
|-------|--------|
| visibilityStats | No aggregate visibility metric from MediaPipe |
| occlusionRate | Cannot be computed from MediaPipe PoseLandmarker |
| blurRatio | Cannot be computed from MediaPipe PoseLandmarker |
| droppedFrames | Cannot be reliably tracked with STRATEGY_KEEP_ONLY_LATEST |