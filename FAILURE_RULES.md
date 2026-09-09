# Revised Failure Rules

## Problem with Current Design

Current design included:
- `avgLatencyMs > 50 → LATENCY_HIGH` (unapproved performance threshold)

This threshold was not approved and should be removed.

## Revised Failure Detection

### Level 1: Observable Failures

| Condition | Diagnostic State | Action |
|-----------|-----------------|--------|
| CameraManager fails to initialize | FAILED | Show error UI |
| PoseLandmarker fails to initialize | FAILED | Show error UI |
| Pose result is null | FAILED | Log and continue |
| Landmark list is empty | DEGRADED | Log warning |

### Level 2: Processing Failures

| Condition | Diagnostic State | Action |
|-----------|-----------------|--------|
| STGCN output is null | FAILED | Skip frame |
| FNO output is null | FAILED | Skip frame |
| Risk output is null | FAILED | Skip frame |
| Output shape mismatch | FAILED | Log error |

### Level 3: Performance Observations (NO PASS/FAIL THRESHOLDS)

| Metric | Record | Action |
|--------|--------|--------|
| avgLatencyMs | ✅ | Log |
| p95LatencyMs | ✅ | Log |
| maxLatencyMs | ✅ | Log |
| fps | ✅ | Log |

**Important**: Performance metrics are recorded for observation only. No new pass/fail thresholds should be created.

## Status Transitions

```kotlin
enum class DiagnosticStatus {
    UNKNOWN,    // Initial state
    OK,         // All systems operational
    DEGRADED,   // Some systems degraded
    FAILED,     // Critical failure
    SKIPPED     // Processing skipped
}

fun updateStatus(
    cameraOk: Boolean,
    poseOk: Boolean,
    inferenceOk: Boolean,
    degraded: Boolean
): DiagnosticStatus {
    return when {
        !cameraOk || !poseOk || !inferenceOk -> FAILED
        degraded -> DEGRADED
        cameraOk && poseOk && inferenceOk -> OK
        else -> UNKNOWN
    }
}
```

## What NOT to Include

| Rule | Reason for Removal |
|------|-------------------|
| avgLatencyMs > 50 → LATENCY_HIGH | Unapproved threshold |
| fps < 25 → FPS_LOW | Unapproved threshold |
| Any hardcoded latency threshold | Project benchmarks take precedence |