# Revised Inference Schema

## Problem with Current Design

Current design had:
```json
{
  "inference": {
    "stage": "STGCN",
    "stage": "FNO",
    "stage": "Risk"
  }
}
```

This is invalid JSON (duplicate keys) and doesn't match the actual inference pipeline.

## Revised Inference Schema

```kotlin
data class InferenceState(
    val stgcn: InferenceStage,
    val fno: InferenceStage,
    val risk: InferenceStage
)

data class InferenceStage(
    val status: DiagnosticStatus,
    val latencyMs: Double? = null,
    val outputValid: Boolean = true
)
```

## JSON Representation

```json
{
  "inference": {
    "stgcn": {
      "status": "OK",
      "latencyMs": 12.5,
      "outputValid": true
    },
    "fno": {
      "status": "OK",
      "latencyMs": 18.3,
      "outputValid": true
    },
    "risk": {
      "status": "OK",
      "latencyMs": 8.7,
      "outputValid": true
    }
  }
}
```

## Latency Measurement Points

| Stage | Source File | Measurement Point |
|-------|------------|-------------------|
| stgcn | SlidingWindowBuffer.kt | Window preparation time |
| fno | MNNInferenceEngine.kt | FNO inference call |
| risk | RiskStateMachine.kt | RiskMLP inference call |

## Status Values

- `OK`: Inference completed successfully
- `DEGRADED`: Inference completed but with warnings
- `FAILED`: Inference failed
- `SKIPPED`: Inference was skipped (e.g., insufficient frames)