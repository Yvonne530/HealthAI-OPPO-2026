# RehabGuardian Benchmarks — Audit Trail

All benchmark raw data, statistics, and validation reports live here. Every number
quoted in the main README can be reproduced with the commands below.

## Structure

```
benchmarks/
├── README.md                     ← this file
├── benchmark_stgcn.py            ← ST-GCN latency (100 warm-up + 5,000 measured)
├── benchmark_fno.py              ← FNO-LSTM latency
├── benchmark_risk.py             ← RiskMLP latency
├── benchmark_pipeline.py         ← full 3-model pipeline latency (incl. feature building)
├── benchmark_validate.py         ← tensor contract + NaN/Inf validation
├── benchmark_utils.py            ← shared MNN session / timing / stats helpers
├── results/                      ← raw evidence (committed to git)
│   ├── stgcn_latency.csv         ← 5,000 per-iteration latencies
│   ├── fno_latency.csv
│   ├── risk_latency.csv
│   ├── pipeline_latency.csv
│   ├── summary.json              ← aggregate statistics (mean/P50/P95/P99/max)
│   └── model_contract.json       ← I/O shape contract + NaN/Inf check results
└── reports/
    ├── BENCHMARK_REPORT.md       ← full benchmark report
    └── VALIDATION_REPORT.md      ← engineering validation status (evidence levels)
```

## Reproduce

```bash
pip install MNN numpy
python benchmarks/benchmark_stgcn.py      # ST-GCN
python benchmarks/benchmark_fno.py        # FNO-LSTM
python benchmarks/benchmark_risk.py       # RiskMLP
python benchmarks/benchmark_pipeline.py   # full pipeline
python benchmarks/benchmark_validate.py   # contract + NaN/Inf checks
```

## Protocol

| Item | Value |
|---|---|
| Backend | MNN 3.6.1, CPU forward, **4 threads** (matches `RGPhaseAEngine.kt` config) |
| Warm-up | 100 iterations |
| Measured | 5,000 iterations per component |
| Timing | `time.perf_counter()` per iteration, session reused (no re-init) |
| Inputs | Deterministic random tensors (seed 42 for contract check) at exact production shapes |
| Metrics | Mean / Std / P50 / P95 / P99 / Min / Max |

**Note:** these are PC CPU model-level latencies. They characterize the models and
pipeline itself; they are *not* directly comparable to on-device latencies
(different CPU architecture, thermal/OS conditions). On-device numbers require the
Reno15 Pro benchmark (planned).
