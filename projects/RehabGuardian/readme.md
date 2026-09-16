# RehabGuardian 🐱

**踝关节康复守护猫** — 全设备端实时康复运动风险监测
On-device real-time rehabilitation risk monitoring: MediaPipe pose → ST-GCN → FNO-LSTM → RiskMLP, all running locally via MNN.

> 目标设备: OPPO Reno15 Pro · 推理引擎: MNN 2.x CPU (4 线程) · 姿态检测: MediaPipe Tasks Vision · 数据库: Room

---

## Engineering Validation

只列出有真实证据支撑的结论，每一条都可以在本仓库中复现：

| Validation | Evidence | Status |
|---|---|---|
| Build reproducibility | `gradlew assembleDebug` → 分 ABI debug APK（arm64-v8a / armeabi-v7a，JDK 17 / Gradle 8.13 / AGP 8.3.2） | ✅ |
| Model-level latency benchmark | 4 组件 × 1,000 measured（4 块 × ≤250，块间空闲间隔），每次运行均通过 flatness 审计（漂移 ≤ 1.34×），原始数据已提交 `benchmarks/results/` | ✅ |
| Tensor contract | `benchmark_validate.py` → `results/model_contract.json`（3/3 PASS，I/O 名称/形状与 `RGPhaseAEngine.kt` 一致） | ✅ |
| NaN / Inf safety | 同一脚本验证，全部输出 `nan_inf_count = 0`（确定性 seed-42 输入） | ✅ |
| PyTorch→ONNX→MNN numerical consistency | 原始训练 checkpoint 不在本仓库中，**尚未实测，不提供编造数字** | ⏳ pending |
| Ground-truth accuracy (MAE/RMSE/F1) | 本仓库无标注数据集 | ⚠️ dataset unavailable |
| Real-device benchmark (Reno15 Pro) | APK 就绪，ADB 实测计划中 | 🔄 planned |

---

## Performance (PC CPU, MNN 3.6.1, 4 threads — 与 Android 端同配置)

| Stage | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| ST-GCN `[1,5,33,3]→[1,23]` | 0.885 ms | 0.695 | 1.802 | 2.419 | 3.976 |
| FNO-LSTM `[1,20,72]→[1,10,12]` | 0.579 ms | 0.426 | 1.499 | 2.352 | 4.216 |
| RiskMLP `[1,20,35]→logits+conf` | 0.084 ms | 0.050 | 0.176 | 0.787 | 2.612 |
| **Full Pipeline**（4×ST-GCN + 特征构建，含 4 次姿态分支） | **3.965 ms** | **3.781** | **5.574** | **6.382** | 7.894 |

**Benchmark Protocol**

```text
Backend   : MNN 3.6.1 (CPU forward, 4 threads — matches RGPhaseAEngine.kt)
Measured  : 1,000 iterations per component, 4 blocks × ≤250, idle gaps between
            blocks; each run flatness-audited before evidence is written
Pipeline  : mirrors RGPhaseAEngine.runPipeline() — 4 × ST-GCN chunks + feature
            building (norm_stats 归一化) + FNO + risk head
Timing    : time.perf_counter() per iteration, session reused
Inputs    : production shapes, deterministic (seed 42 for contract check)
NaN / Inf : 0 occurrences
```

> ⚠️ 上述为 **PC CPU 模型级延迟**，用于验证模型与 pipeline 本身的开销；与真机端到端延迟（含 MediaPipe 姿态检测、预处理、UI）不可直接比较。真机数据见 Validation 表中的 "planned" 项。

---

## Deployment Validation

```text
PyTorch → ONNX → MNN → Android Runtime
   ✓ MNN 模型 I/O 契约自动校验 (3/3 PASS)
   ✓ NaN/Inf 防护三层 (FeatureBuilder / NormStats / Engine try-catch)
   ✓ 混合精度策略: STGCN=FP32 (源头上游，避免误差放大), FNO/Risk=FP16
   ✓ 单次飞行推理 (AtomicBoolean) + 零热路径堆分配
```

验证脚本与原始数据：[`benchmarks/`](benchmarks/README.md) · 完整报告：[`benchmarks/reports/BENCHMARK_REPORT.md`](benchmarks/reports/BENCHMARK_REPORT.md) · 工程验证状态：[`benchmarks/reports/VALIDATION_REPORT.md`](benchmarks/reports/VALIDATION_REPORT.md)

---

## Reproducibility

```bash
cd benchmarks
pip install MNN numpy
python benchmark_stgcn.py       # ST-GCN
python benchmark_fno.py         # FNO-LSTM
python benchmark_risk.py        # RiskMLP
python benchmark_pipeline.py    # 完整 pipeline
python benchmark_validate.py    # 契约 + NaN/Inf 校验
```

所有数字的原始 latency 数据（每个组件 1,000 行 CSV）与统计结果（含 flatness 审计）均已提交，可直接审计。

---

## AI 推理管道

```
Camera (30fps) → MediaPipe PoseLandmarker → 33 关键点 × 20 帧环形缓冲
  ↓ RGPhaseAEngine.infer(frames20)
  ├─ ST-GCN ×4 chunks → jaSeq[20][23]        input [1,5,33,3]  → [1,23]
  ├─ RGFeatureBuilder → bio_seq              [1,20,72] = [ja | vel | acc | com_vel]
  ├─ FNO-LSTM → grf_seq                      [1,20,72] → [1,10,12]
  ├─ RGFeatureBuilder → risk_seq             [1,20,35] = [ja_norm | grf0]
  └─ RiskMLP → risk_logits[1,3] + confidence → 风险等级 (低/中/高) + 解释 + 动作评分
```

## 项目结构

```
app/src/main/java/com/healthai/ankle/
├── inference/   RGPhaseAEngine.kt (核心推理引擎) · RGFeatureBuilder · RGMarkerMapper · RGNormStats
├── db/          Room (SessionEntity / RehabDatabase)
├── ui/          RehabGuardianActivity (CameraX+推理+UI) · PoseOverlayView · RiskChartView · 历史会话 · 最佳动作帧
└── report/      PdfReportBuilder (A4 PDF 报告)
benchmarks/      全部 benchmark 脚本 + 原始结果 + 报告（见 benchmarks/README.md）
```

## 构建

```bash
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-arm64-v8a-debug.apk (Reno15 Pro)
```

## Design Highlights

| 设计决策 | 说明 |
|---|---|
| 单次飞行推理 | `compareAndSet(false,true)` 保证重帧不并发占用 MNN Session，保留最后有效结果回退 |
| 零热路径堆分配 | `init()` 预分配全部 I/O FloatArray，`infer()` 中 `System.arraycopy` 复用 |
| NaN/Inf 三层防护 | FeatureBuilder 清零 → norm_stats 安全解析 → Engine try/catch 回退 |
| 橘猫主题 UI | 阳光奶油橘 · Nunito 字体 · 低/中/高风险 = 🐱/😿/🙀 |

