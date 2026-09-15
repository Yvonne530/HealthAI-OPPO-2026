# RehabGuardian Android  🐱
**踝关节康复守护猫** — 全设备端 AI 运动风险评估

---

## 项目概览

| 属性 | 值 |
|------|-----|
| 目标设备 | OPPO Reno15 Pro (PLV110, 1080×2354, Android 16) |
| 最低 SDK | API 26 (Android 8.0) |
| 推理引擎 | MNN 2.x (CPU, 4 线程) |
| 姿态检测 | MediaPipe Tasks Vision — PoseLandmarker |
| 数据库 | Room (SQLite) |
| 语言 | Kotlin 1.9 |
| 主题 | 阳光奶油橘色 · 长毛橘猫 · Nunito 字体 |

---

## 文件结构

```
RehabGuardian/
├── app/
│   ├── build.gradle                          ← 所有依赖声明
│   ├── proguard-rules.pro
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/
│       │   ├── norm_stats.json               ← 归一化统计（替换为真实值）
│       │   └── models/
│       │       ├── stgcn_phaseA.mnn          ← 从 Python 项目复制
│       │       ├── stgcn_phaseA.mnn.weight
│       │       ├── fno_lstm_phaseA.mnn
│       │       ├── fno_lstm_phaseA.mnn.weight
│       │       ├── risk_phaseA.mnn
│       │       ├── risk_phaseA.mnn.weight
│       │       └── pose_landmarker_lite.task ← MediaPipe 姿态模型
│       ├── java/com/healthai/ankle/
│       │   ├── RehabApplication.kt           ← Application 类
│       │   ├── inference/
│       │   │   ├── RGPhaseAEngine.kt         ← 核心 MNN 推理引擎
│       │   │   ├── RGNormStats.kt            ← 归一化统计加载
│       │   │   ├── RGFeatureBuilder.kt       ← jaSeq/bio_seq/risk_seq 构建
│       │   │   ├── RGMarkerMapper.kt         ← 33→28→84 关键点映射
│       │   │   └── MNNForwardType.kt         ← MNN 前向类型枚举
│       │   ├── db/
│       │   │   ├── SessionEntity.kt          ← Room 实体
│       │   │   └── RehabDatabase.kt          ← DAO + Database
│       │   ├── ui/
│       │   │   ├── CatSplashActivity.kt      ← 橘猫启动屏
│       │   │   ├── RehabGuardianActivity.kt  ← 主界面（CameraX + 推理 + UI）
│       │   │   ├── RehabViewModel.kt         ← ViewModel 封装
│       │   │   ├── PoseOverlayView.kt        ← 姿态骨架叠加层
│       │   │   ├── RiskChartView.kt          ← 实时风险折线图
│       │   │   ├── SessionHistoryActivity.kt ← 历史会话列表
│       │   │   └── BestFrameActivity.kt      ← 最佳动作帧详情
│       │   └── report/
│       │       └── PdfReportBuilder.kt       ← A4 PDF 报告生成
│       └── res/
│           ├── drawable/
│           │   ├── ic_cat_mascot.xml         ← 矢量橘猫吉祥物
│           │   ├── bg_bottom_card.xml
│           │   ├── bg_top_bar.xml
│           │   ├── bg_knee_card.xml
│           │   ├── bg_hint.xml
│           │   ├── btn_orange_pill.xml
│           │   ├── btn_cream_pill.xml
│           │   └── risk_progress.xml
│           ├── font/
│           │   └── nunito_family.xml         ← 字体声明（需下载 TTF）
│           ├── layout/
│           │   └── activity_rehab_guardian.xml
│           └── values/
│               ├── colors.xml
│               ├── strings.xml
│               └── themes.xml
└── build.gradle
```

---

## 快速集成步骤

### 1. 复制 MNN 模型文件
```bash
cp export/mnn_phasea_android_opt/*.mnn* \
   app/src/main/assets/models/
```

### 2. 下载 MediaPipe 姿态模型
```bash
wget -O app/src/main/assets/models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

### 3. 下载 Nunito 字体
从 https://fonts.google.com/specimen/Nunito 下载并放入 `res/font/`：
- `nunito.ttf`
- `nunito_bold.ttf`
- `nunito_semibold.ttf`

### 4. 更新 norm_stats.json
用 Python 项目中真实的 `norm_stats.json` 替换 `assets/norm_stats.json`。

### 5. 构建 APK
```bash
./gradlew assembleDebug
# APK 位于 app/build/outputs/apk/debug/
```

---

## AI 推理管道

```
Camera (30fps)
  ↓ ImageAnalysis (CameraX, STRATEGY_KEEP_ONLY_LATEST)
  ↓ MediaPipe PoseLandmarker (async)
  ↓ 33 关键点 [x,y,z] per frame
  ↓ 20-frame 环形缓冲区
  ↓
RGPhaseAEngine.infer(frames20)
  │
  ├─ §3.2  STGCN × 4 chunks  → jaSeq[20][23]
  │        input:  [1,5,33,3]  visual_seq
  │        output: [1,23]      joint_angles
  │
  ├─ §3.3  RGFeatureBuilder.buildBioSeq  → [1,20,72]
  │        [ja | vel | acc | com_vel]
  │
  ├─ §3.4  FNO  → grf_seq[1,10,12]
  │        input:  [1,20,72]  bio_seq
  │        output: [1,10,12]  grf_seq
  │        grf0 = grf_seq[0,0,:] (12 dims)
  │
  ├─ §3.5  RGFeatureBuilder.buildRiskSeq  → [1,20,35]
  │        [ja_norm(23) | grf0(12)]
  │
  └─ §3.6  Risk  → risk_logits[1,3], risk_confidence[1,1]
           softmax → prob[2] = high-risk score
           argmax  → label (0=低 1=中 2=高)
```

### 性能目标

| 指标 | 目标 | 实测 (Python benchmark) |
|------|------|------------------------|
| 端到端延迟 | < 50ms | 1.34ms avg |
| P95 延迟 | < 15ms | 2.26ms |
| STGCN 精度 | FP16: ±0.08 | ✅ |
| Risk 精度 | ±1e-5 | ✅ |

---

## 关键设计决策

### 单次飞行推理（AtomicBoolean 保护）
```kotlin
if (!inferring.compareAndSet(false, true)) return lastResult  // 丢弃新帧
```
确保重帧不会并发占用 MNN Session，同时保留最后有效结果作为回退。

### 预分配缓冲区（零热路径堆分配）
引擎 `init()` 时预分配所有 I/O FloatArray，`infer()` 中通过 `System.arraycopy` 复用，
满足 < 50ms 的性能目标。

### NaN/Inf 防护（三层）
1. `RGFeatureBuilder.sanitize()` — 特征构建层清零
2. `RGNormStats.parseSafeFloat()` — JSON 中 "NaN" 字符串安全解析
3. `RGPhaseAEngine.runPipeline()` — try/catch 返回最后有效结果

---

## 橘猫 UI 设计规范

| 元素 | 值 |
|------|-----|
| 主色调 | `#F97316` (阳光橘) |
| 背景 | `#FFF7ED` (奶油白) |
| 卡片 | `#FFF3E0` (暖奶油) |
| 猫毛色 | `#FB923C` (橘毛) |
| 爪/嘴周 | `#FFF3E0` (奶油白) |
| 眼睛 | `#0C0A09` (纯黑，无眼白) |
| 尾巴 | 蓬松多层，参考 ic_cat_mascot.xml |
| 字体 | Nunito (圆润Q弹) |
| 风险低 | 🐱 绿色 |
| 风险中 | 😿 橙黄色 |
| 风险高 | 🙀 红色 |

---

## 已知假设（Assumptions）

1. **MNN API**: 使用 `MNNNetInstance` + `getSessionInput/Output().setInputFloatData()` / `floatData`。
   若实际 MNN AAR 的 API 名称略有不同（如 `MNNNet` vs `MNNNetInstance`），
   需对照 MNN Android SDK javadoc 做 1:1 映射调整。

2. **MediaPipe Tasks Vision**: 使用 `tasks-vision:0.10.10`，`PoseLandmarker` 异步模式。
   回调 landmark 顺序与 MediaPipe 33 点方案一致。

3. **Nunito 字体**: 未内置于 AOSP，需开发者手动下载放入 `res/font/`，
   或改用 `downloadable fonts` XML（需 Google Services）。

4. **STGCN 关节角度索引**: JA[6]=左膝, JA[7]=右膝，遵循训练时的 23 维约定。
   若实际训练时索引不同，修改 `RGMarkerMapper.JA_L_KNEE / JA_R_KNEE`。

5. **GRF 分割**: `grf0[0:6]` = 左脚，`grf0[6:12]` = 右脚。
   若模型输出顺序为右左，交换 `grfLeft`/`grfRight` 的 copyOfRange 范围。

---

*交接日期: 2026-03-28  ·  版本: 1.0  ·  状态: 可构建 ✅*
