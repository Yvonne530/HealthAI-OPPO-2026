# MNN Android 快速参考清单

> 供 Android 开发团队使用，集成 STGCN/FNO/Risk 疾病预警模型

---

## 📦 核心文件交接清单

```
export/mnn_phasea_android_opt/
├── stgcn_phaseA.mnn (56 KB)        ✅ 必需
├── stgcn_phaseA.mnn.weight (936 KB)  ✅ 必需
├── fno_lstm_phaseA.mnn (1.66 MB)     ✅ 必需
├── fno_lstm_phaseA.mnn.weight (97 KB) ✅ 必需
├── risk_phaseA.mnn (15 KB)           ✅ 必需
└── risk_phaseA.mnn.weight (85 KB)    ✅ 必需

总大小: 2.80 MB
```

**部署到 Android**:
```bash
# 方案 1: 打入 APK 的 assets/
mkdir -p src/main/assets/models/
cp export/mnn_phasea_android_opt/*.mnn* src/main/assets/models/

# 方案 2: 应用运行时从服务器下载到 app-private storage
# 使用 MD5/SHA256 校验，见 MNN_CHECKSUMS.json
```

---

## 🔌 模型接口 Quick Reference

| Model | Input Name | Input Shape | Input Type | Output Name | Output Shape | Latency |
|-------|-----------|-------------|-----------|------------|-------------|---------|
| **STGCN** | `visual_seq` | `[1,5,33,3]` | FP32 | `joint_angles` | `[1,23]` | 0.77 ms |
| **FNO** | `bio_seq` | `[1,20,72]` | FP32 | `grf_seq` | `[1,10,12]` | 0.48 ms |
| **Risk** | `risk_seq` | `[1,20,35]` | FP32 | `risk_logits` | `[1,3]` | 0.09 ms |
| | | | | `risk_confidence` | `[1,1]` | |

**全流程延迟**: 1.34 ms (average), 2.26 ms (P95), 8.23 ms (max) → **< 15ms ✅**

---

## 📋 特征工程 (Feature Processing)

### STGCN 输入构造
```
visual_seq [1, 5, 33, 3]:
  - 5 frames: MediaPipe Pose 最近 5 帧（60 FPS → ~83ms history）
  - 33 nodes: MediaPipe 33 个关键点
  - 3 channels: (x, y, z) 坐标 / 归一化坐标 (取决于你的预处理)
  
建议预处理:
  1. MediaPipe 直接输出坐标 (pixel domain)
  2. 无需额外归一化 (模型自带)
  3. 缺失值用 0 填充
```

### FNO 输入构造（最复杂）
```
bio_seq [1, 20, 72]:
  [0:23]    Joint Angles         (来自 STGCN 输出)
  [23:46]   Joint Velocities    (numerical diff: angles[t] - angles[t-1])
  [46:69]   Joint Accelerations (numerical diff: vel[t] - vel[t-1])
  [69:72]   CoM Velocity in 3D   (center of mass displacement per frame)
  
需要维护的滚动缓冲区:
  - joint_angles_history: deque(maxlen=20)
  - velocities_history: deque(maxlen=20)
  - accelerations_history: deque(maxlen=20)
  
每帧更新:
  1. 从 STGCN 得到新的 joint_angles
  2. 计算 velocity = joint_angles[t] - joint_angles[t-1]
  3. 计算 acceleration = velocity[t] - velocity[t-1]
  4. 计算 com_vel 来自尸体重心跟踪（或用二阶中心差分）
  5. 拼接为 [ja, vel, acc, com_vel] → 72 dims
```

### Risk 输入构造
```
risk_seq [1, 20, 35]:
  主要是 STGCN 输出的子集 + 生理信号
  
  [0:35]    特征子集: 
            - 关键关节角度 (knee, hip, ankle angles) × 2 左右
            - 速度/加速度
            
  + 心率 (Heart Rate):
    获取方式: Android WearOS / 手机传感器 API
    范围: 60-180 bpm
    
  + 睡眠质量 (Sleep Quality):
    获取方式: 设备传感器或用户输入
    范围: 0-1 (0=poor, 1=excellent)
    
示例代码（见 MNNModelInference.kt）:
  val heartRate = getHeartRateFromWearable()  // float
  val sleepQuality = getSleepQuality()        // float [0,1]
  prediction = mnnInference.inferRiskLevel(
    mediapipeLandmarks, heartRate, sleepQuality
  )
```

---

## 🎯 输出解释

### STGCN Output
```
joint_angles [1, 23]:
  [0-2]:   右肩 (3 angles)
  [3-5]:   左肩
  ...
  [20-22]: 脚踝旋转等
  
范围: [-π, π] radians (无需后处理)
用途: 输入 FNO
```

### FNO Output
```
grf_seq [1, 10, 12]:
  [0:6]:   左脚 GRF (Fz, Mx, My) × 每帧
  [6:12]:  右脚 GRF (相同)
  
  Fz: 垂直力 (normalized by body weight)
  Mx, My: 力矩 (normalized)
  
10 frames: 未来 10 帧预测 (~167ms @ 60FPS)

范围: [-3.0, 3.0] (一般来说，超过是异常)
用途: 用于步态分析、跌倒预警
```

### Risk Output
```
risk_logits [1, 3]:
  [0]: 低风险的原始分数
  [1]: 中风险的原始分数
  [2]: 高风险的原始分数
  
处理步骤:
  1. softmax(logits) → probabilities [0-1]
  2. argmax(probabilities) → class_idx (0/1/2)
  3. risk_label = ["LOW", "MEDIUM", "HIGH"][class_idx]
  
示例:
  logits = [2.5, -0.3, -2.2]
  probs = softmax(logits) = [0.92, 0.07, 0.01]
  class = 0, label = "LOW"
  
阈值建议（可调整）:
  - 高风险: prob[2] > 0.70
  - 中风险: prob[1] > 0.33
  - 低风险: 其他

risk_confidence [1, 1]:
  模型对预测的置信度 (0-1)
  建议: 仅在 confidence > 0.6 时触发alert
```

---

## ⚡ 性能指标

### 延迟分布 (1000 推理)
```
STGCN:
  平均: 0.77 ms
  P95:  1.38 ms
  最大: 3.64 ms
  
FNO:
  平均: 0.48 ms
  P95:  0.73 ms
  最大: 2.03 ms
  
Risk:
  平均: 0.09 ms
  P95:  0.15 ms
  最大: 2.56 ms
  
全流程: 1.34 ms avg
```

### 内存占用
```
模型权重: 2.80 MB (全部 3 个模型)
运行时中间激活: ~50 MB (peak, 后续释放)
输入缓冲区: ~100 KB

总 APK 增量: ~3 MB (4%，可接受)
```

### 关键性能参数 (Target)
```
✅ 目标 1: 单次推理 < 15 ms
   实际: 1.34 ms avg → 还有 13.66 ms 余量
   
✅ 目标 2: 1000 次连续推理稳定
   实际: 全部通过稳定性测试
   
✅ 目标 3: 模型文件紧凑
   实际: 2.80 MB，可打入 APK
```

---

## 🛠️ 集成步骤

### Step 1: 环境准备
```gradle
// build.gradle
dependencies {
    implementation 'com.alibaba.android:mnn:1.2.0'  // 或更新版本
}
```

### Step 2: 模型加载
```kotlin
// 在 Application 或主 Activity 中
val mnnInference = MNNModelInference(context)

// 模型会自动从 assets 复制到 app-private storage
// 首次加载: ~500 ms (I/O)
// 后续加载: < 50 ms (缓存)
```

### Step 3: 集成 MediaPipe Pose
```kotlin
// MediaPipe Solutions 库
implementation 'com.google.mediapipe:solution-pose'

// 每帧回调
poseListener = object : Pose.PoseListener {
    override fun onResult(results: Pose.PoseResult) {
        val landmarks = results.poseLandmarks.map { 
            floatArrayOf(it.x, it.y, it.z)
        }.toTypedArray()
        
        // 输入到 MNN
        onPoseDetected(landmarks, heartRate)
    }
}
```

### Step 4: 启动推理循环
```kotlin
fun onPoseDetected(
    landmarks: Array<FloatArray>,
    heartRate: Float
) {
    // 维护 5-frame 滑窗
    poseBuffer.add(landmarks)
    if (poseBuffer.size > 5) poseBuffer.removeAt(0)
    
    // 就绪后运行推理
    if (poseBuffer.size == 5) {
        val prediction = mnnInference.inferRiskLevel(
            poseBuffer.toTypedArray(),
            heartRate,
            sleepQuality = 0.8f
        )
        
        // 处理输出
        handleRiskPrediction(prediction)
    }
}
```

### Step 4.5: FNO .MNN 接入补充（当前导出版）
```kotlin
// FNO files
private const val FNO_GRAPH = "models/fno_lstm_phaseA.mnn"
private const val FNO_WEIGHT = "models/fno_lstm_phaseA.mnn.weight"

// Input / Output contract
// input : bio_seq [1,20,72]
// output: grf_seq [1,10,12]

fun buildBioFeature(
  jointAngles: FloatArray,   // 23
  prevJointAngles: FloatArray,
  prevVel: FloatArray,
  comVel3d: FloatArray       // 3
): FloatArray {
  val vel = FloatArray(23) { i -> jointAngles[i] - prevJointAngles[i] }
  val acc = FloatArray(23) { i -> vel[i] - prevVel[i] }
  return FloatArray(72).also { out ->
    System.arraycopy(jointAngles, 0, out, 0, 23)
    System.arraycopy(vel, 0, out, 23, 23)
    System.arraycopy(acc, 0, out, 46, 23)
    System.arraycopy(comVel3d, 0, out, 69, 3)
  }
}

// Ensure finite values before FNO inference
fun ensureFinite(x: FloatArray) {
  for (v in x) require(v.isFinite()) { "bio_seq contains NaN/Inf" }
}
```

关键点：
- FNO 需要完整 20 帧窗口；不足时用最近有效帧复制补齐。
- 72 维顺序必须固定，不要交换维度。
- 输出 `grf_seq` 可以直接用于下游风险特征或步态展示。

### Step 5: 输出处理及UI
```kotlin
fun handleRiskPrediction(pred: RiskPrediction) {
    // 分类 + Alert
    when (pred.riskClass) {
        0 -> {
            statusView.text = "✅ 低风险 (${(pred.confidence*100).toInt()}%)"
            statusView.setTextColor(Color.GREEN)
        }
        1 -> {
            statusView.text = "⚠️ 中风险 (${(pred.confidence*100).toInt()}%)"
            statusView.setTextColor(Color.YELLOW)
            // 可选: 弱震动提醒
            vibrator.vibrate(100)
        }
        2 -> {
            statusView.text = "🚨 高风险 (${(pred.confidence*100).toInt()}%)"
            statusView.setTextColor(Color.RED)
            // 检查是否需要强制告知用户
            if (pred.confidence > 0.7f) {
                showCriticalAlert()
            }
        }
    }
    
    // 日志
    Log.d("Health", "Risk: ${pred.riskLabel}, Conf: ${pred.confidence}, " +
                   "Latency: ${pred.latencyMs}ms")
}
```

---

## 🚀 部署优化建议

| 问题 | 方案 | 优先级 |
|------|------|--------|
| APK 体积 | 模型打入 AAB 的 on-demand modules | 中 |
| 冷启动慢 | 后台预加载模型 | 低 |
| 功耗 | 限制推理频率成 10 Hz（每 100ms）| 中 |
| 内存峰值 | 顺序推理（不并行 3 个模型） | 高 |
| 隐私 | 所有推理在设备端进行，无云上传 | 高 |

---

## 📊 验证 Checklist

部署前请确认:

- [ ] 所有 6 个 `.mnn` 文件已复制到 assets
- [ ] 文件大小 & MD5 校验一致（见 MNN_CHECKSUMS.json）
- [ ] MNN runtime library 已正确链接 (libmnn.so)
- [ ] MediaPipe Pose 初始化成功 (logcat 无错误)
- [ ] 首次、second、第 100 次推理结果一致（无内存泄漏）
- [ ] 心率数据获取正常（非 NaN 或负值）
- [ ] Risk 预测输出在 [0, 3] 范围内
- [ ] 端到端延迟 < 15 ms

---

## 📞 故障排除

### 问题: 模型加载失败
```
错误: "Failed to load MNN models: ..."
排查:
  1. 文件是否在 assets/models/ 下?
  2. 文件权限正确吗?
  3. MNN 版本兼容吗? (应该 >= 1.2.0)
```

### 问题: NaN 输出
```
症状: risk_logits 包含 NaN
原因: 
  1. 输入特征范围错误（缺失值未处理）
  2. 历史缓冲区未正确维护
排查:
  1. 检查 bio_seq 所有值是否 finite: Math.isFinite(x) for all x
  2. 确认速度/加速度差分计算正确
  3. 验证 CoM 跟踪不为 0
```

### 问题: 延迟超过 15 ms
```
原因多数是:
  1. UI 线程阻塞 → 用后台线程推理
  2. 其他应用抢 CPU → 限制后台进程
  3. MediaPipe 帧率不稳定 → 增加缓冲
解决:
  new Thread {
      val pred = mnnInference.inferRiskLevel(...)
      runOnUiThread { updateUI(pred) }
  }.start()
```

---

## 📚 完整源代码参考

- **集成范例**: `MNNModelInference.kt` (此项目)
- **特征工程**: `data/hdf5_dataset.py` (PyTorch 版，逻辑通用)
- **风险规则**: `inference/risk_state_machine.py`
- **模型参数**: `configs/config.yaml`

---

## 📞 联系方式

**技术问题反馈**:
- MNN 文件完整性问题: 检查 MD5 (MNN_CHECKSUMS.json)
- 模型推理精度问题: 对比 logs/mnn_quality_report.json
- 集成困难: 参考 MNNModelInference.kt 代码示例

---

**文档版本**: 1.0  
**生成日期**: 2026-03-28  
**状态**: Ready for Android Development ✅
