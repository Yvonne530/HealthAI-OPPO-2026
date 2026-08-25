# RehabGuardian 优化前后对比报告

## 概述

本次优化针对 RehabGuardian 康复训练风险监测系统在真实测试中发现的四大核心问题进行了系统性修复。

---

## 一、优化项目汇总

### 1. 统一风险阈值（优先级：高）

**问题描述：**
- `SessionManager` 使用 0.35/0.70 作为低/中风险分界
- `ReportGenerator` 使用 0.4/0.7 作为低/中风险分界
- `FrameData.kt` (Session.riskLevelText) 使用 0.35/0.70
- 同一风险评分在不同模块得出不同风险等级结论

**修复方案：**
- 新增 `RiskThresholds.kt` 作为全局唯一阈值定义
- DANGER > 0.7, WARNING > 0.4, else SAFE
- 所有模块引用 `RiskThresholds.riskLevel()`, `.advice()`, `.DANGER`, `.WARNING`

**影响文件：**
- `RiskStateMachine.kt` — 使用 `RiskThresholds.DANGER/WARNING`
- `ReportGenerator.kt` — 使用 `RiskThresholds.riskLevel()`
- `SessionManager.kt` — 使用 `RiskThresholds.riskLevel()`, `.advice()`
- `FrameData.kt` — 使用 `RiskThresholds.riskLevel()`

---

### 2. 过滤低可见性关键点（优先级：高）

**问题描述：**
- `PoseProcessor` 只检查整体检测置信度（0.5），不检查每个关键点（landmark）的可见性
- 低可见性关键点被零填充后仍送入推理管道
- 对于屏幕录制画面+模拟器摄像头拍摄的场景，MediaPipe 对远处关键点的可见性很低（< 0.3）
- 零填充的关键点导致关节角度计算严重失真

**修复方案：**
- 对 33 个关键点逐一检查 `visibility()`，低于 0.3 的零填充
- 关键下肢关键点（髋/膝/踝，索引 23-28）全部可见性 >= 0.3 才输出结果
- 任一关键关键点不达标，整帧被丢弃，不进入推理管道

**影响文件：**
- `PoseProcessor.kt` — 新增 per-landmark 可见性过滤 + 关键关键点校验

---

### 3. 修复 GRF 计算（优先级：中）

**问题描述：**
- `BiomechanicsAnalyzer.analyze()` 使用 `(1 - (hipY - ankleY)).coerceIn(0f, 1f)` 作为 GRF 代理
- 这是纯几何近似，与真实地面反作用力毫无物理关系
- `MNNInferenceEngine` 已有 FNO 模型输出真实 `grfLeft[2]` 和 `grfRight[2]`
- 报告中提到的落地冲击力（GRF）基于伪数据

**修复方案：**
- 移除 BiomechanicsAnalyzer 中的简化 GRF 代理计算，设为 0f
- 在 `Session.topAnomalies()` 中，GRF 判断改用 `it.grfLeft[2] + it.grfRight[2]`（来自 FNO 模型真实输出）
- `MNNInferenceEngine.infer()` 已经通过 FNO 模型输出 `grfLeft` 和 `grfRight`，并在 `safetyCheck()` 中使用
- 在 `FrameData` 中已存储真实的 `grfLeft` 和 `grfRight` 数组

**影响文件：**
- `BiomechanicsAnalyzer.kt` — 移除简化 GRF，`grfMagnitude = 0f`
- `FrameData.kt` — topAnomalies() 已使用真实 FNO GRF 输出

---

### 4. 加入解剖学合理性校验（优先级：中）

**问题描述：**
- 无肢体长度校验：髋-膝距离和膝-踝距离极端不对称时不告警
- 无关节角度钳位：膝过伸角度（如 > 190° 或 < 0°）被视为真实值
- MediaPipe Lite 模型在低质量输入下可能输出违反解剖学常识的关键点

**修复方案：**
- 计算左右下肢肢体长度（髋到膝距离），检查是否为零（< 0.05，归一化坐标）
- 检查双侧肢体长度比：超出 0.5x-2.0x 范围视为检测异常
- 膝关节角度钳位到 [10°, 190°] 的解剖学合理范围
- 返回 `BiomechanicsMetrics.valid` 和 `invalidReason` 用于下游诊断

**影响文件：**
- `BiomechanicsAnalyzer.kt` — 新增肢体长度校验 + 角度钳位

---

## 二、代码变更统计

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `RiskThresholds.kt` | 新增 | 全局风险阈值定义 |
| `PoseProcessor.kt` | 修改 | 添加 per-landmark 可见性过滤 |
| `BiomechanicsAnalyzer.kt` | 修改 | 移除伪 GRF，添加解剖学校验 |
| `RiskStateMachine.kt` | 修改 | 引用 RiskThresholds |
| `ReportGenerator.kt` | 修改 | 引用 RiskThresholds |
| `SessionManager.kt` | 修改 | 引用 RiskThresholds |
| `FrameData.kt` | 修改 | 引用 RiskThresholds |

---

## 三、预期效果

1. **识别精度提升**：低可见性关键点过滤 + 解剖学校验减少无效帧进入推理管道
2. **关节角度准确性提升**：角度钳位 + 肢体长度校验防止极端值
3. **风险等级一致性**：所有模块使用统一阈值，消除决策冲突
4. **GRF 数据可靠性**：报告中的冲击力数据来自 FNO 模型而非几何近似
5. **APK 编译成功**：所有 Kotlin 代码通过编译

---

## 四、构建结果

```
BUILD SUCCESSFUL in 6s
40 actionable tasks: 5 executed, 35 up-to-date
```

APK 输出路径：`app/build/outputs/apk/debug/app-debug.apk`

---

*报告生成时间：2026-08-25*
*由 RehabGuardian AI 工程团队自动生成*
