# RehabGuardian · 端侧 ACL 损伤风险实时监测系统

> Real-time On-Device ACL Injury Risk Monitoring System
> 第十九届全国大学生软件创新大赛（SWC2026）参赛项目 · OPPO 手机端侧 AI

单目 RGB 摄像头实时提取人体姿态，在手机端离线完成 **关节角度估计 → 地面反作用力（GRF）预测 → 三级 ACL 风险分类** 的完整推理链。零云端、零穿戴设备。

## 系统架构

```
Camera2 (60fps) ──► MediaPipe Pose (33 keypoints)
                          │
                          ▼
              ┌───────────────────────┐
              │ ST-GCN                │  (B,5,33,3) → joint_angles(23) + markers(28×3)
              │ 关节分组编码+对称约束   │
              └──────────┬────────────┘
                         ▼
              ┌───────────────────────┐
              │ FNO (+Learnable Lag)  │  (B,20,72)  → GRF 未来10帧 (~167ms) ×12通道
              │ FFT主路/LSTM端侧降级    │
              └──────────┬────────────┘
                         ▼
              ┌───────────────────────┐
              │ RiskMLP + FSM         │  (B,20,35)  → 3级风险 + 置信度 + 可解释原因
              │ 迟滞/EWMA/心率动态阈值  │
              └───────────────────────┘
```

**核心创新**
- **可学习时延对齐（Learnable Temporal Lag）**：GRF 是力响应信号，天然滞后于关节运动 20–100ms。用可微 sigmoid 参数化 lag 让模型自动学习视觉输入与 GRF 响应间的物理延迟，从"同步拟合"升级为"因果建模"
- **FNO 端侧降级路径**：FFT 算子在移动端 NPU 兼容性差，导出时切换 LSTM-compatible 分支（`--useOriginRNNImpl`），保证全机型可用
- **ST-GCN 结构化先验**：按 OpenSim gait2392 关节分组编码（左腿链/右腿链/骨盆/脊柱/颈），并施加左右镜像对称约束损失 L_sym

## 数据集

| 项目 | 规格 |
|---|---|
| 来源 | Camargo2021（AddBiomechanics 平台开放获取） |
| 规模 | 20 名受试者（AB06–AB30）、120 个 .b3d trial |
| 特征 | 23 维关节角度/角速度/角加速度、双脚 GRF 各 6 维（Fx,Fy,Fz,Mx,My,Mz）、质心 3 维 |
| 标签 | 物理规则 TeacherLabeler 打标的三级风险标签（低/中/高） |

## 端侧部署性能（实测）

**OPPO Reno15 Pro（Snapdragon 8 Gen 3，Android 16）· MNN 2.9.0 · 1000 次连续推理**

| 模型 | 平均延迟 | P95 | 最大 | 精度 | 大小 |
|---|---|---|---|---|---|
| STGCN | 0.77 ms | 1.38 ms | 3.64 ms | FP32（vs PyTorch 误差 <1e-5） | 0.95 MB |
| FNO (LSTM 路径) | 0.48 ms | 0.73 ms | 2.03 ms | FP16 | 1.75 MB |
| Risk | 0.09 ms | 0.15 ms | 2.56 ms | FP16（误差 ~3e-5） | 0.10 MB |
| **三模型串行** | **1.34 ms** | **2.26 ms** | 8.23 ms | 1000 次无 NaN/Inf | **2.80 MB** |

> 口径：三模型 forward 推理延迟（不含相机采集与 MediaPipe 前处理）。转换链路 `PyTorch (.pth) → ONNX (opset 11) → MNNConvert`。
> App 层整体功能测试见 [开发.md](开发.md) / [测试.md](测试.md)（66/66 单元测试通过）。

## 仓库导航（分支地图）

| 分支 | 内容 |
|---|---|
| [`feat/st-gcn`](../../tree/feat/st-gcn) | **训练管线（PyTorch）**：`models/stgcn.py` / `fno.py` / `risk_model.py`、`train_v2.py`、HDF5 数据集加载、预处理、ONNX 导出脚本、训练权重 checkpoints、[ONNX 导出一致性测试报告](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/feat/st-gcn/ONNX_TEST_REPORT.md) |
| [`ABtest`](../../tree/ABtest) | **MNN 端侧交付包**：模型延迟/精度 benchmark、Android 集成指南（`MNN_ANDROID_INTEGRATION_GUIDE.md`）、Kotlin 参考实现、文件校验和 |
| [`Yvonne530-upload-1`](../../tree/Yvonne530-upload-1) | 项目技术研究报告（`Technical documentation.md`）：问题定义、相关工作、部署细节、验证结果 |
| [`android-app`](../../tree/android-app) | Android Demo App v1：`MNNInferenceEngine.kt`、滑动窗口缓冲、风险状态机、CameraX + MediaPipe 实时渲染（assets 内置 .mnn 模型） |
| [`feat/android_app_two`](../../tree/feat/android_app_two) | Android App v2（RehabGuardian UI）：JNI 直连 MNN C++ API、特征工程管道、可解释风险输出、GRF 预测曲线、最佳帧对比、PDF 报告、Room 会话持久化 |
| `main` | 项目文档（开发文档 / 测试文档） |

## 快速开始

### 训练（PC / Kaggle）
```bash
git checkout feat/st-gcn
pip install -r requirements.txt   # torch, h5py, nimblephysics ...
python preprocess.py              # .b3d → HDF5
python train_v2.py                # 三模型联合训练
python export/export_onnx.py      # ONNX 导出 + 一致性校验
```

### 端侧集成（Android）
```bash
git checkout android-app          # 或 feat/android_app_two
# assets/ 内置 stgcn.mnn / fno_lstm.mnn / risk.mnn（合计 2.80MB）
./gradlew assembleDebug
```

## License

MIT
