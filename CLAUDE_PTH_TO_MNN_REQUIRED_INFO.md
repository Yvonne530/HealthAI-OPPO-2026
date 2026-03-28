# Claude 转换与接入所需信息汇总（pth -> MNN）

本文档汇总了本次沟通中，Claude 要把 3 个 `.pth` 模型转换为 `.mnn` 并在手机端接入时必须知道的信息。

## 1. 当前要转换的模型文件

- `checkpoints/stgcn_bestgrf_PhaseA.pth`
- `checkpoints/fno_bestgrf_PhaseA.pth`
- `checkpoints/risk_bestgrf_PhaseA.pth`

## 2. 三模型执行顺序（端侧推理链路）

1. ST-GCN：输入骨架序列，输出 `joint_angles`
2. 特征构造：由 `joint_angles` 计算 `vel/acc/com` 并拼接
3. FNO：输入 72 维生物力学时序，输出未来 `K=10` 帧 GRF
4. 取 FNO 第一帧 GRF 并反归一化（如果有 norm_stats）
5. RiskMLP：输入 `[joint_angles(23) + grf(12)]` 的 35 维序列（内部可拼接心率/睡眠为 37 维）
6. 后处理：Softmax 高风险概率 + SafetyHeartbeat 规则校验 + 风险状态机平滑

## 3. 各模型输入输出契约（必须严格一致）

### 3.1 ST-GCN

- 输入：`visual_seq`，shape `(B, T, N, 3)`
  - 推理默认：`T=5`, `N=33`
- 输出主头：`joint_angles`，shape `(B, 23)`
- 另有辅助输出：`markers`，shape `(B, 84)`（转换最小闭环可不接）

### 3.2 FNO

- 输入：`bio_seq`，shape `(B, 20, 72)`
- 输出：`grf_seq`，shape `(B, K, 12)`，默认 `K=10`
- 推理时通常取第一帧 `grf_seq[:, 0, :]` 作为当前 GRF
- 说明：原始 FNO 含 FFT 分支，移动端/ONNX 兼容性通常建议导出 LSTM 兼容版本

### 3.3 RiskMLP

- 输入主特征：shape `(B, 20, 35)`，其中 35 = 23(joint_angles) + 12(grf)
- 可选生理输入：心率 + 睡眠（内部扩展为 37 维）
- 输出：
  - `logits`：`(B, 3)`
  - `confidence`：`(B, 1)`
- 风险分数：`risk_score = softmax(logits)[2]`（高风险类别概率）

## 4. 关键配置值（来自当前工程）

- `stgcn.num_nodes = 33`
- `stgcn.num_frames = 5`
- `stgcn.output_dim = 23`
- `stgcn.output_dim_aux = 84`
- `fno.input_dim = 72`
- `fno.output_dim = 12`
- `fno.seq_len = 20`
- `fno.future_k = 10`
- `risk.input_dim = 35`
- `risk.input_dim_physio = 37`
- `risk.num_classes = 3`
- `risk.seq_len = 20`
- `risk.use_physio = true`

## 5. 归一化/反归一化参数来源（必须提供）

- `norm_stats` 路径：`/home/jianyi/data/processed/norm_stats.npz`
- 推理时：FNO 输出 GRF 若为归一化值，需要按 `grf_left/right` 的 `mean/std` 做反归一化
- 统计来源：训练集（HDF5 train split）采样最多 100000 帧计算通道级 `mean/std`
- 建议一并提供：
  - `norm_stats.npz`
  - 若启用 OOD：`ood_stats.npz`

## 6. 标签语义与阈值规则（必须提供给接入方）

### 6.1 风险标签

- `0 = 低风险`
- `1 = 中风险`
- `2 = 高风险`

### 6.2 风险状态机阈值（时间平滑）

- `thresh_warn = 0.40`
- `thresh_danger = 0.70`
- `thresh_safe = 0.25`
- `thresh_warn_dn = 0.50`
- 升级/降级连续帧：
  - `k_up_warn = 3`
  - `k_up_danger = 2`
  - `k_dn_safe = 5`
  - `k_dn_warn = 8`
- `danger_lock_frames = 10`
- `min_confidence_for_upgrade = 0.5`

### 6.3 SafetyHeartbeat 物理规则

- 膝关节阈值：
  - `KNEE_MIN_RAD = -0.0873`（-5° 过伸）
  - `KNEE_MAX_RAD = 2.094`（120°）
  - `KNEE_ASYMM_RAD = 0.2618`（15°）
- GRF 自适应阈值（按体重和生理状态调整）
- 最终标签融合：`final_label = max(model_label, rule_label)`

## 7. 端侧调度与性能目标（当前代码中的目标）

- 快路径目标：`60Hz`
- 快路径延迟目标：`< 15ms`（FNO + RiskModel）
- 动态推理间隔：
  - 低风险：每 10 帧
  - 中风险：每 3 帧
  - 高风险：每帧

## 8. 转换路径建议

推荐统一路径：

1. `.pth -> .onnx`（先完成并验证）
2. `.onnx -> .mnn`

示例转换命令（按你本机 MNNConvert 路径调整）：

```bash
MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --saveExternalData=1
```

可选量化参数（按精度要求决定是否启用）：

```bash
--weightQuantBits=8
```

## 9. 对拍验证（当前已有基础）

工程里已有 ONNX 对拍脚本思路（PyTorch vs ONNX），可继续扩展为 MNN 对拍：

- `test_onnx_inference.py`
- `test_onnx_detailed.py`

建议最终交付前，至少准备 1~3 组固定样本用于端侧回归：

- 固定输入（numpy/txt）
- Python 基准输出（float）
- 可接受误差阈值（mean/max）

## 10. 仍需你补充给 Claude 的信息（代码中无法完全自动推断）

1. 目标手机机型/芯片（例如具体 OPPO 机型）
2. 端侧后端优先级（NPU/GPU/CPU）
3. 量化策略（FP16/INT8）
4. 线上验收 SLA（P50/P95 延迟、内存上限）
5. 最终业务阈值是否按当前实现直接沿用

---

## 可直接发给 Claude 的简版指令

请基于以下约束生成 pth->onnx->mnn 的完整脚本与接入说明：

- 模型：stgcn_bestgrf_PhaseA.pth, fno_bestgrf_PhaseA.pth, risk_bestgrf_PhaseA.pth
- 执行链路：STGCN -> 特征构造(vel/acc/com) -> FNO -> RiskMLP -> SafetyHeartbeat -> RiskStateMachine
- I/O：
  - STGCN: (B,5,33,3) -> (B,23)
  - FNO: (B,20,72) -> (B,10,12)
  - Risk: (B,20,35/37) -> logits(B,3), conf(B,1)
- 标签：0低/1中/2高
- 归一化：使用 norm_stats.npz，GRF 需反归一化
- 目标：60Hz，关键路径 <15ms
- 输出内容：
  1) 三个模型导出 ONNX 脚本
  2) ONNX 转 MNN 脚本
  3) Android 侧输入输出打包示例
  4) MNN 对拍验证脚本（与 Python 基准对齐）
