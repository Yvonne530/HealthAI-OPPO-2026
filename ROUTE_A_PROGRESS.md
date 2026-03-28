# 路线 A 执行进度跟踪

## 当前状态
- **开始时间**：2026-03-28 16:51:58 UTC
- **进程 PID**：见下方
- **日志路径**：`/tmp/fno_lstm_train.log`

## 预计时间表

| 阶段 | 预计耗时 | 开始时间 | 完成时间 | 状态 |
|------|---------|---------|---------|------|
| **Phase 1: FNO-LSTM 训练** | 7 小时 | 16:51 | ~23:51 | ⏳ 运行中 |
| Phase 1.1: 初始化+加载数据 | 3-5 分钟 | 16:51 | 16:55 | ✅ 完成 |
| Phase 1.2: 25 epoch 训练 | ~6 小时 50 分钟 | 16:55 | ~23:45 | ⏳ 运行中 |
| **Phase 2: ONNX 导出+简化** | 5 分钟 | ~23:45 | ~23:50 | ⏹️ 待启动 |
| **Phase 3: MNN 转换** | 3-5 分钟 | ~23:50 | ~23:55 | ⏹️ 待启动 |
| **Phase 4: 严格评测（1000 runs）** | 15-20 分钟 | ~23:55 | ~00:15 | ⏹️ 待启动 |
| **总耗时** | **~7.5 小时** | 16:51 | ~00:15 | ⏳ 运行中 |

## 实时监控命令

### 查看训练进度
```bash
tail -f /tmp/fno_lstm_train.log
```

### 查看当前 epoch（实时）
```bash
tail -20 /tmp/fno_lstm_train.log | grep "Epoch"
```

### 检查进程是否还在运行
```bash
ps aux | grep "train_fno_lstm_fast.py" | grep -v grep
```

### 估计剩余时间
每个 epoch 约 17 分钟，查看最新 epoch 编号，剩余时间 = (25 - 当前epoch数) × 17 分钟

## 当训练完成时（自动执行以下步骤）

### 1. 检查 checkpoint 是否已生成
```bash
ls -lh checkpoints/fno_bestgrf_PhaseA_lstm.pth
```

### 2. 一键导出 + 评测
```bash
python export_and_eval_fno_lstm.py
```
这会自动：
- ✅ 导出 ONNX
- ✅ 简化 ONNX
- ✅ 转 MNN（带 --useOriginRNNImpl）
- ✅ 跑 1000 次严格评测
- ✅ 生成"优秀"判定报告

### 3. 查看最终报告
```bash
cat logs/mnn_quality_report_fno_lstm.json | python -m json.tool
```

## 可能的情况

### 情况 1：训练提前收敛（early stopping 触发）
- **表现**：日志显示 "Early stopping (no improvement for 8 epochs)"
- **耗时**：可能 3-5 小时而非 7 小时
- **操作**：直接运行 `python export_and_eval_fno_lstm.py`

### 情况 2：训练因 OOM 或其他错误中断
- **检查**：`tail -50 /tmp/fno_lstm_train.log` 查看错误信息
- **恢复**：根据错误类型重新运行或调整参数

### 情况 3：评测后 FNO 仍未达"优秀"
- **可能原因**：FP16 精度损失、LSTM 权重初始化等
- **方案**：
  1. 尝试仅用 FP32 导出：修改 export_and_eval_fno_lstm.py，移除 `--fp16` 标记
  2. 再训延长到 50 epoch
  3. 调整 LSTM 隐层维度（current: 256）

## 关键文件位置

| 文件 | 描述 |
|------|------|
| `train_fno_lstm_fast.py` | 快速训练脚本（已运行） |
| `export_and_eval_fno_lstm.py` | 一键导出+评测脚本 |
| `checkpoints/fno_bestgrf_PhaseA_lstm.pth` | 新生成的 LSTM checkpoint |
| `export/mnn_phasea_fno_lstm/fno_lstm_phaseA.mnn` | 最终 MNN 模型 |
| `logs/mnn_quality_report_fno_lstm.json` | 最终评测报告 |
| `/tmp/fno_lstm_train.log` | 实时训练日志 |

## 进度检查清单

在下次对话时，请告诉我：
- [ ] 训练是否已完成（check log 最后一行）
- [ ] checkpoint 文件是否存在
- [ ] 最终 epoch 是多少，best val loss 是多少
- [ ] 是否需要我立即启动导出+评测

---

**下一步**：当你看到训练日志显示 "Training complete" 时，直接告诉我，我会立刻运行导出和评测。
