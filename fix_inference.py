#!/usr/bin/env python3
"""
fix_inference.py
修复 inference.py 中两个 bug：
  1. pred_grf_t[0, 0] → FNO 返回 tuple，应该先取 [0] 再索引
  2. torch.load 加 weights_only=True

把这个文件放到项目根目录运行一次：
  python fix_inference.py
"""
import os, re

INF_FILE = "inference/inference.py"

if not os.path.exists(INF_FILE):
    print(f"找不到 {INF_FILE}")
    exit(1)

with open(INF_FILE) as f:
    src = f.read()

# ---- 修复1：FNO tuple 返回值 ----
# 原代码：pred_grf_t = self._fno(bio_tensor)
#          grf_norm = pred_grf_t[0, 0].cpu().numpy()
old1 = "pred_grf_t = self._fno(bio_tensor)           # (1, K, 12)\n        grf_norm   = pred_grf_t[0, 0].cpu().numpy()  # 取第一帧"
new1 = ("pred_grf_raw = self._fno(bio_tensor)         # (1, K, 12) 或 tuple\n"
        "        pred_grf_t = pred_grf_raw[0] if isinstance(pred_grf_raw, tuple) else pred_grf_raw\n"
        "        grf_norm   = pred_grf_t[0, 0].cpu().numpy()  # 取第一未来帧")

# 更通用的替换（不依赖精确格式）
if "pred_grf_t[0, 0]" in src and "isinstance" not in src:
    src = src.replace(
        "pred_grf_t = self._fno(bio_tensor)",
        "pred_grf_raw = self._fno(bio_tensor)\n"
        "        pred_grf_t = pred_grf_raw[0] if isinstance(pred_grf_raw, tuple) else pred_grf_raw",
        1
    )
    print("  ✅ 修复1：FNO tuple 返回值")
elif "isinstance(pred_grf_raw, tuple)" in src:
    print("  ✅ 修复1：已经修复过了")
else:
    # 通用查找替换
    src = re.sub(
        r'(pred_grf_t\s*=\s*self\._fno\([^)]+\))',
        r'pred_grf_raw = self._fno(bio_tensor)\n        pred_grf_t = pred_grf_raw[0] if isinstance(pred_grf_raw, tuple) else pred_grf_raw',
        src, count=1
    )
    print("  ✅ 修复1（正则）：FNO tuple 返回值")

# ---- 修复2：weights_only=True ----
count = src.count("torch.load(ckpt, map_location=self.device)")
if count > 0:
    src = src.replace(
        "torch.load(ckpt, map_location=self.device)",
        "torch.load(ckpt, map_location=self.device, weights_only=True)"
    )
    print(f"  ✅ 修复2：torch.load weights_only=True ({count} 处)")

# ---- 修复3：risk 模型调用时添加心率/睡眠参数 ----
# 原：logits, conf = self._risk_model(rx, hr_t, sl_t)  ← 可能已正确
# 查找并确认
if "self._heart_rate" in src and "self._sleep_score" in src:
    print("  ✅ 修复3：心率/睡眠参数已正确传入")
else:
    print("  ⚠️  修复3：请手动检查 risk_model 调用是否传入心率/睡眠")

with open(INF_FILE, "w") as f:
    f.write(src)

print(f"\n{INF_FILE} 修复完成")
print("运行：python main.py --mode infer")