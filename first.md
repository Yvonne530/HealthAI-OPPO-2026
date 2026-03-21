你是一名顶级AI系统架构师 + 机器学习工程专家 + 端侧部署专家。

你的任务不是解释，而是**为一个真实参赛项目生成“完整可运行代码体系”**。

---

# 🧠 项目背景（必须严格理解）

项目名称：RehabGuardian 10.0
目标：构建一个**纯端侧运行的ACL损伤风险实时监测系统**

核心特点：

* 完全离线（无云API）
* 基于真实生物力学数据（.b3d）
* 视觉 → 生物力学 → 物理 → 语义 全链路
* 最终部署在手机（MNN推理）

---

# 📦 数据集（必须严格按此结构实现）

数据来源：AddBiomechanics - Camargo2021

数据规模：

* 总帧数：200,838
* 试验数：105
* 标记点：28个（Plug-in-Gait）
* 关节角度：23个

关键字段（来自 nimblephysics）：

```python
data = {
    "joint_angles": (N, 23),        # processingPass[2].pos
    "joint_vel": (N, 23),           # vel
    "joint_acc": (N, 23),           # acc
    "markers": (N, 28, 3),          # markerObservations
    "grf": (N, 6),                  # groundContactWrenches[:,:6]
    "grf_moments": (N, 6),
    "com": (N, 3),
    "contact": (N, 2),
    "t": (N,)
}
```

⚠️ 强制要求：

* 不允许假设字段
* 不允许虚构数据结构
* 必须严格兼容 `.b3d`

---

# 🧩 系统架构（必须完全按此实现）

你需要生成完整代码，包含以下模块：

---

## ① 数据解析模块（b3d_loader.py）

要求：

* 使用 nimblephysics
* 支持批量加载整个数据集
* 输出 numpy / npz
* 自动选择 processingPass 2
* 支持 mmap（避免爆内存）

---

## ② 数据协议模块（rg_sample.py）

实现统一数据结构：

```python
class RGSample:
    visual_seq: (5,33,3)
    joint_angles: (23,)
    joint_vel: (23,)
    joint_acc: (23,)
    markers: (28,3)
    grf: (6,)
    com: (3,)
    risk_score: float
```

要求：

* 支持序列化
* 支持转tensor
* 所有模块必须用它

---

## ③ ST-GCN模块（stgcn_model.py）

任务：
👉 视觉骨骼 → 生物力学坐标

输入：
(5,33,3)

输出：
(107,) = 23关节角 + 84 marker

要求：

* 明确 adjacency matrix
* 使用 PyTorch
* 参数量 < 10M
* 提供 forward()

---

## ④ FNO模块（fno_model.py）

任务：
👉 joint → GRF预测

输入：
(batch, 20, 72)

输出：
(batch, 20, 6)

要求：

* 使用 Fourier Neural Operator
* 必须有 FFT层
* 如果复杂，提供简化版实现（但必须真实）

---

## ⑤ 教师打标模块（teacher_labeler.py）

任务：
👉 自动生成风险标签

输入：
joint_angles + grf

输出：
risk_score + risk_level

要求：

* 不依赖真实LLM（用规则模拟）
* 必须可运行
* 后续可替换为Qwen

---

## ⑥ 蒸馏训练模块（train_pipeline.py）

要求：

* 同时训练：

  * ST-GCN
  * FNO
* 支持：

  * batch训练
  * loss打印
  * checkpoint保存
* loss函数：

```python
loss = coord_loss + physics_loss
```

---

## ⑦ 推理模块（inference.py）

实现完整流程：

```text
visual → ST-GCN → joint → FNO → risk
```

要求：

* 支持单序列输入
* 输出风险等级

---

## ⑧ 导出模块（export_mnn.py）

要求：

* PyTorch → ONNX → MNN
* 提供示例代码
* 可直接运行

---

# ⚙️ 工程要求（非常重要）

必须满足：

1. 每个模块是独立.py文件
2. 所有代码可运行（不能伪代码）
3. 提供 main.py 入口
4. 使用 PyTorch
5. 不允许省略关键实现
6. 所有shape必须严格标注
7. 所有模型必须能 forward

---

# 🚫 禁止行为

* ❌ 不要解释原理
* ❌ 不要写论文式回答
* ❌ 不要只写框架
* ❌ 不要伪代码
* ❌ 不要用“略”

---

# ✅ 输出要求（必须遵守）

按以下顺序输出完整代码：

1. 项目结构树
2. 每个.py文件完整代码
3. main.py
4. 运行说明

---

# 🎯 你的目标

生成一个：

👉 可以直接运行训练
👉 可以真实forward
👉 可以导出模型
👉 可以作为比赛demo

的完整工程代码

---

现在开始输出，不要解释。
