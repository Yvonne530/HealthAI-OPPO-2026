---

# 🚀 RehabGuardian 10.0 最终版工程提示词

你是一名顶级 AI 系统架构师 + 机器学习工程专家 + 端侧部署专家。  
你的任务不是解释，而是**为一个真实参赛项目生成“完整可运行代码体系”**。  

**项目名称**：RehabGuardian 10.0  
**目标**：构建一个纯端侧运行的 ACL 损伤风险实时监测系统，实现视觉骨骼 → 生物力学坐标 → 物理预测 → 风险预警的完整链路，最终部署在手机（MNN 推理）。

---

## 📌 核心原则（必须严格遵守）

1. **完全离线**：不依赖任何云端 API。  
2. **基于真实数据**：所有模型使用 AddBiomechanics 真实 .b3d 数据训练。  
3. **全链路可运行**：生成的代码必须包含数据加载、训练、推理、模型导出，且能通过 `python main.py` 执行。  
4. **禁止伪代码**：每个模块必须提供可执行代码，不得出现“略”、“待补充”等占位。  
5. **模块独立**：每个功能写在独立 `.py` 文件中，结构清晰。

---

## 🧠 技术栈与约束

- Python 3.8+  
- PyTorch 1.10+（训练）  
- MNN（导出与端侧推理）  
- nimblephysics（加载 .b3d）  
- mediapipe（视觉骨骼提取）  
- 其他：numpy, scipy, scikit-learn, matplotlib  

---

## 📦 数据集（必须严格使用）

**来源**：AddBiomechanics - Camargo2021（.b3d 文件）  
**规模**：105 个试验，200,838 帧，28 个 Plug-in-Gait 标记点，23 个关节角度。

### 必须从 .b3d 中提取的字段（使用 `processingPass[2]`）：

```python
data = {
    "joint_angles": (N, 23),        # processingPass[2].pos
    "joint_vel": (N, 23),           # processingPass[2].vel
    "joint_acc": (N, 23),           # processingPass[2].acc
    "markers": (N, 28, 3),          # markerObservations
    "grf": (N, 6),                  # groundContactWrenches[:,:6]
    "grf_moments": (N, 6),          # groundContactWrenches[:,6:]
    "com": (N, 3),                  # processingPass[2].comPos
    "contact": (N, 2),              # processingPass[2].contact
    "t": (N,)
}
```

**强制要求**：
- 不允许虚构字段或数据结构。
- 加载时必须支持 mmap 模式，避免内存溢出。
- 按试验（trial）划分训练/验证/测试（80/10/10），防止数据泄露。

---

## 🧩 系统架构（必须完整实现以下模块）

你必须生成以下所有模块的完整代码。每个模块需独立文件，并提供清晰的接口。

### ① 数据解析模块（`b3d_loader.py`）
- 使用 `nimblephysics` 加载 .b3d 文件。
- 批量处理整个数据集，输出 `.npz` 或内存映射文件。
- 自动选择 `processingPass 2`（精度最高）。
- 返回一个包含所有字段的字典或 Dataset 类。

### ② 统一数据协议（`rg_sample.py`）
定义 `RGSample` 类，作为全系统数据交换格式：

```python
class RGSample:
    visual_seq: torch.Tensor   # (5, 33, 3)
    joint_angles: torch.Tensor # (23,)
    joint_vel: torch.Tensor    # (23,)
    joint_acc: torch.Tensor    # (23,)
    markers: torch.Tensor      # (28, 3)
    grf: torch.Tensor          # (6,)
    com: torch.Tensor          # (3,)
    risk_score: float          # 风险分数 (0~1)
    risk_level: str            # "low" / "medium" / "high"
```

要求：支持序列化、支持转 tensor、所有模块输入/输出均基于此类。

### ③ ST-GCN 模型（`stgcn_model.py`）
**任务**：视觉骨骼序列 → 生物力学坐标  
**输入**：`(batch, 5, 33, 3)`  
**输出**：`(batch, 107)` = 23 关节角度 + 84 标记点坐标（28×3）  

**必须实现**：
- 根据给定的边连接构建邻接矩阵。
- 图卷积层 × 3，每层后接 BatchNorm + ReLU。
- 全局平均池化 + 全连接层（256 → 107）。
- 参数量 < 10M。

### ④ FNO 模型（`fno_model.py`）
**任务**：生物力学特征 → 未来 20 帧 GRF  
**输入**：`(batch, 20, 72)`  
  - 72 维构成：`[joint_angles(23), joint_vel(23), joint_acc(23), com(3)]`  
**输出**：`(batch, 20, 6)`（未来 20 帧的 Fx,Fy,Fz,Mx,My,Mz）

**必须实现**：
- 使用傅里叶神经算子（Fourier Neural Operator），包含 FFT 层和频域滤波。
- 提供简化但可运行的实现（若完整 FNO 复杂，可提供基于 `torch.fft.rfft` 的基础版本）。
- 支持 modes 参数（建议 modes=12）。

### ⑤ 教师打标模块（`teacher_labeler.py`）
**任务**：根据关节角度和 GRF 自动生成风险标签（模拟 7B 教师模型）。  
**输入**：`joint_angles` (N,23) + `grf` (N,6)  
**输出**：`risk_score` (0~1) + `risk_level` (low/medium/high)

**规则（必须实现）**：
- 使用膝关节屈曲角度（索引 6 和 13）和垂直 GRF（Fz）计算风险。
- 如果任一膝角度 < 0°（过伸）或 > 120°（过度屈曲）→ 高风险。
- 如果 GRF 峰值 > 1.5×体重（默认 70kg）→ 高风险。
- 其他情况根据阈值划分中/低风险。
- 输出需平滑（如移动平均）。

### ⑥ 蒸馏训练模块（`train_pipeline.py`）
**任务**：联合训练 ST-GCN 和 FNO。  
**要求**：
- 支持批量训练、验证。
- 使用配置管理超参数（config.yaml）。
- 损失函数：`loss = MSE(joint_angles, gt_joint_angles) + MSE(grf, gt_grf)`。
- 记录训练曲线，保存最佳模型 checkpoint。
- 支持早停（patience=10）、余弦退火学习率调度。

### ⑦ 推理模块（`inference.py`）
**任务**：接收视频/骨骼输入，输出风险等级。  
**接口**：

```python
def inference(input_data, mode="video", include_semantic=False):
    """
    mode="video": input_data 是视频帧 (H,W,3)，用 MediaPipe 提取 33 点 3D 骨骼（深度可假设 z=0）
    mode="pose": input_data 已是骨骼点 (33,3)
    include_semantic: 是否生成文本解释（可扩展）
    """
```

**流程**：
1. 提取视觉骨骼（33×3）。
2. 维护 5 帧缓存，输入 ST-GCN 得到 107 维生物力学坐标。
3. 从生物力学坐标提取关节角度、角速度、角加速度、COM，构建 72 维特征，维护 20 帧缓存。
4. 输入 FNO 预测未来 20 帧 GRF。
5. 基于预测 GRF 峰值和当前关节角度计算风险分数与等级。
6. 返回 `RGSample`。

### ⑧ 模型导出模块（`export_mnn.py`）
**任务**：将训练好的 PyTorch 模型转为 MNN 格式。  
**要求**：
- 分别导出 ST-GCN 和 FNO 为 `.mnn` 文件。
- 提供转换脚本和示例。
- 支持动态输入（batch 维度可设为 -1）。

---

## ⚙️ 工程要求（必须满足）

- **项目结构**：按以下目录组织，每个模块一个文件：

```
project/
├── data/
│   ├── raw/               # .b3d 文件
│   ├── processed/         # 预处理后 .npz
│   └── datasets.py        # PyTorch Dataset
├── models/
│   ├── stgcn.py
│   ├── fno.py
│   └── risk_assessment.py
├── trainers/
│   ├── train_stgcn.py
│   ├── train_fno.py
│   └── utils.py
├── inference/
│   ├── inference.py
│   └── heytap_health_adapter.py  # 可选 OPPO 健康接口
├── configs/
│   └── config.yaml
├── utils/
│   ├── b3d_loader.py
│   ├── rg_sample.py
│   └── metrics.py
├── notebooks/            # 可选
├── requirements.txt
├── main.py               # 训练入口
└── README.md
```

- **配置文件**：使用 YAML 管理所有超参数（包括数据路径、模型参数、训练参数）。
- **随机种子**：固定 seed=42，确保可复现。
- **代码规范**：包含类型提示、详细 docstring，遵循 PEP8。
- **异常处理**：在文件读取、模型加载时添加 try/except。

---

## 🚫 禁止行为

- ❌ 不要解释原理或写论文式回答。
- ❌ 不要只提供框架或伪代码。
- ❌ 不要省略关键实现（如邻接矩阵构建、FFT 层）。
- ❌ 不要硬编码路径（必须使用配置文件）。
- ❌ 不要忽略数据划分（按试验划分）。

---

## ✅ 输出要求（必须严格按此顺序）

你只需输出代码，不要额外文字。按以下顺序：

1. **项目结构树**（文本形式）
2. **每个 `.py` 文件的完整代码**（按文件路径列出，如 `data/datasets.py`、`models/stgcn.py` 等）
3. **`main.py` 完整代码**
4. **运行说明**（简要，放在 `README.md` 中）

确保所有代码可直接复制保存，并能在给定硬件（RTX 4070 8GB）上完成训练与推理。

---

## 🎯 最终目标

生成一个：

- ✅ 可以直接运行训练  
- ✅ 可以真实 forward  
- ✅ 可以导出 MNN 模型  
- ✅ 可以作为比赛演示的完整工程  

---

**现在开始输出，不要解释。**
