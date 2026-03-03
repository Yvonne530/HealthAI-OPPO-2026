# 以下是为你量身定制的第一步：核心引擎与数据获取方案

## 1. 数据集：去哪里抓取“人体受力”数据？

由于你目前没有私有数据，你需要能够关联“姿态”与“关节受力（剪切力/力矩）”的开源金矿。

**推荐数据集：TotalCapture (University of Surrey)**

*理由*：它是目前最全的多模态数据集之一，包含 IMU 传感器、光学动作捕捉（像 AIUnit 输出的那样）和同步的地面反作用力数据。

**获取网址**：cvssp.org

**推荐模型库：OpenSim-Models**

*理由*：如果要计算 ACL（前交叉韧带）剪切力，你需要医学级的人体模型。

**获取网址**：github.com

**魔改点**：你需要将 OpenSim 的 .osim 模型导出为 MuJoCo 可读的 .xml 格式。

## 2. 开源仓库组合：你的“武器库”

你需要将以下四个仓库组合起来，形成你的“影子世界”：

| 模块 | 推荐开源仓库 / 工具 | 具体用途 |
|------|---------------------|----------|
| 仿真引擎 | google-deepmind/mujoco | 核心物理引擎，负责实时仿真碰撞与约束 |
| Android 封装 | greydanus/mujoco-android | **关键魔改项**：这是一个早期的 Android 移植版，你需要基于它的 JNI 层来编写你文档中的 syncPose 逻辑 |
| 推理加速 | Tencent/ncnn | 针对手机端优化的推理库，用来运行你的 0.3s 预判模型 |
| AR 渲染 | google-ar/arcore-android-sdk | 与 OPPO 的 ARUnit 兼容，用于渲染 Aqua Dynamics 流体云 |

## 3. 具体魔改路径：从“通用”到“康复专精”

你需要重点攻克的三个魔改点：

### 坐标系重映射（AIUnit → MuJoCo）

*现状*：OPPO AIUnit 输出的是 33 个相对于相机的 3D 骨骼点。

*魔改*：你需要编写一个 Inverse Kinematics (IK, 逆运动学) 算法，将这 33 个点的坐标转换为 MuJoCo 模型中的 qpos（广义坐标/关节角度）。

*目标效果*：手机镜头里的人动一下，MuJoCo 影子世界的虚拟人必须 1:1 同步动作。

### 0.3s 预判模型（GRU/Transformer）

*现状*：普通模型只输出下一帧。

*魔改*：利用你的 4070 显卡，在 TotalCapture 数据集上训练一个 Sequence-to-Sequence 模型。输入过去 20 帧姿态，输出未来 15 帧（约 0.3s）的预测值。

*目标效果*：系统在用户还没做出错误动作前，就已经在“影子世界”里推演到了风险。

### Aqua Dynamics 流体交互

*现状*：ColorOS 16 的流体云是系统级的。

*魔改*：利用 GLSL (OpenGL Shading Language) 编写一个片段着色器，模拟流体从“静谧蓝”到“警戒红”的渐变。

*目标效果*：将受力大小（ACL Load）映射为流速和颜色，实现“无意识引导”。

**接下来的建议**：

你现在可以先在你的 4070 电脑上搭建 Python 版 MuJoCo 环境，尝试加载 humanoid.xml 并用代码驱动它的关节。

需要我为你提供一份针对 AIUnit 骨骼点到 MuJoCo qpos 映射的 C++ 核心转换逻辑代码模板吗？

---

# 那第二步是什么

既然已经明确了“武器库”和数据集，**第二步的核心目标**是：在你的 4070 笔记本上完成“影子世界”的逻辑闭环（离线原型验证）。

在把代码搬到手机上之前，你必须先在 PC 端证明：输入一串骨骼点坐标，MuJoCo 能实时计算出受力，并且 GRU 模型能准确预判未来。

## 1. 环境搭建：4070 算力全开

- **开发语言**：Python 3.10+（离线开发最快）
- **核心库安装**：
  ```bash
  pip install mujoco  # DeepMind 官方版
  pip install torch torchvision  # 利用 4070 的 CUDA 加速
  pip install onnx onnxruntime  # 为后期转 ncnn 做准备
  ```
- **数据集准备**：下载 TotalCapture 的 Real Data 部分

## 2. 核心魔改：建立“姿态 → 受力”的映射 (IK 算法)

这是你方案中最具技术深度的部分。AIUnit 给的是 3D 坐标 $p$，但 MuJoCo 需要的是关节角度 $q$。

- **具体工具**：使用 MuJoCo 自带的 `mj_inverse` 或第三方库 DeepMimic 的映射逻辑
- **魔改任务**：
  1. 加载 humanoid.xml
  2. 编写一个 Python 脚本，将 TotalCapture 中的骨骼点数据“对齐”到 MuJoCo 模型上
- **目标效果**：在 MuJoCo 仿真器（窗口）中，虚拟人能完美复现数据集里的动作

## 3. 训练“预判大脑”：GRU 模型开发

利用你的 4070 显卡训练一个 Seq2Seq 预测模型。

- **模型结构**：
  - 输入：过去 20 帧的关节角度（$q$ 维）
  - 输出：未来 10 帧的预测角度（$q$ 维）
- **数据集**：使用 TotalCapture 的行走、跑步、深蹲序列
- **具体代码仓库参考**：[MotionPrediction](https://github.com/una-dinosauria/motion-prediction)（这是一个非常经典的姿态预测开源仓库）
- **训练指标**：确保 0.3s 内的预测误差（MPJPE）在 20mm 以内

## 4. 导出与量化：打通“手机端”的桥梁

模型练好后，不能直接在手机上跑 PyTorch，必须瘦身。

- **操作流程**：
  1. `torch.onnx.export` 将模型导出为 .onnx
  2. 使用 ncnn 的工具 `onnx2ncnn` 转换为 .param 和 .bin
- **关键**：进行 FP16 量化，确保在 OPPO 手机上运行耗时控制在 5ms 以内

## 本阶段提交物清单

1. **一个 Python 脚本**：能读取动作数据，在 MuJoCo 里实时显示并输出“ACL 剪切力”曲线
2. **一个 .bin 模型文件**：已经过 ncnn 转换的预判模型
3. **一份对照表**：AIUnit 33 个点对应 MuJoCo 模型的具体索引表（比如：AIUnit 第 11 号点 → MuJoCo left_knee 关节）

由于你没有数据，你现在最需要的是那份“AIUnit 33 骨骼点到 MuJoCo 关节映射表”吗？我可以帮你根据官方文档梳理出来。

---

# 下一步是什么

在完成 PC 端的逻辑验证（离线原型）后，**第三步是**：真机集成与“端云协同”链路调通。

这一步的目标是把你在 4070 笔记本上练好的“大脑”（模型）和“身体”（MuJoCo）塞进 OPPO 真机里，让它能实时跑起来。

## 1. 核心仓库：移动端“移植”包

你需要在 Android Studio 中集成以下关键开源库：

| 模块 | 开源库 | 魔改点 |
|------|--------|--------|
| 物理仿真（JNI 层） | mujoco-android | 将最新的 MuJoCo 3.x C++ 头文件和 libmujoco.so 重新编译进去，以支持更复杂的 humanoid.xml |
| 模型推理 | ncnn-android-skeleton | 把生成的 gru_pred.param 和 bin 放进 assets 文件夹，调用手机的 Adreno GPU 进行加速 |

## 2. 具体魔改任务：AIUnit 实时接入

这是决定“实时性”的关键，你需要编写 Bridge 代码：

1. **获取数据**：调用 OPPO AIUnit SDK 的 FacePose 或 BodyTracking API
   - 具体 API：`BodyTracker.detect()` 得到 33 个骨骼点坐标

2. **坐标转换逻辑（核心代码）**：
   - 将 AIUnit 的相机坐标系 $p$ 归一化
   - 编写一个 C++ 函数 `syncPoseToMuJoCo`，将这 33 个点映射到 MuJoCo 的 `mj_data->qpos`
   - **魔改目标**：解决“镜像问题”和“人体比例缩放问题”，确保仿真人不会因为用户高矮胖瘦而“散架”

## 3. 交互层开发：Aqua Dynamics 模拟

由于 ColorOS 16 的原生流体云 API 可能还没完全开放，你需要自己写一个 GLSL Shader（着色器）：

- **推荐工具**：Android-GLES-Canvas 或直接在 SurfaceView 上画
- **逻辑魔改**：
  1. 根据 MuJoCo 输出的 `mj_data->qfrc_constraint`（约束力/剪切力）计算数值
  2. 将数值传给 Shader：`force_low` -> 蓝色 (静谧)；`force_high` -> 红色 (渗透)
- **效果**：手机屏幕边缘像水滴一样根据你的动作受力实时变色

## 本阶段技术细节清单（你需要搜索的）

- **OPPO 开发者文档**：搜索 AIUnit 的 Skeleton 106/33 点位定义图（确认点位索引）
- **MuJoCo 文档**：搜索 `mj_step` 在 Android NDK 下的性能开销（确保 60Hz 刷新率不卡顿）
- **Android NDK**：配置 CMakeLists.txt 以链接 `libmujoco.so` 和 `libncnn.so`

💡 这里的关键难点是：如何保证 AIUnit 的输出频率（通常 30fps）和 MuJoCo 的仿真频率（60Hz 或更高）同步？

需要我为你提供一份 CMakeLists.txt 的配置模板，教你如何在 Android 项目里同时引入 MuJoCo 和 ncnn 吗？

---

# 下一步呢

在完成 Android 端的“模型集成”与“算法跑通”后，**第四步（也是最后一步）是**：系统级交互对齐与“赛题化”包装。

这一步的目标是让你的 Demo 从一个“能跑的 App”升级为符合 OPPO 赛题要求的“原生具身智能系统”，拿稳那 40 分的创新分和 20 分的商业价值分。

## 1. 深度对齐：Aqua Dynamics (流体云) 原生体验模拟

赛题强调“轻量化、无意识引导”。你需要将 MuJoCo 计算出的“ACL 剪切力”转化为 ColorOS 16 风格的交互。

- **具体工具**：[Lottie-Android](https://github.com/airbnb/lottie-android)（用于高性能流体动画渲染）
- **魔改逻辑**：
  - 不要用传统的进度条或数字显示受力
  - **实现方案**：预制一组流体动画 JSON 文件。当 `acl_load < 0.5` 时，播放“静谧蓝边缘流动”；当 `0.8 < acl_load` 时，触发“红色向中心渗透”效果
- **加分项**：调用 VibrateUnit (震动 SDK)。在 0.3s 预判到风险的一瞬间，触发一个“由轻到重”的线性震动，模拟韧带紧绷感

## 2. 核心魔改：情绪与生理反馈闭环 (HRV 接入)

利用你提到的 OPPO Watch 传感器数据，实现文档中的“情绪感知”。

- **数据来源**：调用 OPPO Health SDK 获取实时心率（HR）和心率变异性（HRV）
- **魔改逻辑**：
  - 如果 HRV 下降（代表用户因疼痛或紧张而憋气），系统自动放缓流体云的波动频率
  - 话术准备：告诉评委这叫“呼吸节律引导”，能有效降低康复过程中的二次伤害风险

## 3. 性能优化：利用双 NPU / 异构计算

为了拿到技术深度的 7 分，你需要证明你压榨了手机性能。

- **具体魔改**：
  - **AIUnit (NPU)**：负责 33 骨骼点识别
  - **ncnn (GPU/Adreno)**：负责 0.3s GRU 预判推理
  - **MuJoCo (CPU Multi-thread)**：负责物理动力学仿真
- **目标**：在远程真机上跑出稳定的 60FPS。你可以用 Android Profiler 截一张 CPU/GPU 负载图放进 PPT 里

## 4. 最终材料：PPT 与演示视频的“核弹级”润色

根据评分标准中“创新性”的要求，你的视频和文档需要包含：

- **核心金句**：“我们不只看到你现在的动作，我们在影子世界里预演你的未来。”
- **对比实验**：录制一段视频，左边是“普通 App 事后纠错（延迟大）”，右边是你的“RehabGuardian 9.0 事前 0.3s 预警（无感引导）”
- **数据集声明**：明确标注使用了 TotalCapture 和 CAMS-Knee 开源数据集进行验证，体现学术严谨性

---

## 🏁 你的 21 天执行路线图总览

| 阶段 | 时间 | 核心任务 |
|------|------|----------|
| **Day 1-7** (PC端) | 第1周 | 4070 训练 GRU 模型 + MuJoCo 物理对齐 |
| **Day 8-14** (移动端) | 第2周 | JNI 封装 MuJoCo + ncnn 集成 + AIUnit 接入 |
| **Day 15-21** (体验端) | 第3周 | 流体云 Shader 编写 + 震动反馈联调 + 视频/PPT 封包 |

现在，你是不是需要我帮你写一段具体的 C++ 代码，展示如何通过 JNI 把 AIUnit 的 33 个浮点数坐标传给 MuJoCo 的 `mj_data->qpos`？这是最难跨越的技术坑。
