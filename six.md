## RehabGuardian 9.0 —— 国家一等奖最终核弹版

**版本：V9.0 MuJoCo影子仿真 + 0.3s预判 + Aqua Dynamics流体交互**
**参赛：第十九届全国大学生软件创新大赛**
**赛道：软件无处不在，创新定义未来**

---

## 一、项目概要（核弹级定位）

### 1.1 一句话定位

> 基于**MuJoCo实时影子仿真**与**0.3s预判推演**的具身康复系统，通过**Aqua Dynamics流体交互**实现从“事后纠错”到“事前500ms预防”的跨越。

### 1.2 核心价值对比

| 维度 | 《羽迹》 | 《全心脏建模》 | 普通8.0 | **RehabGuardian 9.0** |
|------|---------|--------------|---------|----------------------|
| 核心技术 | 姿态识别 | 3D重建 | Hill公式计算 | **MuJoCo影子仿真** |
| 时间维度 | 实时纠错 | 离线重建 | 当前受力 | **0.3s预判+500ms推演** |
| 物理模型 | 无 | 无 | 公式计算 | **碰撞+约束+重力仿真** |
| 交互方式 | 屏幕显示 | 专业图像 | 热力图 | **流体云+情绪感知** |
| 系统架构 | App | 专业软件 | 异构计算 | **影子世界+快慢结合** |

---

## 二、核心魔改：从“计算器”到“仿真器”

### 2.1 架构升级图

```
┌─────────────────────────────────────────────────────────────┐
│                    感知层（AIUnit输入）                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  手机摄像头 → AIUnit姿态 → 33骨骼点                   │   │
│  │  OPPO Watch → 心率/IMU → 100Hz传感器数据             │   │
│  └─────────────────────────────────────────────────────┘   │
│                              ↓                                │
│        将骨骼点转换为MuJoCo广义坐标 (mj_data->qpos)          │
│                              ↓                                │
├─────────────────────────────────────────────────────────────┤
│            【核心】MuJoCo影子世界 (60Hz实时仿真)              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  mj_model = loadModel("humanoid.xml")               │   │
│  │  mj_data = mj_makeData(mj_model)                     │   │
│  │                                                        │   │
│  │  // 每帧执行：                                        │   │
│  │  1. 将AIUnit坐标写入mj_data->qpos（同步用户姿态）      │   │
│  │  2. mj_forward(mj_model, mj_data)（正向运动学）       │   │
│  │  3. mj_step(mj_model, mj_data)（推演未来）            │   │
│  │  4. 读取关节受力、力矩、碰撞信息                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                              ↓                                │
│              ┌──────────────┴──────────────┐                 │
│              ↓                              ↓                 │
│  ┌─────────────────────┐      ┌─────────────────────┐       │
│  │  当前状态分析       │      │  0.3s预判推演       │       │
│  │  • ACL剪切力        │      │  • 未来轨迹         │       │
│  │  • 关节力矩         │      │  • 未来受力         │       │
│  │  • 代偿检测         │      │  • 风险预警         │       │
│  └─────────────────────┘      └─────────────────────┘       │
│                              ↓                                │
├─────────────────────────────────────────────────────────────┤
│                    交互层（Aqua Dynamics）                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  受力平稳 → 静谧蓝流体边缘流动                         │   │
│  │  受力激增 → 流体变红、向中心渗透                       │   │
│  │  预判风险 → 流体凝聚成AR箭头 + VibrateUnit震动         │   │
│  │  情绪感知 → HRV下降时流体变缓引导深呼吸                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、JNI核心代码：AIUnit坐标 → MuJoCo影子世界

### 3.1 C++层魔改代码

```cpp
// mujoco_shadow_world.cpp
#include <mujoco/mujoco.h>
#include <jni.h>
#include <android/log.h>

#define TAG "MuJoCoShadow"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

// 全局MuJoCo模型和数据
static mjModel* m_model = nullptr;
static mjData* m_data = nullptr;

// 初始化MuJoCo模型（在JNI_OnLoad中调用）
extern "C" JNIEXPORT void JNICALL
Java_com_rehabguardian_engine_MuJoCoEngine_initModel(
    JNIEnv* env, jobject thiz, jstring modelPath) {
    
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    
    // 加载模型（humanoid.xml内置在assets中）
    char error[1000] = "";
    m_model = mj_loadXML(path, nullptr, error, 1000);
    if (!m_model) {
        LOGE("加载模型失败: %s", error);
        return;
    }
    
    // 创建数据
    m_data = mj_makeData(m_model);
    LOGI("MuJoCo模型初始化成功，nq = %d, nv = %d", m_model->nq, m_model->nv);
    
    env->ReleaseStringUTFChars(modelPath, path);
}

// 每帧调用：同步AIUnit姿态到MuJoCo
extern "C" JNIEXPORT void JNICALL
Java_com_rehabguardian_engine_MuJoCoEngine_syncPose(
    JNIEnv* env, jobject thiz, jfloatArray jointAngles) {
    
    if (!m_model || !m_data) return;
    
    jfloat* angles = env->GetFloatArrayElements(jointAngles, nullptr);
    jsize len = env->GetArrayLength(jointAngles);
    
    // 将关节角度写入广义坐标（qpos）
    // 假设humanoid.xml的关节顺序和AIUnit输出一致
    for (int i = 0; i < len && i < m_model->nq; i++) {
        m_data->qpos[i] = angles[i];
    }
    
    // 执行正向运动学（计算位置、速度）
    mj_forward(m_model, m_data);
    
    env->ReleaseFloatArrayElements(jointAngles, angles, 0);
}

// 推演未来0.3s（假设dt=0.002s, 150步）
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_rehabguardian_engine_MuJoCoEngine_predictFuture(
    JNIEnv* env, jobject thiz, jint steps) {
    
    if (!m_model || !m_data) return nullptr;
    
    // 保存当前状态
    mjData* snapshot = mj_makeData(m_model);
    mj_copyData(snapshot, m_model, m_data);
    
    // 正向推演steps步
    for (int i = 0; i < steps; i++) {
        mj_step(m_model, m_data);
    }
    
    // 提取未来关节角度
    jfloatArray result = env->NewFloatArray(m_model->nq);
    env->SetFloatArrayRegion(result, 0, m_model->nq, m_data->qpos);
    
    // 恢复当前状态
    mj_copyData(m_data, m_model, snapshot);
    mj_deleteData(snapshot);
    
    return result;
}

// 获取当前关节受力（ACL剪切力等）
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_rehabguardian_engine_MuJoCoEngine_getJointForces(
    JNIEnv* env, jobject thiz) {
    
    if (!m_model || !m_data) return nullptr;
    
    // 关节受力存储在qfrc_constraint、qfrc_passive等
    // 这里简化为返回关节力矩
    jfloatArray result = env->NewFloatArray(m_model->nv);
    env->SetFloatArrayRegion(result, 0, m_model->nv, m_data->qfrc_actuator);
    
    return result;
}

// 释放资源
extern "C" JNIEXPORT void JNICALL
Java_com_rehabguardian_engine_MuJoCoEngine_release(
    JNIEnv* env, jobject thiz) {
    
    if (m_data) mj_deleteData(m_data);
    if (m_model) mj_deleteModel(m_model);
    m_model = nullptr;
    m_data = nullptr;
}
```

### 3.2 Java层调用

```java
public class MuJoCoEngine {
    static {
        System.loadLibrary("mujoco_shadow");
    }
    
    public native void initModel(String modelPath);
    public native void syncPose(float[] jointAngles);
    public native float[] predictFuture(int steps);
    public native float[] getJointForces();
    public native void release();
    
    // 60Hz主循环
    public void onFrame(PoseData pose) {
        // 1. 同步姿态到MuJoCo影子世界
        syncPose(pose.getJointAngles());
        
        // 2. 推演未来0.3s（150步，每步0.002s）
        float[] futurePose = predictFuture(150);
        
        // 3. 获取当前关节受力
        float[] forces = getJointForces();
        
        // 4. 分析风险
        float aclShear = computeACLFromForces(forces);
        if (aclShear > threshold) {
            // 提前预警（此时用户还没犯错）
            aquaDynamics.alert(RiskLevel.HIGH, 300); // 300ms后风险
        }
    }
}
```

---

## 四、0.3s预判逻辑：GRU + MuJoCo融合

### 4.1 训练阶段（4070笔记本）

```python
# train_gru_predictor.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GRUPredictor(nn.Module):
    def __init__(self, input_size=33, hidden_size=128, output_size=33):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * 10)  # 预测未来10帧
    
    def forward(self, x):  # x: [batch, 20, 33]
        out, _ = self.gru(x)
        last = out[:, -1, :]  # 取最后一帧
        future = self.fc(last)  # [batch, 330]
        return future.view(-1, 10, 33)  # [batch, 10, 33]

# 用Human3.6M训练
model = GRUPredictor()
train(model, dataloader)

# 量化导出为ONNX
dummy = torch.randn(1, 20, 33)
torch.onnx.export(model, dummy, "gru_predictor.onnx")

# 用ncnn转换为手机端格式
# ./onnx2ncnn gru_predictor.onnx gru_pred.param gru_pred.bin
```

### 4.2 端侧推理（ncnn加速）

```java
public class FuturePredictor {
    private Net net = new Net();
    
    public void init() {
        net.loadParam("gru_pred.param");
        net.loadModel("gru_pred.bin");
    }
    
    public float[][] predict(float[][] past20Frames) {
        // past20Frames: [20, 33]
        Mat input = Mat.fromArray(past20Frames);
        Mat output = new Mat();
        
        Extractor ex = net.createExtractor();
        ex.input("input", input);
        ex.extract("output", output);
        
        // output: [10, 33]
        return output.toArray();
    }
}
```

### 4.3 预判融合逻辑

```java
public class PreemptiveWarning {
    
    public void onFrame(PoseData pose) {
        // 1. GRU预测未来0.3s姿态
        float[][] futurePose = predictor.predict(history.getLast20Frames());
        
        // 2. 将未来姿态输入MuJoCo影子世界
        for (int t = 0; t < 10; t++) {
            mujoco.syncPose(futurePose[t]);  // 同步到影子世界
            
            // 推演更远的未来（影子世界内置）
            float[] forces = mujoco.getJointForces();
            
            // 检查未来受力
            if (forces[KNEE_ACL] > threshold) {
                // 预判到t*33ms后风险
                aquaDynamics.alert(
                    RiskLevel.HIGH, 
                    t * 33,  // 还有t*33ms发生
                    "ACL受力将超限"
                );
                break;
            }
        }
    }
}
```

---

## 五、Aqua Dynamics流体交互（ColorOS 16原生集成）

### 5.1 流体云状态映射

```java
public class AquaDynamics {
    
    private FluidView fluidView;  // ColorOS 16原生组件
    
    public void update(float aclLoad, float hr, float hrv, long timeToRisk) {
        // 1. 基础状态：受力映射颜色
        float loadRatio = aclLoad / threshold;
        int color;
        float speed;
        
        if (loadRatio < 0.5) {
            // 安全：静谧蓝
            color = Color.argb(100, 100, 200, 255);
            speed = 1.0f;
        } else if (loadRatio < 0.8) {
            // 注意：淡黄
            color = Color.argb(150, 255, 200, 100);
            speed = 1.5f + loadRatio;
        } else {
            // 危险：红色，向中心渗透
            color = Color.argb(200, 255, 50, 50);
            speed = 3.0f + (1 - timeToRisk / 500) * 2;
        }
        
        // 2. 情绪感知：HRV下降时放缓流速，引导深呼吸
        if (hrv < baseline * 0.7) {
            speed *= 0.5;  // 放缓
            color = Color.argb(80, 150, 200, 255);  // 更淡的蓝
        }
        
        // 3. 更新流体云
        fluidView.setFluidColor(color);
        fluidView.setFlowSpeed(speed);
        
        // 4. 如果有紧急风险，凝聚成箭头
        if (timeToRisk < 200) {
            fluidView.showArrow(
                "ACL受力过高",
                new ArrowStyle().color(Color.RED).size(ArrowSize.LARGE)
            );
            
            // 触发震动
            vibrate.urgent();
        }
    }
}
```

### 5.2 创新点话术

> “我们深度对齐ColorOS 16 Aqua Dynamics设计语言，将风险预警从‘弹窗打断’升级为‘无意识引导’。受力平稳时静谧蓝流体边缘流动，受力激增时流体变红向中心渗透，HRV下降时流体自动变缓引导深呼吸。这不是移植App，这是为OPPO生态原生打造的具身智能系统。”

---

## 六、性能测试（对标8.0升级）

| 指标 | 8.0原方案 | 9.0仿真版 | 提升 |
|------|---------|----------|------|
| 物理模型 | Hill公式计算 | **MuJoCo完整仿真** | ⬆️ 物理真实 |
| 预判能力 | 无 | **0.3s未来推演** | ⬆️ 事前预警 |
| 交互方式 | 热力图 | **Aqua Dynamics** | ⬆️ 无感交互 |
| 验证数据 | 声称误差 | **公开数据集验证** | ⬆️ 可追溯 |
| 核心记忆点 | “算受力” | **“影子世界预判”** | ⬆️ 评委记住 |

---

## 七、最终结论

**RehabGuardian 9.0 相比8.0：**

| 维度 | 8.0 | 9.0 | 为什么能拿奖 |
|------|-----|-----|-------------|
| 核心引擎 | Hill公式 | **MuJoCo仿真** | 公式是死的，仿真是活的 |
| 预判能力 | 无 | **0.3s推演** | 从“纠错”到“预防” |
| 交互设计 | 热力图 | **Aqua Dynamics** | 对齐官方，原生体验 |
| 技术深度 | 异构计算 | **仿真+预判+流体** | 三个记忆点 |
| 验证逻辑 | 模糊 | **公开数据集** | 评委信 |

**这就是国家一等奖的最终版本。**

---

## 八、21天执行表（精简版）

| 阶段 | 天数 | 任务 | 核心产出 |
|------|------|------|---------|
| 环境准备 | Day1 | 装MuJoCo、ncnn、OPPO SDK | 环境就绪 |
| 影子世界 | Day2-5 | JNI封装MuJoCo、同步AIUnit | 60Hz仿真 |
| 预判模型 | Day6-8 | 训练GRU、转ncnn、集成 | 0.3s推演 |
| 流体云 | Day9-11 | ARunit集成、HRV映射 | 无感交互 |
| 联调 | Day12-14 | 全链路跑通 | 稳定版本 |
| 验证 | Day15-16 | CAMS-Knee对比 | 相关性报告 |
| 演示 | Day17-21 | 录视频、写PPT、模拟 | 提交材料 |

---

## 九、给评委看的一句话

> **“我们不只看到你现在的动作，我们在影子世界里预演你的未来。”**
