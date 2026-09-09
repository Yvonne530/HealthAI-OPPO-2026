# RehabGuardian Android 应用部署指南

本文档详细说明如何从零开始构建并打包 RehabGuardian Android 应用，最终得到可安装的 APK 文件。整个过程无需粘贴大量代码，只需按步骤创建文件并复制内容即可。

---

## 1. 环境准备

### 1.1 开发环境
- **操作系统**：Windows 10/11、macOS 或 Linux（推荐 Ubuntu 20.04+）
- **Android Studio**：最新稳定版（当前推荐 Giraffe 或 Hedgehog）
- **JDK**：Android Studio 自带 JDK 11 或 17，无需额外安装
- **Android SDK**：通过 Android Studio SDK Manager 安装 API 26 及以上版本
- **Git**（可选）：用于版本管理

### 1.2 硬件要求
- 开发机：8 GB RAM 以上，建议 16 GB
- 测试设备：Android 8.0 (API 26) 以上手机，支持 Camera2 接口（绝大部分主流机型）
- 数据线：用于真机调试（推荐开启 USB 调试）

---

## 2. 项目创建

1. 打开 Android Studio，选择 **New Project**。
2. 在模板列表中选择 **Empty Activity**。
3. 填写项目信息：
   - **Name**：`RehabGuardian`
   - **Package name**：`com.rehabguardian`
   - **Save location**：选择本地文件夹
   - **Language**：Kotlin
   - **Minimum SDK**：API 26 (Android 8.0)
4. 点击 **Finish**，等待项目初始化完成。

---

## 3. 文件结构搭建

项目创建后，需要在 `app/src/main/java/com/rehabguardian/` 下建立以下包（Package）：
- `data`：存放数据模型（FrameData、Session、HistoryRecord）
- `ui`：存放界面相关类（RiskRenderer、SessionManager 等）
- `camera`：相机和姿态处理（CameraManager、PoseProcessor）
- `inference`：AI 推理引擎（MNNInferenceEngine、SlidingWindowBuffer、NormalizationParams）
- `analysis`：生物力学分析和风险状态机（RiskStateMachine、BiomechanicsAnalyzer、ReportGenerator）

创建方法：右键点击 `com.rehabguardian` → New → Package，依次输入上述名称。

---

## 4. 依赖配置

编辑 `app/build.gradle` 文件，确保包含以下关键依赖（具体版本以官方最新为准）：
- CameraX (core, camera2, lifecycle, view)
- MediaPipe Tasks Vision
- MNN（需要手动添加 .aar 文件）
- Kotlin Coroutines
- Gson
- Room（可选，用于历史记录持久化）
- MPAndroidChart（可选，用于趋势图表）

> **重要**：MNN 的 `.aar` 文件需从 [MNN GitHub Releases](https://github.com/alibaba/MNN/releases) 下载 Android 版本（如 `MNN-2.0.0.aar`），并放入 `app/libs/` 目录。然后在 `build.gradle` 中添加 `implementation fileTree(dir: 'libs', include: ['*.aar', '*.jar'])`。

配置完成后，点击 **Sync Now** 同步项目。

---

## 5. 代码文件放置

### 5.1 创建 Kotlin 文件
在每个包中创建对应的 `.kt` 文件（右键包 → New → Kotlin Class/File），文件名如下：

| 包 | 文件名 |
|----|--------|
| `data` | `FrameData.kt` |
| `ui` | `RiskRenderer.kt`、`SessionManager.kt`、`SkeletonRenderer.kt`（可选）、`RiskOverlayView.kt`（可选） |
| `camera` | `CameraManager.kt`、`PoseProcessor.kt` |
| `inference` | `MNNInferenceEngine.kt`、`SlidingWindowBuffer.kt`、`NormalizationParams.kt` |
| `analysis` | `RiskStateMachine.kt`、`BiomechanicsAnalyzer.kt`、`ReportGenerator.kt` |
| 根目录 | `MainActivity.kt` |

### 5.2 复制代码
将之前提供的对应 Kotlin 代码完整复制到每个文件中。特别注意：
- `RiskRenderer.kt` 应包含坐标缩放修复、风险进度条、延迟显示等增强功能。
- `MNNInferenceEngine.kt` 应包含 Session 复用、差分特征、Risk 时间序列等修复。
- `SlidingWindowBuffer.kt` 应改为泛型版本。
- `MainActivity.kt` 应包含完整的相机、推理、渲染联动逻辑。

> **注意**：所有代码均需保证包名与创建时一致（`package com.rehabguardian.xxx`）。

---

## 6. 资源文件配置

### 6.1 布局文件
在 `res/layout/` 下创建 `activity_main.xml`（若已存在则覆盖）。内容应包含：
- 一个 `PreviewView` 用于相机预览
- 一个 `RiskRenderer` 自定义视图作为叠加层
- 一个 `TextView` 显示状态信息
- 两个按钮（录制、切换相机）

### 6.2 颜色和主题
在 `res/values/colors.xml` 中定义应用所需颜色（如绿色、橙色、红色等）。主题可直接使用 MaterialComponents 主题，无需修改。

---

## 7. 模型文件放置

所有模型文件需放入 `app/src/main/assets/` 目录（若目录不存在则手动创建）。需要放置以下文件：
- `pose_landmarker_lite.task`：MediaPipe 姿态检测模型（约 12 MB）
- `stgcn.mnn`：ST-GCN 模型（约 10 MB）
- `fno.mnn`：FNO 模型（约 15 MB）
- `risk.mnn`：风险分类模型（约 5 MB）
- `norm_stats.json`：归一化参数（几 KB）

> **获取方式**：模型文件需从训练结果导出，或在项目资产中提供。`pose_landmarker_lite.task` 可从 [MediaPipe 官方](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) 下载。

---

## 8. 权限配置

编辑 `AndroidManifest.xml`，在 `<application>` 标签前添加以下权限：
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" android:maxSdkVersion="28" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.VIBRATE" />
<!-- 可选：OPPO Health SDK 相关权限 -->
<uses-permission android:name="com.heytap.permission.HEALTH_READ" />
<uses-permission android:name="com.heytap.permission.HEALTH_WRITE" />
```

并在 `<application>` 标签内添加 `android:hardwareAccelerated="true"` 以提升渲染性能。

---

## 9. 构建与调试

### 9.1 真机调试
1. 手机开启 **开发者模式** 和 **USB 调试**。
2. 用数据线连接电脑，手机弹出“允许 USB 调试”时确认。
3. 在 Android Studio 工具栏选择手机设备，点击 **Run** 按钮（绿色三角形）。
4. 应用安装后自动启动，观察相机预览、骨架绘制、风险指示是否正常。

### 9.2 模拟器调试
- 若暂无真机，可创建 API 26 以上的模拟器（建议 x86_64 架构，支持 OpenGL ES 3.0）。
- 注意模拟器摄像头需配置为虚拟或网络摄像头，否则 MediaPipe 可能无法正常检测。

### 9.3 日志查看
在 Android Studio 的 **Logcat** 窗口中过滤 `MNNEngine`、`PoseProcessor` 等标签，查看推理延迟、模型加载状态等信息。

---

## 10. 生成 APK

### 10.1 生成调试版 APK
调试版 APK 可直接安装，但未签名，适合快速测试。操作步骤：
1. 点击菜单 **Build → Build Bundle(s) / APK(s) → Build APK(s)**
2. 等待构建完成，右下角会弹出提示，点击 **locate** 打开文件夹，即可找到 `app-debug.apk`。
3. 将该 APK 传输到手机安装（需允许安装未知来源应用）。

### 10.2 生成发布版 APK（签名）
正式发布需要签名。步骤：
1. 点击菜单 **Build → Generate Signed Bundle / APK**。
2. 选择 **APK**，点击 Next。
3. 创建或选择密钥库（Key store）。若为首次，点击 **Create new**，填写信息后生成 `.jks` 文件。
4. 填写密钥别名、密码，点击 Next。
5. 选择构建类型为 **release**，签名版本选择 **V1 + V2**，点击 Finish。
6. 等待构建完成，最终 APK 位于 `app/release/` 目录。

---

## 11. 常见问题与解决

### 11.1 编译错误：找不到 MNN 类
- 确认 `app/libs/` 目录存在且 `.aar` 文件已放入。
- 检查 `build.gradle` 中是否包含 `implementation fileTree(dir: 'libs', include: ['*.aar', '*.jar'])`。
- 执行 **File → Sync Project with Gradle Files**。

### 11.2 运行时闪退：MediaPipe 模型加载失败
- 确认 `pose_landmarker_lite.task` 文件在 `assets/` 目录下，文件名大小写正确。
- 检查 AndroidManifest 中是否声明了相机权限且已动态申请（MainActivity 中已实现）。

### 11.3 骨架绘制错位或缩放错误
- 确认 `RiskRenderer` 中使用的是归一化坐标（MediaPipe 默认输出为 0~1），并正确乘以屏幕宽高。
- 若使用世界坐标，需额外转换，建议保持归一化坐标。

### 11.4 推理延迟过高（> 50ms）
- 确认 MNN Session 已复用，而非每帧创建。
- 检查 `MNNInferenceEngine` 中的线程数设置是否合理（一般 4 线程）。
- 若设备性能不足，可考虑降低输入分辨率（CameraX 中设置更小的目标分辨率）。

### 11.5 录制时内存增长过快
- `SessionManager` 中的 `replayBuffer` 限制了最大帧数（根据 FPS 动态计算），不会无限增长。
- 长时间录制时，可考虑将历史会话写入数据库，释放内存。

---

## 12. 总结

遵循以上步骤，你应能成功构建 RehabGuardian 应用并生成 APK。该应用集成了 MediaPipe 姿态估计、MNN 端侧推理、生物力学分析和实时风险评估，可在 Android 手机上实现 ACL 损伤风险的实时监测与分析。如在部署过程中遇到未列出的问题，可查看 Logcat 日志或查阅相关依赖的官方文档。
