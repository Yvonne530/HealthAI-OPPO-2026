# 动悟AndroidAPP

基于 llama.cpp 的 Android 本地多模态 AI 助手应用，支持文本对话、图片分析、联网搜索和对话历史管理功能。

## 功能特性

- 🤖 **本地 AI 对话** - 基于 llama.cpp 的本地大语言模型，数据不上传云端
- 📷 **图片分析** - 支持多模态模型进行图片理解和分析
- 🌐 **联网搜索** - 集成 Bing 搜索，支持 RAG 检索增强生成
- 💬 **流式输出** - AI 回复采用逐字显示效果，提供更好的交互体验
- 📝 **Markdown 渲染** - 支持 Markdown 格式渲染，包括表格、任务列表等
- 💾 **对话历史** - 支持保存和加载对话历史记录
- 📁 **模型管理** - 支持多模型管理和快速切换
- ⚙️ **参数可调** - 支持温度、top_p、top_k 等生成参数调整
- 🎨 **Material Design** - 采用 Material Design 设计风格，界面美观

## 技术栈

- **开发语言**: Kotlin (Android)
- **原生代码**: C++ (CMake)
- **UI 框架**: ViewBinding + Material Design
- **模型引擎**: llama.cpp
- **最低 SDK**: 24 (Android 7.0)
- **目标 SDK**: 34 (Android 14)
- **NDK 版本**: 26.1.10909125
- **CMake 版本**: 3.22.1

## 项目结构

```
app/
├── src/main/
│   ├── java/com/erlab/actuaware/
│   │   ├── MainActivity.kt          # 主活动
│   │   ├── ChatAdapter.kt           # 聊天适配器
│   │   ├── HistoryAdapter.kt        # 历史记录适配器
│   │   ├── HistoryHelper.kt         # 历史记录管理工具
│   │   ├── ModelAdapter.kt          # 模型适配器
│   │   ├── ModelFile.kt             # 模型文件数据类
│   │   ├── ModelLoadProgressDialog.kt  # 模型加载进度对话框
│   │   ├── NetworkHelper.kt         # 网络请求工具
│   │   ├── ToolChatManager.kt       # 工具聊天管理器
│   │   └── ToolManager.kt           # 工具管理器
│   ├── cpp/
│   │   ├── CMakeLists.txt           # CMake 配置
│   │   ├── llama_jni.cpp            # JNI 接口层
│   │   └── llama.cpp/               # llama.cpp 源码
│   ├── res/                         # 资源文件
│   └── AndroidManifest.xml
└── build.gradle.kts
```

## 快速开始

### 环境要求

- Android Studio Hedgehog | 2023.1.1 或更高版本
- NDK 26.1.10909125
- CMake 3.22.1
- JDK 8 或更高版本
- Android 设备（建议 6GB RAM 以上）

### 编译步骤

1. 克隆仓库
```bash
git clone git@github.com:Yvonne530/HealthAI-OPPO-2026.git
cd HealthAI-OPPO-2026
```

2. 下载 llama.cpp 源码
```bash
cd app/src/main/cpp
git clone https://github.com/ggerganov/llama.cpp.git
```

3. 准备模型文件
   - 下载 GGUF 格式的模型文件和多模态文件（可选）（推荐 Qwen2.5 系列）
   - 将模型文件放置到设备存储或应用目录
   - 在应用中通过设置页面加载模型和多模态文件

4. 编译项目
```bash
./gradlew assembleDebug
```

5. 安装到设备（需有ADB）
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 使用说明

### 加载模型

1. 打开应用，点击底部导航栏“设置”页面
2. 依提示加载模型文件和多模态文件
3. 等待模型加载完成（进度条显示加载进度）
4. 可以在设置页面设置温度、top_p、top_k 等参数

### 文本对话

1. 在输入框输入问题
2. 点击发送按钮
3. AI 将逐字输出回复
4. 可随时点击停止按钮中止生成

### 图片分析

1. 点击输入框旁边的图片图标
2. 选择图片或拍照
3. AI 将分析图片内容并给出回复

### 联网搜索

1. 在侧边栏点击"联网设置"
2. 开启联网功能
3. 发送问题时，系统会自动搜索相关信息并整合到回复中

### 对话历史

1. 点击侧边栏"历史记录"
2. 可查看、加载或删除历史对话
3. 当前对话自动保存


## 配置说明

### 系统提示词

默认提示词为"你是一个有用的助手。"，可通过侧边栏"设置系统提示词"修改。系统提示词会影响 AI 的回复风格和角色设定。

### 联网设置

- 搜索引擎: Bing (cn.bing.com)
- 返回结果数: 5 条
- 可通过侧边栏"联网设置"独立控制开关

### 详细参数说明

- **温度 (Temperature)**: 控制输出的随机性，范围 0.0-2.0，值越高越随机
- **Top P**: 核采样参数，范围 0.0-1.0，控制输出的多样性
- **Top K**: 保留的高概率词数，限制候选词数量
- **最大长度**: 单次回复的最大 token 数
- **上下文大小**: 对话历史保留的 token 数


## 开发说明

在电脑上安装了ADB并连接手机后，可通过以下命令查看应用调试日志：

```Bash
adb logcat | grep -E "MainActivity|llama"
```
 


### 修改搜索引擎

编辑 `NetworkHelper.kt` 中的搜索 URL 和解析逻辑，当前使用 Bing 搜索 API。

### 添加新的 JNI 方法

1. 在 `MainActivity.kt` 声明 native 方法
2. 在 `llama_jni.cpp` 实现对应函数
3. 重新编译项目

### 修改 UI 样式

编辑 `res/drawable/bg_chat_bubble_*.xml` 修改聊天气泡样式，编辑 `res/layout/` 修改布局文件。

### 添加新功能

参考现有代码结构，主要文件：
- `MainActivity.kt`: 主界面逻辑
- `ChatAdapter.kt`: 聊天列表适配器
- `NetworkHelper.kt`: 网络请求处理
- `ToolManager.kt`: 工具调用管理

## 性能优化

- **Flash Attention**: 默认启用，加速注意力计算
- **GPU 加速**: 支持 Vulkan（需兼容设备）
- **批处理大小**: 512 tokens，平衡速度和内存
- **多线程**: 根据设备 CPU 核心数自动调整

## 注意事项

1. **内存占用**: 大模型需要更多内存，建议 6GB RAM 以上设备
2. **网络访问**: 联网功能需要网络连接，可能产生流量费用
3. **隐私安全**: 所有 AI 处理在本地进行，数据不上传（搜索查询除外）
4. **多模态模型**: 图片分析需要支持视觉的多模态模型

## 更新日志

### v1.0
- 初始版本发布
- 支持本地 AI 对话
- 支持图片分析
- 支持联网搜索
- 支持对话历史管理
- 支持多模型管理
- 支持 Markdown 渲染
- 支持流式输出

## 后续工作

- [ ] 图片分析结果使用markdown渲染
- [ ] 添加文件读取和写入工具（或者，一部分功能使用OPPO内置的端侧AI能力而不是用户自己选择的模型？）
- [ ] 添加TTS朗读功能
- [ ] 添加骨骼点识别并显示在图片中功能
- [ ] 添加视频分析功能（如果模型本身支持视频输入，应该比较容易实现）
- [ ] 添加运动建模功能
- [ ] 优化UI界面，并添加用户默认头像、模型头像以及APP图标等等
- [ ] 优化模型推理效率
