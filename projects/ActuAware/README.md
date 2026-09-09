# 动悟 - Android 本地 AI 助手

基于 llama.cpp 的 Android 本地多模态 AI 助手应用，支持文本对话、图片分析、骨骼点检测、联网搜索、本地知识库和对话历史管理功能。

## 功能特性

- **本地 AI 对话** - 基于 llama.cpp 的本地大语言模型，数据不上传云端
- **本地知识库** - 支持 assets 目录加载自定义知识库，增强 AI 回答准确性
- **图片分析** - 支持多模态模型进行图片理解和分析
- **骨骼点检测** - 使用 ML Kit 实时检测人体姿态，绘制骨骼点
- **联网搜索** - 集成 Bing 搜索，支持 RAG 检索增强生成
- **流式输出** - AI 回复采用逐字显示效果，智能滚动
- **Markdown 渲染** - 支持 Markdown 格式渲染，包括表格、任务列表等
- **对话历史** - 支持保存和加载对话历史记录
- **参数可调** - 支持温度、top_p、top_k 等生成参数调整
- **性能自适应** - 自动检测设备性能并优化配置

## 技术栈

| 项目 | 说明 |
|------|------|
| 开发语言 | Kotlin + C++ (JNI) |
| UI 框架 | ViewBinding + Material Design |
| 模型引擎 | llama.cpp |
| 骨骼检测 | ML Kit Pose Detection |
| 最低 SDK | 24 (Android 7.0) |
| 目标 SDK | 34 (Android 14) |
| NDK 版本 | 26.1.10909125 |

## 项目结构

```
app/src/main/
├── java/com/erlab/actuaware/
│   ├── MainActivity.kt           # 主活动
│   ├── ActuAwareApplication.kt   # 应用入口
│   ├── ChatAdapter.kt            # 聊天适配器
│   ├── HistoryAdapter.kt         # 历史记录适配器
│   ├── HistoryHelper.kt          # 历史记录管理
│   ├── NetworkHelper.kt          # 网络请求
│   ├── ToolManager.kt            # 工具管理器
│   ├── ToolChatManager.kt        # 工具聊天管理
│   ├── PerformanceManager.kt     # 性能管理
│   └── ImageCacheManager.kt      # 图像缓存管理
├── cpp/
│   ├── CMakeLists.txt            # CMake 配置
│   ├── llama_jni.cpp             # JNI 接口
│   └── llama.cpp/                # llama.cpp 源码
└── assets/
    └── knowledge/                # 本地知识库目录
        ├── 说明.txt
        ├── 常见问题.md
        └── 帮助文档.md
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

3. 准备知识库文件（可选）
```bash
# 将知识库文件放入 assets/knowledge/ 目录
# 支持 .txt 和 .md 格式
cp your_knowledge.md app/src/main/assets/knowledge/
```

4. 准备模型文件
   - 下载 GGUF 格式的模型文件（如 Qwen、Llama 等）
   - 如需图片分析，下载对应的 mmproj 文件
   - 将模型文件放置到设备存储

5. 编译项目
```bash
./gradlew assembleDebug
```

6. 安装到设备
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 使用说明

### 加载模型

1. 打开应用，点击底部导航栏"设置"页面
2. 选择模型文件和多模态文件（可选）
3. 等待模型加载完成
4. 可在设置页面调整温度、top_p、top_k 等参数

### 本地知识库

知识库用于增强 AI 的回答能力，将特定领域的知识注入到系统提示词中。

**添加知识库文件：**

将 `.txt` 或 `.md` 文件放入 `app/src/main/assets/knowledge/` 目录：

```
app/src/main/assets/knowledge/
├── 产品说明.txt
├── 常见问题.md
└── 使用指南.md
```

**知识库格式示例：**

```markdown
# 产品说明

## 功能介绍
本产品支持以下功能：
- 功能A：描述...
- 功能B：描述...

## 注意事项
- 注意点1
- 注意点2
```

**工作原理：**

加载模型时，应用会自动读取知识库文件并整合到系统提示词中：

```
[用户的系统提示词]

====== 本地知识库 ======
【产品说明.txt】
文件内容...

【常见问题.md】
文件内容...

====== 知识库结束 ======

请优先根据以上知识库内容回答用户问题。
```

### 文本对话

1. 在聊天页面输入问题
2. 点击发送按钮
3. AI 将逐字输出回复

### 图片分析

1. 点击输入框旁边的图片图标
2. 选择图片或拍照
3. AI 将分析图片内容并给出回复

### 动作识别

1. 切换到"识别"页面
2. 选择或拍摄图片
3. 系统自动检测骨骼点并绘制
4. AI 分析动作是否规范并给出建议

### 对话历史

1. 切换到"历史"页面
2. 查看、加载或删除历史对话
3. 当前对话自动保存

## 配置说明

### 系统提示词

默认提示词为"你是一个有用的助手。"，可在设置页面修改。加载模型时，知识库内容会自动追加到系统提示词后。

### 生成参数

| 参数 | 说明 | 范围 |
|------|------|------|
| 温度 (Temperature) | 控制输出的随机性 | 0.0-2.0 |
| Top P | 核采样参数 | 0.0-1.0 |
| Top K | 保留的高概率词数 | 1-100 |
| 最大长度 | 单次回复的最大 token 数 | 128-4096 |
| 上下文大小 | 对话历史保留的 token 数 | 512-8192 |

## 开发说明

### 查看日志

```bash
adb logcat | grep -E "ActuAware|MainActivity|LlamaJNI|erlab.actuaware"
```

### 添加新的 JNI 方法

1. 在 `MainActivity.kt` 声明 native 方法
2. 在 `llama_jni.cpp` 实现对应函数
3. 重新编译项目

### 添加知识库文件

1. 将 `.txt` 或 `.md` 文件放入 `app/src/main/assets/knowledge/`
2. 重新编译安装应用
3. 加载模型时自动加载知识库

## 性能优化

- **Flash Attention**: 默认启用，加速注意力计算
- **NEON 指令集**: ARM 优化，提升推理速度
- **设备自适应**: 根据设备性能自动调整参数
- **图像缓存**: LRU 缓存避免重复解码

## 注意事项

1. **内存占用**: 大模型需要更多内存，建议 6GB RAM 以上设备
2. **网络访问**: 联网功能需要网络连接
3. **隐私安全**: 所有 AI 处理在本地进行，数据不上传（搜索查询除外）
4. **多模态模型**: 图片分析需要支持视觉的多模态模型和 mmproj 文件
5. **知识库大小**: 知识库会占用上下文窗口，建议控制总大小在合理范围内

## 许可证

本项目仅供学习和研究使用。
