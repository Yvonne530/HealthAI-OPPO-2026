# 项目指南：动悟 (Llama.cpp Android 多模态 AI 助手)

## 项目概述

动悟是一个基于 llama.cpp 的 Android 本地多模态 AI 助手应用，支持文本对话、图片分析、联网搜索和对话历史管理功能。

## 核心技术栈

- **语言**: Kotlin (Android)
- **原生代码**: C++ (CMake)
- **UI**: ViewBinding + Material Design
- **模型引擎**: llama.cpp (本地大语言模型)
- **多模态**: mtmd (多模态处理)
- **联网**: Bing 搜索 API
- **Markdown**: Markwon 库

## 项目结构

```
app/
├── src/main/
│   ├── java/com/erlab/actuaware/
│   │   ├── MainActivity.kt              # 主活动
│   │   ├── ChatAdapter.kt               # 聊天气泡适配器
│   │   ├── HistoryAdapter.kt            # 历史记录适配器
│   │   ├── HistoryHelper.kt             # 历史记录管理工具
│   │   ├── ModelAdapter.kt              # 模型适配器
│   │   ├── ModelFile.kt                 # 模型文件数据类
│   │   ├── ModelLoadProgressDialog.kt   # 模型加载进度对话框
│   │   ├── NetworkHelper.kt             # 网络请求工具类
│   │   ├── ToolChatManager.kt           # 工具聊天管理器
│   │   └── ToolManager.kt               # 工具管理器
│   ├── cpp/
│   │   ├── llama_jni.cpp               # JNI 接口层
│   │   └── llama.cpp/                  # llama.cpp 源码
│   ├── res/
│   │   ├── layout/                     # 布局文件
│   │   │   ├── activity_main.xml
│   │   │   ├── item_chat_message_*.xml # 气泡布局
│   │   │   └── bg_chat_bubble_*.xml    # 气泡背景
│   │   └── drawable/
│   └── AndroidManifest.xml
└── build.gradle.kts
```

## 关键功能

### 1. 聊天气泡界面
- 位置: `MainActivity.kt`, `ChatAdapter.kt`
- 布局: `item_chat_message_user.xml`, `item_chat_message_ai.xml`, `item_chat_message_system.xml`
- 特点: 使用 RecyclerView 实现聊天气泡样式，支持三种消息类型（用户、AI、系统）
- **流式输出**: AI 回复采用模拟流式输出，逐字显示效果
- **元数据显示**: AI 消息下方显示 token 数量和生成用时
- **复制功能**: AI 消息提供一键复制按钮
- **Markdown 渲染**: 支持 Markdown 格式，包括表格、任务列表等

### 2. 图片分析
- JNI 方法: `nativeAnalyzeImage()`
- 流程: 用户发送图片 → JNI 处理 → AI 分析 → 返回结果
- 限制: 需要加载多模态模型和 mmproj 文件

### 3. 联网搜索
- 位置: `NetworkHelper.kt`
- 搜索引擎: Bing (cn.bing.com)
- 方法: `search()` - 搜索查询，`fetchUrl()` - 获取网页内容
- 集成方式: RAG（检索增强生成）- 搜索结果 + 用户问题 → 本地模型
- **设置**: 通过侧边栏"联网设置"按钮独立控制

### 4. 对话历史管理
- 位置: `HistoryHelper.kt`, `HistoryAdapter.kt`
- 功能: 保存、加载、删除对话历史
- 存储: 使用 SharedPreferences 或文件存储
- 界面: 侧边栏"历史记录"按钮

### 5. 模型管理
- 位置: `ModelAdapter.kt`, `ModelFile.kt`, `ModelLoadProgressDialog.kt`
- JNI 方法: `nativeLoadModel()`, `nativeFreeModel()`
- 支持格式: GGUF 格式的 llama.cpp 模型
- 性能优化: Flash Attention, GPU 加速 (Vulkan)
- UI: 模型列表、加载进度对话框

### 6. 工具管理
- 位置: `ToolManager.kt`, `ToolChatManager.kt`
- 功能: 管理和调用各种工具（如联网搜索等）
- 扩展: 可添加新的工具类型

## 重要文件说明

### MainActivity.kt

**关键变量**:
- `enableNetwork`: 联网功能开关（保存到 SharedPreferences）
- `isModelLoaded`: 模型加载状态
- `systemPrompt`: 系统提示词
- `conversationHistory`: 对话历史记录

**关键方法**:
- `sendMessage()`: 发送消息（支持文本、图片、联网、流式输出）
- `simulateStreamingOutput()`: 模拟流式输出效果
- `loadModelInBackground()`: 后台加载模型
- `addMessageToChat()`: 添加消息到聊天界面
- `showNetworkSettingsDialog()`: 联网设置对话框
- `showGenerationParamsDialog()`: 生成参数设置对话框
- `showHistoryDialog()`: 历史记录对话框
- `showModelSelectionDialog()`: 模型选择对话框

### ChatAdapter.kt

**数据结构**:
- `ChatMessage`: 消息类型（用户、AI、系统）
- `AIMessage`: 包含消息内容、token 数量、用时

**关键方法**:
- `addAIMessage()`: 添加 AI 消息
- `updateLastAIMessage()`: 流式更新最后一个 AI 消息
- `updateLastAIMessageMeta()`: 更新元数据（token、用时）

### HistoryHelper.kt

**关键方法**:
- `saveConversation()`: 保存对话历史
- `loadConversation()`: 加载对话历史
- `deleteConversation()`: 删除对话历史
- `getConversationList()`: 获取对话历史列表

### ModelAdapter.kt

**数据结构**:
- `ModelFile`: 模型文件信息（名称、路径、大小等）

**关键方法**:
- `setModels()`: 设置模型列表
- `getModelCount()`: 获取模型数量
- `selectModel()`: 选择模型

### NetworkHelper.kt

**搜索配置**:
- 使用 Bing 中文站点 (cn.bing.com)
- RSS 格式获取结果
- 返回最多 5 条搜索结果

**方法**:
- `search(query, callback)`: 执行搜索
- `fetchUrl(urlStr, callback)`: 获取网页内容
- `isNetworkAvailable()`: 检查网络连接

### ToolManager.kt

**功能**:
- 管理可用工具列表
- 执行工具调用
- 处理工具返回结果

### ToolChatManager.kt

**功能**:
- 管理工具与聊天的集成
- 处理工具调用的聊天上下文

### llama_jni.cpp

**全局变量**:
- `g_model`: 模型对象
- `g_ctx`: 上下文对象
- `g_mtmd_ctx`: 多模态上下文
- `g_sampler`: 采样器
- `g_conversationTokens`: 对话历史 tokens

**导出方法**:
- `nativeLoadModel`: 加载模型
- `nativeAnalyzeImage`: 分析图片
- `nativeChat`: 文本对话
- `nativeResetChatHistory`: 重置历史
- `nativeFreeModel`: 释放模型

## 布局文件

### activity_main.xml
- 主界面布局
- 侧边栏包含：模型选择、系统提示词、生成参数、联网设置、加载模型、历史记录、清空历史

### item_chat_message_*.xml
- `item_chat_message_user.xml`: 用户消息气泡（右对齐，蓝色）
- `item_chat_message_ai.xml`: AI 消息气泡（左对齐，灰色），包含元数据和复制按钮
- `item_chat_message_system.xml`: 系统消息气泡（居中，浅橙色）

### item_history_*.xml
- 历史记录列表项布局

### item_model_*.xml
- 模型列表项布局

## 构建配置

### AndroidManifest.xml
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
android:usesCleartextTraffic="true"
```

### build.gradle.kts
- 编译 SDK: 34
- 最低 SDK: 24
- NDK: 26.1.10909125
- CMake: 3.22.1
- 支持 ABI: arm64-v8a
- 包名: com.erlab.actuaware

## 编译命令

```bash
# 编译 Debug 版本
./gradlew assembleDebug

# 编译 Release 版本
./gradlew assembleRelease

# 清理构建
./gradlew clean
```

## 常见修改任务

### 修改搜索引擎

位置: `NetworkHelper.kt`

```kotlin
// 修改搜索 URL
val url = URL("https://cn.bing.com/search?q=$encodedQuery&format=rss&count=5")

// 修改解析逻辑
private fun parseBingResponse(response: String): String {
    // 解析新的搜索结果格式
}
```

### 修改聊天气泡样式

位置: `res/drawable/bg_chat_bubble_*.xml`

```xml
<!-- 用户气泡 -->
<solid android:color="#2196F3" />
<corners android:topLeftRadius="16dp" ... />

<!-- AI 气泡 -->
<solid android:color="#E8E8E8" />
```

### 添加新的 JNI 方法

1. 在 `MainActivity.kt` 声明:
```kotlin
private external fun nativeNewMethod(param: String): String
```

2. 在 `llama_jni.cpp` 实现:
```cpp
extern "C" JNIEXPORT jstring JNICALL
Java_com_erlab_actuaware_MainActivity_nativeNewMethod(
    JNIEnv *env, jobject /* this */, jstring param) {
    // 实现逻辑
}
```

### 修改系统提示词

位置: `MainActivity.kt` 变量 `systemPrompt`

```kotlin
private var systemPrompt: String = "你是一个有用的助手。"
```

也可以通过侧边栏的"设置系统提示词"按钮动态修改。

### 修改模型性能参数

位置: `llama_jni.cpp` 函数 `nativeLoadModel`

```cpp
// 线程数
ctx_params.n_threads = 8;

// 上下文大小
ctx_params.n_ctx = 8192;

// 批处理大小
ctx_params.n_batch = 512;

// Flash Attention
ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

// KV cache 量化
ctx_params.type_k = GGML_TYPE_F16;
```

### 添加新工具

1. 在 `ToolManager.kt` 定义新工具类型
2. 在 `ToolChatManager.kt` 实现工具调用逻辑
3. 在 `MainActivity.kt` 集成工具调用

## 注意事项

1. **模型加载**: 必须先加载模型才能进行对话或图片分析
2. **多模态支持**: 图片分析需要加载 mmproj 文件或使用自带多模态的模型
3. **联网功能**: 需要网络连接，搜索可能需要几秒钟
4. **上下文限制**: 超出上下文大小会自动截断历史
5. **内存管理**: 长时间使用建议定期清空对话历史
6. **隐私安全**: 所有处理在本地进行，数据不上传（除了网络搜索查询）
7. **存储权限**: Android 13+ 需要相册权限访问图片
8. **模型文件**: 模型文件较大，建议使用外部存储

## 开发建议

1. **测试模型**: 使用小型模型（如 Qwen2.5-0.5B）进行快速测试
2. **性能监控**: 使用 Logcat 查看性能日志，TAG 为 "LlamaJNI"
3. **错误处理**: 所有网络和 JNI 操作都有异常处理
4. **UI 响应**: 长时间操作使用后台线程 + ProgressDialog
5. **代码风格**: 遵循 Kotlin 官方代码风格
6. **版本控制**: 使用 Git 管理代码，定期提交

## 相关资源

- llama.cpp: https://github.com/ggerganov/llama.cpp
- llama.cpp AGENTS.md: 包含 AI 辅助开发指南
- 模型下载: Hugging Face (搜索 GGUF 格式)
- Qwen 模型: https://huggingface.co/Qwen

## 联网功能说明

当前使用 Bing 搜索 API（无需 API Key），如果需要更好的搜索质量：

1. 申请 Bing Web Search API: https://www.microsoft.com/cognitive-services/en-us/bing-web-search-api
2. 获取 API Key
3. 修改 `NetworkHelper.kt` 使用官方 API

## 版本信息

- 项目名称: 动悟 (ActuAware)
- 项目版本: 1.0
- 包名: com.erlab.actuaware
- Kotlin: 1.8
- Gradle: 8.4
- llama.cpp: 集成在 `app/src/main/cpp/llama.cpp/`

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
- 支持工具调用框架