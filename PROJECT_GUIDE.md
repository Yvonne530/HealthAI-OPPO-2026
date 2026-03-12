**项目指南：动悟 (Llama.cpp Android 多模态 AI 助手)**  
**项目概述**  
动悟是一个基于 llama.cpp 的 Android 本地多模态 AI 助手应用，支持文本对话、图片分析、骨骼点检测、联网搜索和对话历史管理功能。  
**核心技术栈**  
- **语言**: Kotlin (Android)  
- **原生代码**: C++ (CMake)  
- **UI**: ViewBinding + Material Design  
- **模型引擎**: llama.cpp (本地大语言模型)  
- **多模态**: mtmd (多模态处理)  
- **骨骼检测**: ML Kit Pose Detection  
- **实时视频**: CameraX  
- **联网**: Bing 搜索 API  
- **Markdown**: Markwon 库  
**项目结构**  
app/  
 ├── src/main/  
 │   ├── java/com/erlab/actuaware/  
 │   │   ├── MainActivity.kt              # 主活动  
 │   │   ├── ActuAwareApplication.kt      # 应用入口类  
 │   │   ├── ChatAdapter.kt               # 聊天气泡适配器  
 │   │   ├── HistoryAdapter.kt            # 历史记录适配器  
 │   │   ├── HistoryHelper.kt             # 历史记录管理工具  
 │   │   ├── ModelLoadProgressDialog.kt   # 模型加载进度对话框  
 │   │   ├── NetworkHelper.kt             # 网络请求工具类  
 │   │   ├── ToolChatManager.kt           # 工具聊天管理器  
 │   │   ├── ToolManager.kt               # 工具管理器  
 │   │   ├── PerformanceManager.kt        # 性能管理器  
 │   │   └── ImageCacheManager.kt         # 图像缓存管理器  
 │   ├── cpp/  
 │   │   ├── llama_jni.cpp               # JNI 接口层  
 │   │   └── llama.cpp/                  # llama.cpp 源码  
 │   ├── res/  
 │   │   ├── layout/                     # 布局文件  
 │   │   ├── drawable/                   # 背景和图标  
 │   │   ├── menu/                       # 菜单配置  
 │   │   └── values/                     # 颜色、字符串等  
 │   └── AndroidManifest.xml  
 └── build.gradle.kts  
   
**关键功能**  
**1. 聊天气泡界面**  
- 位置: MainActivity.kt, ChatAdapter.kt  
- 布局: item_chat_message_user.xml, item_chat_message_ai.xml, item_chat_message_system.xml  
- 特点: 使用 RecyclerView 实现聊天气泡样式，支持三种消息类型（用户、AI、系统）  
- **流式输出**: AI 回复采用流式输出，逐字显示效果  
- **元数据显示**: AI 消息下方显示 token 数量和生成用时  
- **复制功能**: AI 消息提供一键复制按钮  
- **Markdown 渲染**: 支持 Markdown 格式，包括表格、任务列表等  
**2. 图片分析与动作识别**  
- JNI 方法: nativeAnalyzeImage()  
- 流程: 用户发送图片 → 骨骼检测 → JNI 处理 → AI 分析 → 返回结果  
- 功能: 支持姿态检测、骨骼点绘制、AI 分析结果 Markdown 渲染  
- **智能滚动**: 输出时自动滚动，用户向上滚动时暂停  
**3. 实时摄像头捕获**  
- 使用 CameraX 实现实时视频捕获  
- 实时骨骼点检测和显示  
- 支持前后摄像头切换  
**4. 联网搜索**  
- 位置: NetworkHelper.kt  
- 搜索引擎: Bing (cn.bing.com)  
- 方法: search() - 搜索查询，fetchUrl() - 获取网页内容  
- 集成方式: RAG（检索增强生成）- 搜索结果 + 用户问题 → 本地模型  
**5. 对话历史管理**  
- 位置: HistoryHelper.kt, HistoryAdapter.kt  
- 功能: 保存、加载、删除对话历史  
- 存储: 使用 JSON 文件存储在 files/history 目录  
- 保存内容: 对话历史、模型路径、参数设置、系统提示词  
**6. 模型管理**  
- 位置: MainActivity.kt, ModelLoadProgressDialog.kt  
- JNI 方法: nativeLoadModel(), nativeFreeModel()  
- 支持格式: GGUF 格式的 llama.cpp 模型  
- 性能优化: Flash Attention, GPU 加速 (Vulkan), 自适应配置  
**7. 工具管理**  
- 位置: ToolManager.kt, ToolChatManager.kt  
- 功能: 文件创建、读取、修改、删除  
- 安全验证: 路径验证、文件类型验证、大小限制  
**8. 性能管理**  
- 位置: PerformanceManager.kt  
- 功能: 自动检测设备性能等级  
- 自适应配置: 线程数、批处理大小、上下文大小、摄像头帧率  
**JNI 接口**  
**导出方法**  
// 性能参数设置  
 private external fun nativeSetPerformanceParams(threads: Int, batchSize: Int, gpuLayers: Int, enableFlashAttention: Boolean)  
   
 // 模型加载与释放  
 private external fun nativeLoadModel(modelPath: String, mmprojPath: String, systemPrompt: String, maxTokens: Int, contextSize: Int, temperature: Float, topP: Float, topK: Int): String  
 private external fun nativeFreeModel()  
   
 // 对话相关  
 private external fun nativeChat(userInput: String, resetHistory: Boolean): String  
 private external fun nativeResetChatHistory()  
 private external fun nativeRestoreContext(historyJson: String): String  
   
 // 图片分析  
 private external fun nativeAnalyzeImage(imageData: IntArray, width: Int, height: Int, userInput: String): String  
   
 // 流式输出  
 private external fun nativeInitStreamingCallback(callback: StreamingCallback)  
 private external fun nativeCleanupStreamingCallback()  
   
**布局文件**  
**activity_main.xml**  
- 主界面布局  
- 底部导航栏：聊天、识别、历史、设置  
**item_chat_message_*.xml**  
- item_chat_message_user.xml: 用户消息气泡（右对齐）  
- item_chat_message_ai.xml: AI 消息气泡（左对齐），包含元数据和复制按钮  
- item_chat_message_system.xml: 系统消息气泡（居中）  
**item_history.xml**  
- 历史记录列表项布局  
**构建配置**  
**build.gradle.kts**  
- 编译 SDK: 34  
- 最低 SDK: 24  
- NDK: 26.1.10909125  
- CMake: 3.22.1  
- 支持 ABI: arm64-v8a  
- 包名: com.erlab.actuaware  
**主要依赖**  
- AndroidX Core, AppCompat, Material  
- Navigation Component  
- CameraX (相机预览和视频捕获)  
- ML Kit Pose Detection (骨骼点检测)  
- Markwon (Markdown 渲染)  
**编译命令**  
# 编译 Debug 版本  
 ./gradlew assembleDebug  
   
 # 编译 Release 版本  
 ./gradlew assembleRelease  
   
 # 清理构建  
 ./gradlew clean  
   
**性能优化建议**  
1. **设备自适应**: PerformanceManager 自动根据设备性能调整参数  
2. **图像缓存**: ImageCacheManager 管理 LRU 缓存，避免重复解码  
3. **异步操作**: 网络请求、模型加载等耗时操作在后台线程执行  
4. **内存管理**: 定期清理对话历史，避免内存溢出  
**注意事项**  
1. **模型加载**: 必须先加载模型才能进行对话或图片分析  
2. **多模态支持**: 图片分析需要加载 mmproj 文件或使用自带多模态的模型  
3. **联网功能**: 需要网络连接，搜索可能需要几秒钟  
4. **上下文限制**: 超出上下文大小会自动截断历史  
5. **内存管理**: 长时间使用建议定期清空对话历史  
6. **隐私安全**: 所有处理在本地进行，数据不上传（除了网络搜索查询）  
7. **存储权限**: Android 13+ 需要相册权限访问图片  
**版本信息**  
- 项目名称: 动悟 (ActuAware)  
- 包名: com.erlab.actuaware  
- Gradle: 8.4  
- NDK: 26.1.10909125  
**更新日志**  
**最新版本**  
- 支持本地 AI 对话  
- 支持图片分析（含骨骼点检测）  
- 支持实时摄像头捕获  
- 支持联网搜索  
- 支持对话历史管理  
- 支持 Markdown 渲染  
- 支持流式输出  
- 支持智能滚动  
- 支持设备性能自适应  
