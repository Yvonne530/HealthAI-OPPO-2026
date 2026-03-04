#include <jni.h>
#include <string>
#include <android/log.h>
#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>

#include "stb/stb_image.h"

#define TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// mtmd 日志回调函数
static void mtmd_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    switch (level) {
        case GGML_LOG_LEVEL_INFO:
            __android_log_print(ANDROID_LOG_INFO, "MTMD", "%s", text);
            break;
        case GGML_LOG_LEVEL_WARN:
            __android_log_print(ANDROID_LOG_WARN, "MTMD", "%s", text);
            break;
        case GGML_LOG_LEVEL_ERROR:
            __android_log_print(ANDROID_LOG_ERROR, "MTMD", "%s", text);
            break;
        default:
            __android_log_print(ANDROID_LOG_DEBUG, "MTMD", "%s", text);
            break;
    }
}

static bool isModelLoaded = false;
static struct llama_model *g_model = nullptr;
static struct llama_context *g_ctx = nullptr;
static struct mtmd_context *g_mtmd_ctx = nullptr;
static struct llama_sampler *g_sampler = nullptr;

// 对话历史和系统提示词
static std::string g_systemPrompt = "你是一个有用的助手。";
static std::vector<llama_token> g_conversationTokens;
static bool g_supportsMultimodalInModel = false; // 模型是否自带多模态支持
static int g_max_tokens = 512; // 最大生成token数
static int g_context_size = 8192; // 上下文大小
static std::string g_mmprojPath = ""; // 保存多模态投影器文件路径
static float g_temperature = 0.7f; // 温度
static float g_top_p = 0.9f; // Top-P
static int g_top_k = 40; // Top-K

// 流式输出回调
static JavaVM* g_jvm = nullptr;
static jobject g_callback_object = nullptr;
static jmethodID g_callback_method = nullptr;

extern "C" JNIEXPORT jstring JNICALL
Java_com_erlab_actuaware_MainActivity_nativeLoadModel(
        JNIEnv *env,
        jobject /* this */,
        jstring modelPath,
        jstring mmprojPath,
        jstring systemPrompt,
        jint maxTokens,
        jint contextSize,
        jfloat temperature,
        jfloat topP,
        jint topK) {


    const char *modelPathCStr = env->GetStringUTFChars(modelPath, nullptr);
    const char *mmprojPathCStr = mmprojPath != nullptr ? env->GetStringUTFChars(mmprojPath, nullptr) : "";
    const char *systemPromptCStr = systemPrompt != nullptr ? env->GetStringUTFChars(systemPrompt, nullptr) : "你是一个有用的助手。";

    LOGI("尝试加载模型: %s", modelPathCStr);
    LOGI("多模态文件: %s", mmprojPathCStr);
    LOGI("系统提示词: %s", systemPromptCStr);
    LOGI("最大生成token: %d", maxTokens);
    LOGI("上下文大小: %d", contextSize);

    // 保存参数
    g_systemPrompt = systemPromptCStr;
    g_max_tokens = maxTokens > 0 ? maxTokens : 512;
    g_context_size = contextSize >= 1024 ? contextSize : 8192;
    g_temperature = temperature > 0 ? temperature : 0.7f;
    g_top_p = topP > 0 ? topP : 0.9f;
    g_top_k = topK > 0 ? topK : 40;

    LOGI("实际使用的参数 - max_tokens: %d, context_size: %d, temp: %.2f, topP: %.2f, topK: %d", 
        g_max_tokens, g_context_size, g_temperature, g_top_p, g_top_k);

    LOGI("实际使用的参数 - max_tokens: %d, context_size: %d", g_max_tokens, g_context_size);

    // 释放之前的模型
    if (g_mtmd_ctx != nullptr) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    g_mmprojPath = ""; // 清空 mmproj 路径

    // 设置模型参数 - 性能优化
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999; // 尝试使用GPU层（如果有Vulkan支持）
    model_params.use_mmap = true;  // 使用mmap加速加载
    model_params.use_mlock = false; // 不锁定内存

    // 加载模型
    g_model = llama_model_load_from_file(modelPathCStr, model_params);

    if (g_model == nullptr) {
        LOGE("模型加载失败: %s", modelPathCStr);
        env->ReleaseStringUTFChars(modelPath, modelPathCStr);
        env->ReleaseStringUTFChars(mmprojPath, mmprojPathCStr);
        isModelLoaded = false;
        return env->NewStringUTF("模型加载失败: 无法加载GGUF文件");
    }

    LOGI("模型加载成功");

    // 初始化上下文 - 性能优化
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = g_context_size;  // 使用用户设置的上下文大小
    ctx_params.n_threads = 8; // 增加线程数到8
    ctx_params.n_batch = 512; // 增加批处理大小
    ctx_params.n_ubatch = 512; // 物理批处理大小
    ctx_params.n_threads_batch = 8; // 批处理线程数
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // 启用Flash Attention
    ctx_params.type_k = GGML_TYPE_F16;  // 使用FP16优化KV cache内存

    LOGI("初始化上下文，n_ctx=%d", ctx_params.n_ctx);

    g_ctx = llama_init_from_model(g_model, ctx_params);

    if (g_ctx == nullptr) {
        LOGE("上下文初始化失败");
        llama_model_free(g_model);
        g_model = nullptr;
        env->ReleaseStringUTFChars(modelPath, modelPathCStr);
        env->ReleaseStringUTFChars(mmprojPath, mmprojPathCStr);
        isModelLoaded = false;
        return env->NewStringUTF("模型加载失败: 无法初始化上下文");
    }

    LOGI("上下文初始化成功");

    // 创建采样器 - 性能优化的采样策略
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    
    // 添加采样器链
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(g_top_k));  // Top-K采样
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(g_top_p, 1));  // Top-P采样
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(g_temperature));  // 温度采样
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(42));  // 随机种子
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));  // Top-K采样
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.95, 1));  // Top-P采样
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7));  // 温度采样
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(42));  // 随机种子

    LOGI("采样器创建成功");

    // 检查模型是否支持多模态
    // 某些新模型（如Qwen2-VL等）已将多模态整合到主模型中
    // 这里标记为可能支持，实际处理时再判断
    if (mmprojPathCStr == nullptr || strlen(mmprojPathCStr) == 0) {
        LOGI("未提供mmproj文件，检查模型是否自带多模态支持");
        g_supportsMultimodalInModel = true;
    }

    // 加载多模态投影器（mmproj）- 如果提供了mmproj文件
    if (mmprojPathCStr != nullptr && strlen(mmprojPathCStr) > 0) {
        // 保存 mmproj 路径
        g_mmprojPath = mmprojPathCStr;
        
        struct mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = true;  // 尝试使用GPU
        mtmd_params.n_threads = 8;
        mtmd_params.print_timings = false;
        mtmd_params.warmup = true; // 启用预热
        mtmd_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        g_mtmd_ctx = mtmd_init_from_file(mmprojPathCStr, g_model, mtmd_params);

        if (g_mtmd_ctx == nullptr) {
            LOGE("多模态投影器加载失败: %s", mmprojPathCStr);
            // 如果mmproj加载失败，但模型可能自带多模态，继续尝试
            if (!g_supportsMultimodalInModel) {
                llama_free(g_ctx);
                g_ctx = nullptr;
                llama_model_free(g_model);
                g_model = nullptr;
                if (mmprojPath != nullptr) env->ReleaseStringUTFChars(mmprojPath, mmprojPathCStr);
                env->ReleaseStringUTFChars(modelPath, modelPathCStr);
                if (systemPrompt != nullptr) env->ReleaseStringUTFChars(systemPrompt, systemPromptCStr);
                isModelLoaded = false;
                return env->NewStringUTF("模型加载失败: 无法加载多模态投影器文件");
            } else {
                LOGI("mmproj加载失败，但模型可能自带多模态支持，继续使用");
            }
        } else {
            LOGI("多模态投影器加载成功");
            g_supportsMultimodalInModel = true;
        }
    } else if (g_supportsMultimodalInModel) {
        LOGI("未提供mmproj文件，假设模型自带多模态支持");
    }

    // 启用 mtmd 日志输出
    mtmd_helper_log_set(mtmd_log_callback, nullptr);

    isModelLoaded = true;

    env->ReleaseStringUTFChars(modelPath, modelPathCStr);
    if (mmprojPath != nullptr && mmprojPathCStr != nullptr) {
        env->ReleaseStringUTFChars(mmprojPath, mmprojPathCStr);
    }
    if (systemPrompt != nullptr) {
        env->ReleaseStringUTFChars(systemPrompt, systemPromptCStr);
    }

    std::string resultMsg = "模型加载成功！\n\n性能优化已启用:\n- 上下文大小: 8192\n- 线程数: 8\n- 批处理大小: 512\n- Flash Attention: 启用\n- GPU加速: 尝试启用\n\n";
    if (g_mtmd_ctx != nullptr || g_supportsMultimodalInModel) {
        resultMsg += "多模态功能已启用，可以分析图片。";
    } else {
        resultMsg += "文本对话功能已启用，支持多轮对话。";
    }
    resultMsg += "\n\n系统提示词: " + g_systemPrompt;

    return env->NewStringUTF(resultMsg.c_str());
}

// 初始化流式输出回调
extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeInitStreamingCallback(
        JNIEnv *env,
        jobject /* this */,
        jobject callback) {

    // 保存 JavaVM
    env->GetJavaVM(&g_jvm);

    // 清除旧的回调对象
    if (g_callback_object != nullptr) {
        env->DeleteGlobalRef(g_callback_object);
    }

    // 创建新的全局引用
    g_callback_object = env->NewGlobalRef(callback);

    // 获取回调方法
    jclass callback_class = env->GetObjectClass(callback);
    g_callback_method = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");

    LOGI("流式输出回调已初始化");
}

// 清理流式输出回调
extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeCleanupStreamingCallback(
        JNIEnv *env,
        jobject /* this */) {

    if (g_callback_object != nullptr) {
        env->DeleteGlobalRef(g_callback_object);
        g_callback_object = nullptr;
    }
    g_callback_method = nullptr;
    LOGI("流式输出回调已清理");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_erlab_actuaware_MainActivity_nativeAnalyzeImage(
        JNIEnv *env,
        jobject /* this */,
        jbyteArray imageData,
        jint width,
        jint height,
        jstring userInput) {

    if (!isModelLoaded || g_ctx == nullptr || g_model == nullptr || g_mtmd_ctx == nullptr) {
        return env->NewStringUTF("请先加载模型和多模态投影器");
    }

    LOGI("开始分析图片: %dx%d", width, height);

    // 获取用户文本输入
    const char *userInputCStr = userInput != nullptr ? env->GetStringUTFChars(userInput, nullptr) : "";
    std::string userText(userInputCStr);
    if (userText.empty()) {
        userText = "请分析这张图片。";  // 默认提示
    }
    LOGI("用户输入: %s", userText.c_str());

    // 获取图像数据
    jbyte* imageBytes = env->GetByteArrayElements(imageData, nullptr);
    jsize imageLength = env->GetArrayLength(imageData);

    if (imageBytes == nullptr || imageLength == 0) {
        LOGE("图像数据为空");
        env->ReleaseByteArrayElements(imageData, imageBytes, 0);
        if (userInput != nullptr) env->ReleaseStringUTFChars(userInput, userInputCStr);
        return env->NewStringUTF("图像数据为空");
    }

    // 使用stb_image从内存中加载图像
    int nx, ny, nc;
    unsigned char* stbi_data = stbi_load_from_memory(
        reinterpret_cast<const unsigned char*>(imageBytes),
        imageLength,
        &nx,
        &ny,
        &nc,
        3  // 强制转换为RGB
    );

    env->ReleaseByteArrayElements(imageData, imageBytes, 0);

    if (stbi_data == nullptr) {
        LOGE("无法解码图像数据");
        if (userInput != nullptr) env->ReleaseStringUTFChars(userInput, userInputCStr);
        return env->NewStringUTF("无法解码图像数据，请确保图像格式正确");
    }

    LOGI("图像解码成功: %dx%d, 通道数: %d", nx, ny, nc);

    // 创建mtmd_bitmap
    struct mtmd_bitmap *bitmap = mtmd_bitmap_init(nx, ny, stbi_data);
    stbi_image_free(stbi_data);

    if (bitmap == nullptr) {
        LOGE("无法创建mtmd_bitmap，图像尺寸: %dx%d", nx, ny);
        std::string errorMsg = "无法创建图像位图，图像尺寸: " + std::to_string(nx) + "x" + std::to_string(ny);
        if (userInput != nullptr) env->ReleaseStringUTFChars(userInput, userInputCStr);
        return env->NewStringUTF(errorMsg.c_str());
    }

    LOGI("mtmd_bitmap创建成功");

    // 准备输入文本，必须包含图像标记
    // 使用用户的实际输入
    std::string prompt;
    if (g_conversationTokens.empty()) {
        // 首次对话，添加系统提示词
        prompt = "<|im_start|>system\n" + g_systemPrompt + "<|im_end|>\n";
    }
    prompt += "<|im_start|>user\n<__media__>" + userText + "<|im_end|>\n<|im_start|>assistant\n";

    struct mtmd_input_text text;
    text.text = prompt.c_str();
    text.add_special = false;  // 不自动添加特殊标记，因为已经手动添加了
    text.parse_special = true;

    // 创建输入块
    struct mtmd_input_chunks *chunks = mtmd_input_chunks_init();

    // 分割图像
    const struct mtmd_bitmap *bitmaps[1] = {bitmap};
    int32_t tokenize_res = mtmd_tokenize(
        g_mtmd_ctx,
        chunks,
        &text,
        bitmaps,
        1
    );

    if (tokenize_res != 0) {
        LOGE("Tokenize失败，错误码: %d", tokenize_res);
        std::string errorMsg;
        if (tokenize_res == 1) {
            errorMsg = "错误：提示词中的图像标记数量与提供的图像数量不匹配";
        } else if (tokenize_res == 2) {
            errorMsg = "错误：图像预处理失败，可能是图像格式或尺寸不支持";
        } else {
            errorMsg = "错误：Tokenize失败，错误码: " + std::to_string(tokenize_res);
        }
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return env->NewStringUTF(errorMsg.c_str());
    }

    LOGI("Tokenize成功，块数量: %zu", mtmd_input_chunks_size(chunks));

    int32_t n_batch = 512;
    llama_pos n_past = g_conversationTokens.size();
    llama_seq_id seq_id = 0;
    llama_pos new_n_past = 0;

    LOGI("开始评估chunks，参数: n_past=%d, seq_id=%d, n_batch=%d", n_past, seq_id, n_batch);
    LOGI("上下文信息: 上下文大小=%d", llama_n_ctx(g_ctx));
    LOGI("对话历史tokens数: %zu", g_conversationTokens.size());
    LOGI("g_mtmd_ctx: %p, g_ctx: %p", (void*)g_mtmd_ctx, (void*)g_ctx);
    
    // 检查上下文状态
    LOGI("检查上下文状态...");
    LOGI("  上下文是否已初始化: %s", g_ctx != nullptr ? "是" : "否");
    LOGI("  模型是否已加载: %s", g_model != nullptr ? "是" : "否");
    LOGI("  多模态上下文是否已初始化: %s", g_mtmd_ctx != nullptr ? "是" : "否");
    
    // 尝试获取上下文的当前状态
    if (g_ctx != nullptr) {
        LOGI("  KV cache 状态: 已占用位置 = %zu", g_conversationTokens.size());
    }

    LOGI("即将调用 mtmd_helper_eval_chunks...");

    int32_t ret = mtmd_helper_eval_chunks(g_mtmd_ctx, g_ctx, chunks, n_past, seq_id, n_batch, true, &new_n_past);
    LOGI("mtmd_helper_eval_chunks 返回，返回值: %d, new_n_past: %d", ret, new_n_past);

    if (ret != 0) {
        LOGE("评估chunks失败，错误码: %d", ret);
        LOGE("可能的原因: 上下文状态问题、模型与mmproj不匹配、或参数设置错误");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return env->NewStringUTF(("处理图像失败，错误码: " + std::to_string(ret) + " (可能是上下文状态问题)").c_str());
    }

    LOGI("所有chunks处理完成，new_n_past = %d", new_n_past);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);

    // 开始文本生成
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> generated_tokens;
    int max_tokens = 256;  // 最多生成256个token

    LOGI("开始文本生成，最多生成 %d 个token", max_tokens);
    LOGI("当前new_n_past: %d, 上下文大小: %d", new_n_past, llama_n_ctx(g_ctx));
    LOGI("采样器: %p", (void*)g_sampler);

    // 重置采样器
    llama_sampler_reset(g_sampler);

    for (int i = 0; i < max_tokens; i++) {
        LOGI("=== 生成循环 %d/%d ===", i + 1, max_tokens);
        
        // 检查是否超出上下文大小
        if (new_n_past >= llama_n_ctx(g_ctx)) {
            LOGE("超出上下文大小: %d >= %d", new_n_past, llama_n_ctx(g_ctx));
            break;
        }

        LOGI("准备采样，位置: %d", new_n_past);

        // 采样下一个token
        llama_token token = llama_sampler_sample(g_sampler, g_ctx, -1);
        LOGI("采样结果: token_id = %d", token);

        llama_sampler_accept(g_sampler, token);

        // 检查是否到达结束标记
        if (llama_vocab_is_eog(vocab, token)) {
            LOGI("遇到结束标记，停止生成");
            break;
        }

        generated_tokens.push_back(token);
        LOGI("生成第 %d 个token，token_id: %d，位置: %d", i+1, token, new_n_past);

        // 流式回调：每 2 个 token 回调一次
        if (generated_tokens.size() % 2 == 0 && g_callback_object != nullptr && g_callback_method != nullptr) {
            // 将已生成的 tokens 转换为文本
            std::string partial_text = "";
            char buffer[256];
            for (size_t j = 0; j < generated_tokens.size(); j++) {
                int32_t n_chars = llama_token_to_piece(vocab, generated_tokens[j], buffer, sizeof(buffer), 0, true);
                if (n_chars > 0) {
                    partial_text.append(buffer, n_chars);
                }
            }

            // 调用 Java 回调
            JNIEnv* callback_env = nullptr;
            if (g_jvm->GetEnv((void**)&callback_env, JNI_VERSION_1_6) == JNI_EDETACHED) {
                g_jvm->AttachCurrentThread(&callback_env, nullptr);
            }

            jstring j_partial_text = callback_env->NewStringUTF(partial_text.c_str());
            callback_env->CallVoidMethod(g_callback_object, g_callback_method, j_partial_text);
            callback_env->DeleteLocalRef(j_partial_text);

            LOGI("图像分析流式回调: 已生成 %zu 个token, 文本长度: %zu", generated_tokens.size(), partial_text.length());
        }

        // 添加token到batch进行下一次解码
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = token;
        batch.pos[0] = new_n_past++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = seq_id;
        batch.logits[0] = true;  // 重要：需要为下一个token生成logits
        batch.n_tokens = 1;  // 明确设置token数量

        LOGI("准备解码token: %d, 位置: %d", batch.token[0], batch.pos[0]);
        int decode_ret = llama_decode(g_ctx, batch);
        llama_batch_free(batch);

        if (decode_ret != 0) {
            LOGE("解码失败，错误码: %d", decode_ret);
            LOGE("解码失败时状态 - 位置: %d, 上下文大小: %d, 训练上下文: %d",
                 new_n_past - 1, llama_n_ctx(g_ctx), llama_model_n_ctx_train(g_model));
            break;
        }
        LOGI("解码成功");
    }

    // 强制回调剩余的 token（确保最后几个 token 不会丢失）
    if (!generated_tokens.empty() && g_callback_object != nullptr && g_callback_method != nullptr) {
        std::string partial_text = "";
        char buffer[256];
        for (size_t j = 0; j < generated_tokens.size(); j++) {
            int32_t n_chars = llama_token_to_piece(vocab, generated_tokens[j], buffer, sizeof(buffer), 0, true);
            if (n_chars > 0) {
                partial_text.append(buffer, n_chars);
            }
        }
        
        JNIEnv* callback_env = nullptr;
        if (g_jvm->GetEnv((void**)&callback_env, JNI_VERSION_1_6) == JNI_EDETACHED) {
            g_jvm->AttachCurrentThread(&callback_env, nullptr);
        }
        
        jstring j_partial_text = callback_env->NewStringUTF(partial_text.c_str());
        callback_env->CallVoidMethod(g_callback_object, g_callback_method, j_partial_text);
        callback_env->DeleteLocalRef(j_partial_text);
        
        LOGI("图像分析最终回调: 已生成 %zu 个token, 文本长度: %zu", generated_tokens.size(), partial_text.length());
    }

    LOGI("文本生成完成，共生成 %zu 个token", generated_tokens.size());
    
    LOGI("文本生成完成，共生成 %zu 个token", generated_tokens.size());

    // 将tokens转换为文本
    std::string generated_text = "";
    if (!generated_tokens.empty()) {
        char buffer[256];
        for (size_t i = 0; i < generated_tokens.size(); i++) {
            int32_t n_chars = llama_token_to_piece(vocab, generated_tokens[i], buffer, sizeof(buffer), 0, true);
            if (n_chars > 0) {
                generated_text.append(buffer, n_chars);
            }
        }
        LOGI("生成的文本长度: %zu", generated_text.length());
    }

    // 更新对话历史
    // 将图片对话添加到历史中
    std::vector<llama_token> histTokens;
    histTokens.resize(prompt.length() * 2);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), histTokens.data(), histTokens.size(), true, false);
    if (n_tokens > 0) {
        histTokens.resize(n_tokens);
        g_conversationTokens.insert(g_conversationTokens.end(), histTokens.begin(), histTokens.end());
    }
    g_conversationTokens.insert(g_conversationTokens.end(), generated_tokens.begin(), generated_tokens.end());

    // 添加对话结束标记
    std::vector<llama_token> endTokens;
    endTokens.resize(11);
    n_tokens = llama_tokenize(vocab, "<|im_end|>\n", 11, endTokens.data(), endTokens.size(), true, false);
    if (n_tokens > 0) {
        endTokens.resize(n_tokens);
        g_conversationTokens.insert(g_conversationTokens.end(), endTokens.begin(), endTokens.end());
    }

    return env->NewStringUTF(("图片分析结果:\n\n" + generated_text).c_str());
}

extern "C" JNIEXPORT jstring JNICALL

Java_com_erlab_actuaware_MainActivity_nativeChat(

        JNIEnv *env,

        jobject /* this */,

        jstring userInput,

        jboolean resetHistory) {



    if (!isModelLoaded || g_ctx == nullptr || g_model == nullptr) {

        return env->NewStringUTF("请先加载模型");

    }



    const char *userInputCStr = env->GetStringUTFChars(userInput, nullptr);

    std::string inputText(userInputCStr);



    LOGI("=== 开始文本对话 ===");

    LOGI("输入: %s, 重置历史: %d", userInputCStr, resetHistory);

    LOGI("当前对话历史tokens数量: %zu", g_conversationTokens.size());



    const llama_vocab * vocab = llama_model_get_vocab(g_model);



    // 如果需要重置历史

    if (resetHistory) {

        LOGI("重置对话历史...");

        g_conversationTokens.clear();

        // 重新初始化上下文以清除KV cache

        if (g_ctx != nullptr) {

            llama_free(g_ctx);

            g_ctx = nullptr;

        }

        struct llama_context_params ctx_params = llama_context_default_params();

        ctx_params.n_ctx = g_context_size;

        ctx_params.n_threads = 8;

        ctx_params.n_batch = 512;

        ctx_params.n_ubatch = 512;

        ctx_params.n_threads_batch = 8;

        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        ctx_params.type_k = GGML_TYPE_F16;

        g_ctx = llama_init_from_model(g_model, ctx_params);

        

        if (g_ctx == nullptr) {

            LOGE("重置后重新初始化上下文失败");

            env->ReleaseStringUTFChars(userInput, userInputCStr);

            return env->NewStringUTF("重置对话历史失败: 无法重新初始化上下文");

        }

        LOGI("对话历史和KV cache已重置");

    }



    // 检查上下文使用情况

    size_t current_usage = g_conversationTokens.size();

    size_t context_limit = llama_n_ctx(g_ctx);

    LOGI("上下文使用: %zu / %d (%.1f%%)", current_usage, context_limit, 

         100.0 * current_usage / context_limit);



    // 如果历史为空，添加系统提示词

    if (g_conversationTokens.empty()) {

        LOGI("添加系统提示词...");

        std::string systemPrompt = "<|im_start|>system\n" + g_systemPrompt + "<|im_end|>\n";

        std::vector<llama_token> sysTokens;

        sysTokens.resize(systemPrompt.length() * 2);

        int32_t n_tokens = llama_tokenize(vocab, systemPrompt.c_str(), systemPrompt.length(), sysTokens.data(), sysTokens.size(), true, false);

        if (n_tokens > 0) {

            sysTokens.resize(n_tokens);

            g_conversationTokens.insert(g_conversationTokens.end(), sysTokens.begin(), sysTokens.end());

            LOGI("系统提示词添加完成，tokens数: %d", n_tokens);

        }

    }



    // 添加用户输入

    LOGI("添加用户输入...");

    std::string userPrompt = "<|im_start|>user\n" + inputText + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> userTokens;

    userTokens.resize(userPrompt.length() * 2);

    int32_t n_tokens = llama_tokenize(vocab, userPrompt.c_str(), userPrompt.length(), userTokens.data(), userTokens.size(), true, false);

    if (n_tokens > 0) {

        userTokens.resize(n_tokens);

        size_t before_add = g_conversationTokens.size();

        g_conversationTokens.insert(g_conversationTokens.end(), userTokens.begin(), userTokens.end());

        LOGI("用户输入添加完成，新增tokens: %d，总数: %zu -> %zu", 

             n_tokens, before_add, g_conversationTokens.size());

    }



    // 只处理新添加的tokens（用户输入部分），利用KV cache

    size_t start_pos = g_conversationTokens.size() - userTokens.size();

    LOGI("只处理新tokens: 位置 %zu 到 %zu (共 %zu 个)", 

         start_pos, g_conversationTokens.size() - 1, userTokens.size());



    llama_seq_id seq_id = 0;

    

    // 处理新的用户输入tokens

    for (size_t i = 0; i < userTokens.size(); i++) {

        llama_batch batch = llama_batch_init(1, 0, 1);

        batch.token[0] = userTokens[i];

        batch.pos[0] = start_pos + i;

        batch.n_seq_id[0] = 1;

        batch.seq_id[0][0] = seq_id;

        batch.logits[0] = (i == userTokens.size() - 1); // 只为最后一个token计算logits

        batch.n_tokens = 1;



        LOGI("处理新token[%zu]: id=%d, pos=%zu", i, batch.token[0], batch.pos[0]);

        

        int decode_ret = llama_decode(g_ctx, batch);

        llama_batch_free(batch);



        if (decode_ret != 0) {

            LOGE("解码用户输入失败，错误码: %d, pos=%zu", decode_ret, batch.pos[0]);

            env->ReleaseStringUTFChars(userInput, userInputCStr);

            return env->NewStringUTF(("处理输入失败，错误码: " + std::to_string(decode_ret)).c_str());

        }

    }



    LOGI("用户输入处理完成，开始生成回复");



    // 生成回复

    std::vector<llama_token> generated_tokens;

    llama_token token;

    bool stopGenerated = false;



    // 重置采样器

    llama_sampler_reset(g_sampler);



    for (int i = 0; i < g_max_tokens; i++) {

        size_t current_pos = g_conversationTokens.size() + generated_tokens.size();

        

        // 检查上下文

        if (current_pos >= (size_t)llama_n_ctx(g_ctx)) {

            LOGE("超出上下文大小: current_pos=%zu >= %d", current_pos, llama_n_ctx(g_ctx));

            break;

        }



        // 采样下一个token

        token = llama_sampler_sample(g_sampler, g_ctx, -1);

        llama_sampler_accept(g_sampler, token);



        // 检查是否到达结束标记

        if (llama_vocab_is_eog(vocab, token)) {

            LOGI("遇到结束标记(EOS)");

            break;

        }



        // 检查是否到达对话结束标记 <|im_end|>

        std::vector<llama_token> endCheckTokens;

        endCheckTokens.resize(10);

        int32_t n_end = llama_tokenize(vocab, "<|im_end|>", 10, endCheckTokens.data(), endCheckTokens.size(), true, false);

        if (n_end > 0 && token == endCheckTokens[0]) {

            LOGI("遇到对话结束标记");

            stopGenerated = true;

            break;

        }



        generated_tokens.push_back(token);

        // 每 2 个 token 回调一次
        if (generated_tokens.size() % 2 == 0 && g_callback_object != nullptr && g_callback_method != nullptr) {
            // 将已生成的 tokens 转换为文本
            std::string partial_text = "";
            char buffer[256];
            for (size_t j = 0; j < generated_tokens.size(); j++) {
                int32_t n_chars = llama_token_to_piece(vocab, generated_tokens[j], buffer, sizeof(buffer), 0, true);
                if (n_chars > 0) {
                    partial_text.append(buffer, n_chars);
                }
            }

            // 调用 Java 回调
            JNIEnv* callback_env = nullptr;
            if (g_jvm->GetEnv((void**)&callback_env, JNI_VERSION_1_6) == JNI_EDETACHED) {
                g_jvm->AttachCurrentThread(&callback_env, nullptr);
            }

            jstring j_partial_text = callback_env->NewStringUTF(partial_text.c_str());
            callback_env->CallVoidMethod(g_callback_object, g_callback_method, j_partial_text);
            callback_env->DeleteLocalRef(j_partial_text);

            LOGI("流式回调: 已生成 %zu 个token, 文本长度: %zu", generated_tokens.size(), partial_text.length());
        }

        LOGI("生成token[%d]: id=%d, pos=%zu", i + 1, token, current_pos);



        // 添加token到batch进行解码

        llama_batch batch = llama_batch_init(1, 0, 1);

        batch.token[0] = token;

        batch.pos[0] = current_pos;

        batch.n_seq_id[0] = 1;

        batch.seq_id[0][0] = seq_id;

        batch.logits[0] = true;

        batch.n_tokens = 1;



        int decode_ret = llama_decode(g_ctx, batch);

        llama_batch_free(batch);



        if (decode_ret != 0) {

            LOGE("生成token失败，错误码: %d", decode_ret);

            break;

        }

    }

    // 强制回调剩余的 token（确保最后几个 token 不会丢失）
    if (!generated_tokens.empty() && g_callback_object != nullptr && g_callback_method != nullptr) {
        std::string partial_text = "";
        char buffer[256];
        for (size_t j = 0; j < generated_tokens.size(); j++) {
            int32_t n_chars = llama_token_to_piece(vocab, generated_tokens[j], buffer, sizeof(buffer), 0, true);
            if (n_chars > 0) {
                partial_text.append(buffer, n_chars);
            }
        }
        
        JNIEnv* callback_env = nullptr;
        if (g_jvm->GetEnv((void**)&callback_env, JNI_VERSION_1_6) == JNI_EDETACHED) {
            g_jvm->AttachCurrentThread(&callback_env, nullptr);
        }
        
        jstring j_partial_text = callback_env->NewStringUTF(partial_text.c_str());
        callback_env->CallVoidMethod(g_callback_object, g_callback_method, j_partial_text);
        callback_env->DeleteLocalRef(j_partial_text);
        
        LOGI("最终流式回调: 已生成 %zu 个token, 文本长度: %zu", generated_tokens.size(), partial_text.length());
    }



    LOGI("生成完成，tokens数: %zu, 停止原因: %s", 

         generated_tokens.size(),

         stopGenerated ? "结束标记" : "达到上限");



    // 将生成的tokens添加到对话历史

    g_conversationTokens.insert(g_conversationTokens.end(), generated_tokens.begin(), generated_tokens.end());



    // 添加对话结束标记

    std::vector<llama_token> endTokens;

    endTokens.resize(11);

    n_tokens = llama_tokenize(vocab, "<|im_end|>\n", 11, endTokens.data(), endTokens.size(), true, false);

    if (n_tokens > 0) {

        endTokens.resize(n_tokens);

        g_conversationTokens.insert(g_conversationTokens.end(), endTokens.begin(), endTokens.end());

    }



    LOGI("对话历史更新完成，总数: %zu", g_conversationTokens.size());



    // 将tokens转换为文本

    std::string generated_text = "";

    if (!generated_tokens.empty()) {

        char buffer[256];

        for (size_t i = 0; i < generated_tokens.size(); i++) {

            int32_t n_chars = llama_token_to_piece(vocab, generated_tokens[i], buffer, sizeof(buffer), 0, true);

            if (n_chars > 0) {

                generated_text.append(buffer, n_chars);

            }

        }

    }



    env->ReleaseStringUTFChars(userInput, userInputCStr);

    return env->NewStringUTF(generated_text.c_str());

}

extern "C" JNIEXPORT jstring JNICALL
Java_com_erlab_actuaware_MainActivity_nativeRestoreContext(
        JNIEnv *env,
        jobject /* this */,
        jstring historyJson) {

    if (!isModelLoaded || g_ctx == nullptr || g_model == nullptr) {
        return env->NewStringUTF("请先加载模型");
    }

    if (historyJson == nullptr) {
        return env->NewStringUTF("历史记录为空");
    }

    const char *historyJsonCStr = env->GetStringUTFChars(historyJson, nullptr);
    std::string historyStr(historyJsonCStr);

    LOGI("=== 开始恢复对话上下文 ===");
    LOGI("历史记录长度: %zu 字符", historyStr.length());

    // 清空对话历史tokens
    g_conversationTokens.clear();

    // 重新初始化上下文以清除KV cache
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = g_context_size;
    ctx_params.n_threads = 8;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads_batch = 8;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.type_k = GGML_TYPE_F16;

    LOGI("重新初始化上下文，n_ctx=%d", ctx_params.n_ctx);
    g_ctx = llama_init_from_model(g_model, ctx_params);

    if (g_ctx == nullptr) {
        LOGE("重新初始化上下文失败");
        env->ReleaseStringUTFChars(historyJson, historyJsonCStr);
        return env->NewStringUTF("恢复上下文失败: 无法重新初始化上下文");
    }

    // 解析历史记录（格式：[用户]|||消息 或 [助手]|||消息）
    std::vector<std::pair<std::string, std::string>> messages; // <角色, 消息>
    size_t pos = 0;
    while (pos < historyStr.length()) {
        size_t bracketStart = historyStr.find('[', pos);
        if (bracketStart == std::string::npos) break;

        size_t bracketEnd = historyStr.find(']', bracketStart);
        if (bracketEnd == std::string::npos) break;

        std::string role = historyStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);

        size_t separator = historyStr.find("|||", bracketEnd);
        if (separator == std::string::npos) break;

        std::string message = historyStr.substr(separator + 3);

        // 查找下一条消息的开始
        size_t nextMessage = message.find("[");
        if (nextMessage != std::string::npos) {
            message = message.substr(0, nextMessage);
        }

        messages.push_back({role, message});
        pos = bracketEnd + role.length() + 3 + message.length();
    }

    LOGI("解析到 %zu 条消息", messages.size());

    const llama_vocab * vocab = llama_model_get_vocab(g_model);

    // 添加系统提示词
    if (!g_systemPrompt.empty()) {
        std::vector<llama_token> systemTokens;
        systemTokens.resize(g_systemPrompt.length() * 2);
        int32_t n_sys_tokens = llama_tokenize(vocab, g_systemPrompt.c_str(), g_systemPrompt.length(), systemTokens.data(), systemTokens.size(), true, false);
        if (n_sys_tokens > 0) {
            systemTokens.resize(n_sys_tokens);
        }
        LOGI("系统提示词 tokens: %zu", systemTokens.size());

        // 批量添加系统提示词tokens
        for (size_t i = 0; i < systemTokens.size(); i++) {
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.token[0] = systemTokens[i];
            batch.pos[0] = g_conversationTokens.size();
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = false;
            batch.n_tokens = 1;

            int decode_ret = llama_decode(g_ctx, batch);
            llama_batch_free(batch);

            if (decode_ret != 0) {
                LOGE("解码系统提示词失败，错误码: %d", decode_ret);
                env->ReleaseStringUTFChars(historyJson, historyJsonCStr);
                return env->NewStringUTF("恢复上下文失败: 解码系统提示词失败");
            }

            g_conversationTokens.push_back(systemTokens[i]);
        }
    }

    // 添加历史消息tokens
    for (const auto& msg : messages) {
        std::string role = msg.first;
        std::string content = msg.second;

        if (content.empty()) continue;

        // 构建完整的消息格式
        std::string fullMessage;
        if (role == "用户") {
            fullMessage = "用户: " + content + "\n助手: ";
        } else if (role == "助手") {
            fullMessage = content + "\n";
        } else {
            continue; // 跳过其他角色
        }

        std::vector<llama_token> tokens;
        tokens.resize(fullMessage.length() * 2);
        int32_t n_msg_tokens = llama_tokenize(vocab, fullMessage.c_str(), fullMessage.length(), tokens.data(), tokens.size(), false, false);
        if (n_msg_tokens > 0) {
            tokens.resize(n_msg_tokens);
        }
        LOGI("添加 %s 消息，tokens: %zu", role.c_str(), tokens.size());

        // 批量添加tokens到KV cache
        for (size_t i = 0; i < tokens.size(); i++) {
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.token[0] = tokens[i];
            batch.pos[0] = g_conversationTokens.size();
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = false;
            batch.n_tokens = 1;

            int decode_ret = llama_decode(g_ctx, batch);
            llama_batch_free(batch);

            if (decode_ret != 0) {
                LOGE("解码消息失败，错误码: %d, 消息: %s", decode_ret, fullMessage.c_str());
                env->ReleaseStringUTFChars(historyJson, historyJsonCStr);
                return env->NewStringUTF("恢复上下文失败: 解码消息失败");
            }

            g_conversationTokens.push_back(tokens[i]);

            // 检查是否超过上下文限制
            if (g_conversationTokens.size() >= (size_t)llama_n_ctx(g_ctx)) {
                LOGE("已达到上下文限制，停止添加更多消息");
                break;
            }
        }

        if (g_conversationTokens.size() >= (size_t)llama_n_ctx(g_ctx)) {
            break;
        }
    }

    LOGI("上下文恢复完成，总tokens: %zu", g_conversationTokens.size());
    env->ReleaseStringUTFChars(historyJson, historyJsonCStr);

    return env->NewStringUTF("上下文恢复成功");
}

extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeResetChatHistory(
        JNIEnv *env,
        jobject /* this */) {

    LOGI("重置JNI层对话历史");

    // 清空对话历史tokens
    g_conversationTokens.clear();

    // 重新初始化上下文以清除KV cache
    if (g_ctx != nullptr && g_model != nullptr) {
        llama_free(g_ctx);
        struct llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = g_context_size;
        ctx_params.n_threads = 8;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;
        ctx_params.n_threads_batch = 8;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx_params.type_k = GGML_TYPE_F16;
        
        LOGI("重新初始化上下文，n_ctx=%d", ctx_params.n_ctx);
        g_ctx = llama_init_from_model(g_model, ctx_params);

        if (g_ctx == nullptr) {
            LOGE("重新初始化上下文失败");
        } else {
            LOGI("上下文重新初始化成功");
        }
    }

    LOGI("JNI层对话历史重置完成，参数: max_tokens=%d, context_size=%d", g_max_tokens, g_context_size);
}

extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeFreeModel(
        JNIEnv *env,
        jobject /* this */) {

    LOGI("释放模型资源");

    if (g_sampler != nullptr) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }

    if (g_mtmd_ctx != nullptr) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
    }

    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    // 清空对话历史
    g_conversationTokens.clear();
    g_systemPrompt = "你是一个有用的助手。";
    g_supportsMultimodalInModel = false;

    isModelLoaded = false;
    LOGI("模型已释放");
}