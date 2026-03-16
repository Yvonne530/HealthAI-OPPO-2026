#include <jni.h>
#include <string>
#include <sstream>
#include <android/log.h>
#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>

#include "stb/stb_image.h"

#define TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define MAX_CONVERSATION_TOKENS 4096

static bool isModelLoaded = false;
static struct llama_model *g_model = nullptr;
static struct llama_context *g_ctx = nullptr;
static struct mtmd_context *g_mtmd_ctx = nullptr;
static struct llama_sampler *g_sampler = nullptr;
static std::string g_systemPrompt = "你是一个有用的助手。";
static std::vector<llama_token> g_conversationTokens;
static bool g_supportsMultimodalInModel = false;
static int g_max_tokens = 512;
static int g_context_size = 8192;
static std::string g_mmprojPath = "";
static float g_temperature = 0.7f;
static float g_top_p = 0.9f;
static int g_top_k = 40;

static JavaVM* g_jvm = nullptr;
static jobject g_callback_object = nullptr;
static jmethodID g_callback_method = nullptr;

static std::string cleanUTF8String(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    for (size_t i = 0; i < input.size(); ) {
        unsigned char c = input[i];
        if (c <= 0x7F) {
            result += c;
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 < input.size() && (input[i+1] & 0xC0) == 0x80) {
                result += input[i]; result += input[i+1]; i += 2;
            } else { result += '?'; i++; }
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 < input.size() && (input[i+1] & 0xC0) == 0x80 && (input[i+2] & 0xC0) == 0x80) {
                result += input[i]; result += input[i+1]; result += input[i+2]; i += 3;
            } else { result += '?'; i++; }
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 < input.size() && (input[i+1] & 0xC0) == 0x80 && (input[i+2] & 0xC0) == 0x80 && (input[i+3] & 0xC0) == 0x80) {
                result += input[i]; result += input[i+1]; result += input[i+2]; result += input[i+3]; i += 4;
            } else { result += '?'; i++; }
        } else { result += '?'; i++; }
    }
    return result;
}

// 检测是否包含反提示（antiprompt），如 "USER:" 等
// 如果模型输出了这些内容，说明它已经结束了回答并开始模拟用户输入
static bool containsAntiprompt(const std::string& text) {
    // 常见的反提示模式
    const std::vector<std::string> antiprompts = {
        "USER:",
        "User:",
        "user:",
        "<|user|>",
        "<|USER|>",
        "[INST]",
        "<<USER>>",
        "\n\n用户:",
        "\n\nUser:",
        "\n\nUSER:"
    };
    
    for (const auto& ap : antiprompts) {
        if (text.find(ap) != std::string::npos) {
            LOGI("检测到反提示: '%s'，停止生成", ap.c_str());
            return true;
        }
    }
    return false;
}

static void streamingCallback(const std::string& text) {
    if (g_callback_object && g_callback_method) {
        JNIEnv* ce = nullptr;
        if (g_jvm->GetEnv((void**)&ce, JNI_VERSION_1_6) == JNI_EDETACHED)
            g_jvm->AttachCurrentThread(&ce, nullptr);
        jstring jt = ce->NewStringUTF(text.c_str());
        ce->CallVoidMethod(g_callback_object, g_callback_method, jt);
        ce->DeleteLocalRef(jt);
    }
}

// 性能参数
static int g_n_threads = 12;
static int g_n_batch = 1024;
static int g_n_gpu_layers = 999;
static bool g_enable_flash_attention = true;

extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeSetPerformanceParams(
        JNIEnv *env, jobject,
        jint threads, jint batchSize, jint gpuLayers, jboolean enableFlashAttention) {

    g_n_threads = threads > 0 ? threads : 12;
    g_n_batch = batchSize > 0 ? batchSize : 1024;
    g_n_gpu_layers = gpuLayers >= 0 ? gpuLayers : 999;
    g_enable_flash_attention = enableFlashAttention;

    LOGI("性能参数已设置: 线程数=%d, 批处理大小=%d, GPU层数=%d, Flash Attention=%s",
         g_n_threads, g_n_batch, g_n_gpu_layers,
         g_enable_flash_attention ? "启用" : "禁用");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_erlab_actuaware_MainActivity_nativeLoadModel(
        JNIEnv *env, jobject,
        jstring modelPath, jstring mmprojPath, jstring systemPrompt,
        jint maxTokens, jint contextSize, jfloat temperature, jfloat topP, jint topK) {

    const char *modelPathCStr = env->GetStringUTFChars(modelPath, nullptr);
    const char *mmprojPathCStr = mmprojPath ? env->GetStringUTFChars(mmprojPath, nullptr) : "";
    const char *systemPromptCStr = systemPrompt ? env->GetStringUTFChars(systemPrompt, nullptr) : "你是一个有用的助手。";

    LOGI("加载模型: %s", modelPathCStr);

    g_systemPrompt = systemPromptCStr;
    g_max_tokens = maxTokens > 0 ? maxTokens : 512;
    g_context_size = contextSize >= 1024 ? contextSize : 8192;
    g_temperature = temperature > 0 ? temperature : 0.7f;
    g_top_p = topP > 0 ? topP : 0.9f;
    g_top_k = topK > 0 ? topK : 40;

    if (g_mtmd_ctx) { mtmd_free(g_mtmd_ctx); g_mtmd_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    g_mmprojPath = "";

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = g_n_gpu_layers;
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    g_model = llama_model_load_from_file(modelPathCStr, model_params);
    if (!g_model) {
        env->ReleaseStringUTFChars(modelPath, modelPathCStr);
        isModelLoaded = false;
        return env->NewStringUTF("模型加载失败");
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = g_context_size;
    ctx_params.n_threads = g_n_threads;
    ctx_params.n_batch = g_n_batch;
    ctx_params.n_ubatch = g_n_batch / 4;
    ctx_params.n_threads_batch = g_n_threads;
    ctx_params.flash_attn_type = g_enable_flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.type_k = GGML_TYPE_Q8_0;
    ctx_params.type_v = GGML_TYPE_Q8_0;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        env->ReleaseStringUTFChars(modelPath, modelPathCStr);
        isModelLoaded = false;
        return env->NewStringUTF("上下文初始化失败");
    }

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(64, 1.1, 0.0, 0.0)); // 重复惩罚: 惩罚最近64个token，系数1.1
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(g_top_k));           // Top-K
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(g_top_p, 1));        // Top-P
    llama_sampler_chain_add(g_sampler, llama_sampler_init_min_p(0.05, 1));          // min_p: 过滤概率太低的token
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(g_temperature));     // Temperature
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(42));                // 随机种子

    if (!mmprojPathCStr || strlen(mmprojPathCStr) == 0) {
        g_supportsMultimodalInModel = true;
    }

    if (mmprojPathCStr && strlen(mmprojPathCStr) > 0) {
        g_mmprojPath = mmprojPathCStr;
        struct mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = g_n_gpu_layers > 0;
        mtmd_params.n_threads = g_n_threads;
        mtmd_params.print_timings = false;
        mtmd_params.warmup = true;
        mtmd_params.flash_attn_type = g_enable_flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        g_mtmd_ctx = mtmd_init_from_file(mmprojPathCStr, g_model, mtmd_params);
        if (g_mtmd_ctx) g_supportsMultimodalInModel = true;
    }

    isModelLoaded = true;
    env->ReleaseStringUTFChars(modelPath, modelPathCStr);
    if (mmprojPath) env->ReleaseStringUTFChars(mmprojPath, mmprojPathCStr);
    if (systemPrompt) env->ReleaseStringUTFChars(systemPrompt, systemPromptCStr);

    std::string resultMsg = "模型加载成功！\n\n性能优化已启用:\n- 上下文大小: " + std::to_string(g_context_size) + "\n- 线程数: " + std::to_string(g_n_threads) + "\n- 批处理大小: " + std::to_string(g_n_batch) + "\n- GPU层数: " + std::to_string(g_n_gpu_layers) + "\n- Flash Attention: " + (g_enable_flash_attention ? "启用" : "禁用") + "\n";
    if (g_mtmd_ctx || g_supportsMultimodalInModel) {
        resultMsg += "多模态功能已启用，可以分析图片。";
    } else {
        resultMsg += "文本对话功能已启用。";
    }
    return env->NewStringUTF(resultMsg.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeInitStreamingCallback(
        JNIEnv *env, jobject, jobject callback) {
    env->GetJavaVM(&g_jvm);
    if (g_callback_object) env->DeleteGlobalRef(g_callback_object);
    g_callback_object = env->NewGlobalRef(callback);
    jclass callback_class = env->GetObjectClass(callback);
    g_callback_method = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
    LOGI("流式回调已初始化");
}

extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeCleanupStreamingCallback(
        JNIEnv *env, jobject) {
    if (g_callback_object) {
        env->DeleteGlobalRef(g_callback_object);
        g_callback_object = nullptr;
    }
    g_callback_method = nullptr;
    LOGI("流式回调已清理");
}

// 优化版：直接使用 DirectByteBuffer 中的数据
// 支持 RGB（3字节/像素）或 ARGB（4字节/像素）格式，通过 bufferSize 自动判断
extern "C" JNIEXPORT jstring JNICALL
Java_com_erlab_actuaware_MainActivity_nativeAnalyzeImageDirect(
        JNIEnv *env, jobject, jobject imageBuffer, jint width, jint height, jstring userInput) {

    if (!isModelLoaded || !g_ctx || !g_model) {
        return env->NewStringUTF("请先加载模型");
    }

    // 清理 KV cache，避免之前对话的干扰
    llama_memory_t mem = llama_get_memory(g_ctx);
    llama_memory_clear(mem, true);
    llama_memory_seq_rm(mem, -1, -1, -1);  // 移除所有序列
    LOGI("已清理 KV cache 和所有序列");

    if (!g_mtmd_ctx) {
        return env->NewStringUTF("请先加载多模态文件(mmproj)以支持图片分析功能");
    }

    const char *userInputCStr = userInput ? env->GetStringUTFChars(userInput, nullptr) : "";
    std::string userText(userInputCStr ? userInputCStr : "请分析这张图片。");
    if (userInput) env->ReleaseStringUTFChars(userInput, userInputCStr);

    // 直接获取 DirectByteBuffer 的指针，无需复制
    unsigned char* bufferData = (unsigned char*)env->GetDirectBufferAddress(imageBuffer);
    if (!bufferData) {
        return env->NewStringUTF("无法获取图像数据缓冲区");
    }

    // 验证缓冲区大小，判断格式
    jlong bufferSize = env->GetDirectBufferCapacity(imageBuffer);
    size_t pixelCount = (size_t)width * height;
    size_t expectedRGB = pixelCount * 3;
    size_t expectedARGB = pixelCount * 4;
    
    bool isARGB = (bufferSize >= (jlong)expectedARGB);
    bool isRGB = (bufferSize >= (jlong)expectedRGB && !isARGB);
    
    if (!isRGB && !isARGB) {
        return env->NewStringUTF("图像数据大小不匹配");
    }

    LOGI("使用 DirectByteBuffer 优化路径，图像尺寸: %dx%d, 格式: %s, 数据大小: %ld", 
         width, height, isARGB ? "ARGB" : "RGB", (long)bufferSize);

    // 分配 RGB 数据缓冲区
    unsigned char* rgbData = new unsigned char[expectedRGB];

    if (isARGB) {
        // ARGB → RGB 转换（Android Bitmap 默认格式）
        for (size_t i = 0; i < pixelCount; i++) {
            // Android ARGB 格式: [A, R, G, B] 或 [B, G, R, A] 取决于 Bitmap 配置
            // Bitmap.copyPixelsToBuffer 输出的是 ARGB_8888 格式: [A, R, G, B]
            rgbData[i * 3 + 0] = bufferData[i * 4 + 1];  // R
            rgbData[i * 3 + 1] = bufferData[i * 4 + 2];  // G
            rgbData[i * 3 + 2] = bufferData[i * 4 + 3];  // B
        }
    } else {
        // 已经是 RGB 格式，直接使用
        memcpy(rgbData, bufferData, expectedRGB);
    }

    // 使用 RGB 数据创建 bitmap
    struct mtmd_bitmap *bitmap = mtmd_bitmap_init(width, height, rgbData);
    delete[] rgbData;
    
    if (!bitmap) return env->NewStringUTF("位图创建失败");

    std::string prompt = "USER: <__image__>\n" + userText + "\nASSISTANT:";
    
    struct mtmd_input_text input_text;
    input_text.text = prompt.c_str();
    input_text.add_special = true;
    input_text.parse_special = true;

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    
    const struct mtmd_bitmap* bitmaps[] = { bitmap };
    int ret = mtmd_tokenize(g_mtmd_ctx, chunks, &input_text, bitmaps, 1);
    if (ret != 0) {
        LOGE("mtmd_tokenize 失败: %d", ret);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        std::string err_msg = "图像处理失败，错误码: " + std::to_string(ret);
        return env->NewStringUTF(err_msg.c_str());
    }

    int32_t n_batch = g_n_batch;
    llama_pos n_past = 0;
    llama_seq_id seq_id = 0;
    llama_pos new_n_past = 0;

    LOGI("Tokenize成功，块数量: %zu", mtmd_input_chunks_size(chunks));

    // 记录开始时间
    auto eval_start = std::chrono::high_resolution_clock::now();
    int32_t eval_ret = mtmd_helper_eval_chunks(g_mtmd_ctx, g_ctx, chunks, n_past, seq_id, n_batch, true, &new_n_past);
    auto eval_end = std::chrono::high_resolution_clock::now();
    auto eval_duration = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start).count();

    LOGI("eval_chunks 完成，返回值: %d, new_n_past: %d, 耗时: %lldms", eval_ret, new_n_past, (long long)eval_duration);

    if (eval_ret != 0) {
        LOGE("评估chunks失败，错误码: %d", eval_ret);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return env->NewStringUTF(("处理图像失败，错误码: " + std::to_string(eval_ret)).c_str());
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> generated_tokens;
    int max_tokens = 2048;
    llama_sampler_reset(g_sampler);
    
    bool antiprompt_detected = false;

    for (int i = 0; i < max_tokens; i++) {
        if (new_n_past >= llama_n_ctx(g_ctx)) {
            break;
        }

        llama_token token = llama_sampler_sample(g_sampler, g_ctx, -1);
        llama_sampler_accept(g_sampler, token);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        generated_tokens.push_back(token);

        // 构建当前文本用于反提示检测
        std::string partial_text;
        char buffer[256];
        for (size_t j = 0; j < generated_tokens.size(); j++) {
            int32_t n = llama_token_to_piece(vocab, generated_tokens[j], buffer, sizeof(buffer), 0, true);
            if (n > 0) partial_text.append(buffer, n);
        }

        // 每个 token 都检测反提示
        if (containsAntiprompt(partial_text)) {
            size_t pos = partial_text.find("USER:");
            if (pos == std::string::npos) pos = partial_text.find("User:");
            if (pos == std::string::npos) pos = partial_text.find("user:");
            if (pos != std::string::npos) {
                partial_text = partial_text.substr(0, pos);
            }
            std::string cleaned_text = cleanUTF8String(partial_text);
            streamingCallback(cleaned_text);
            antiprompt_detected = true;
            break;
        }

        // 每3个token回调一次（仅用于UI更新）
        if (generated_tokens.size() % 3 == 0) {
            std::string cleaned_text = cleanUTF8String(partial_text);
            streamingCallback(cleaned_text);
        }

        // 检测到反提示，停止生成
        if (antiprompt_detected) {
            LOGI("检测到反提示，停止生成");
            break;
        }

        llama_batch ob = llama_batch_init(1, 0, 1);
        ob.token[0] = token;
        ob.pos[0] = new_n_past++;
        ob.n_seq_id[0] = 1;
        ob.seq_id[0][0] = seq_id;
        ob.logits[0] = true;
        ob.n_tokens = 1;
        llama_decode(g_ctx, ob);
        llama_batch_free(ob);
    }

    // 最终回调
    if (!generated_tokens.empty() && !antiprompt_detected) {
        std::string partial_text;
        char buffer[256];
        for (size_t j = 0; j < generated_tokens.size(); j++) {
            int32_t n = llama_token_to_piece(vocab, generated_tokens[j], buffer, sizeof(buffer), 0, true);
            if (n > 0) partial_text.append(buffer, n);
        }
        
        if (containsAntiprompt(partial_text)) {
            size_t pos = partial_text.find("USER:");
            if (pos == std::string::npos) pos = partial_text.find("User:");
            if (pos == std::string::npos) pos = partial_text.find("user:");
            if (pos != std::string::npos) {
                partial_text = partial_text.substr(0, pos);
            }
        }
        
        std::string cleaned_text = cleanUTF8String(partial_text);
        streamingCallback(cleaned_text);
    }
    
    std::string generated_text;
    char buffer[256];
    for (size_t i = 0; i < generated_tokens.size(); i++) {
        int32_t n = llama_token_to_piece(vocab, generated_tokens[i], buffer, sizeof(buffer), 0, true);
        if (n > 0) generated_text.append(buffer, n);
    }
    
    if (containsAntiprompt(generated_text)) {
        size_t pos = generated_text.find("USER:");
        if (pos == std::string::npos) pos = generated_text.find("User:");
        if (pos == std::string::npos) pos = generated_text.find("user:");
        if (pos != std::string::npos) {
            generated_text = generated_text.substr(0, pos);
        }
    }
    
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);

    std::string cleaned = cleanUTF8String(generated_text);
    return env->NewStringUTF(("图片分析结果:\n\n" + cleaned).c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_erlab_actuaware_MainActivity_nativeChat(
        JNIEnv *env, jobject, jstring userInput, jboolean resetHistory) {

    if (!isModelLoaded || !g_ctx || !g_model) {
        return env->NewStringUTF("请先加载模型");
    }

    const char *userInputCStr = env->GetStringUTFChars(userInput, nullptr);
    std::string inputText(userInputCStr);
    env->ReleaseStringUTFChars(userInput, userInputCStr);

    if (resetHistory) g_conversationTokens.clear();

    const llama_vocab * vocab = llama_model_get_vocab(g_model);

    std::string userMsg = "USER: " + inputText + "\n";
    std::vector<llama_token> userTokens;
    userTokens.resize(userMsg.size() * 2);
    int32_t n = llama_tokenize(vocab, userMsg.c_str(), userMsg.size(), userTokens.data(), userTokens.size(), true, false);
    if (n > 0) {
        userTokens.resize(n);
        g_conversationTokens.insert(g_conversationTokens.end(), userTokens.begin(), userTokens.end());
    }

    std::string assistantStart = "ASSISTANT:";
    std::vector<llama_token> asTokens;
    asTokens.resize(assistantStart.size() * 2);
    n = llama_tokenize(vocab, assistantStart.c_str(), assistantStart.size(), asTokens.data(), asTokens.size(), true, false);
    if (n > 0) {
        asTokens.resize(n);
        g_conversationTokens.insert(g_conversationTokens.end(), asTokens.begin(), asTokens.end());
    }

    while (g_conversationTokens.size() > (size_t)(g_context_size - 512)) {
        g_conversationTokens.erase(g_conversationTokens.begin());
    }

    llama_batch batch = llama_batch_init(g_conversationTokens.size(), 0, 1);
    for (size_t i = 0; i < g_conversationTokens.size(); i++) {
        batch.token[i] = g_conversationTokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == g_conversationTokens.size() - 1);
    }
    batch.n_tokens = g_conversationTokens.size();
    llama_decode(g_ctx, batch);
    llama_batch_free(batch);

    std::vector<llama_token> generated_tokens;
    llama_sampler_reset(g_sampler);

    for (int i = 0; i < g_max_tokens; i++) {
        if ((int)g_conversationTokens.size() + i >= llama_n_ctx(g_ctx)) break;

        llama_token token = llama_sampler_sample(g_sampler, g_ctx, -1);
        
        // 打印 token 对应的文本，便于调试
        char token_text[64];
        int32_t token_len = llama_token_to_piece(vocab, token, token_text, sizeof(token_text), 0, true);
        if (token_len > 0) {
            token_text[std::min(token_len, 63)] = '\0';
            LOGI("nativeChat 采样得到 token: %d, 文本: '%s'", token, token_text);
        } else {
            LOGI("nativeChat 采样得到 token: %d (无文本表示)", token);
        }
        
        llama_sampler_accept(g_sampler, token);

        if (llama_vocab_is_eog(vocab, token)) {
            LOGI("nativeChat 遇到结束标记 EOS (token: %d)，停止生成", token);
            break;
        }

        generated_tokens.push_back(token);

        // 构建当前文本用于反提示检测
        std::string partial_text;
        char buffer[256];
        for (size_t j = 0; j < generated_tokens.size(); j++) {
            int32_t n = llama_token_to_piece(vocab, generated_tokens[j], buffer, sizeof(buffer), 0, true);
            if (n > 0) partial_text.append(buffer, n);
        }

        // 检测反提示（如 USER:），如果模型开始生成用户消息格式，则停止
        if (containsAntiprompt(partial_text)) {
            LOGI("nativeChat 检测到反提示，停止生成并截断");
            // 截断反提示部分
            size_t pos = partial_text.find("USER:");
            if (pos == std::string::npos) pos = partial_text.find("User:");
            if (pos == std::string::npos) pos = partial_text.find("user:");
            if (pos != std::string::npos) {
                partial_text = partial_text.substr(0, pos);
            }
            // 回调截断后的文本并退出循环
            std::string cleaned_text = cleanUTF8String(partial_text);
            streamingCallback(cleaned_text);
            break;
        }

        // 每3个token回调一次
        if (generated_tokens.size() % 3 == 0) {
            // 清理 UTF-8 字符，避免 JNI 崩溃
            std::string cleaned_text = cleanUTF8String(partial_text);
            streamingCallback(cleaned_text);
        }

        llama_batch ob = llama_batch_init(1, 0, 1);
        ob.token[0] = token;
        ob.pos[0] = g_conversationTokens.size() + i;
        ob.n_seq_id[0] = 1;
        ob.seq_id[0][0] = 0;
        ob.logits[0] = true;
        ob.n_tokens = 1;
        llama_decode(g_ctx, ob);
        llama_batch_free(ob);
    }

    LOGI("nativeChat生成循环完成，生成了 %zu 个 tokens", generated_tokens.size());

    // 最终回调：确保显示完整的生成文本
    if (!generated_tokens.empty()) {
        LOGI("nativeChat执行最终回调，tokens数量: %zu", generated_tokens.size());
        std::string partial_text;
        char buffer[256];
        for (size_t j = 0; j < generated_tokens.size(); j++) {
            int32_t n = llama_token_to_piece(vocab, generated_tokens[j], buffer, sizeof(buffer), 0, true);
            if (n > 0) partial_text.append(buffer, n);
        }
        // 检测并截断反提示
        if (containsAntiprompt(partial_text)) {
            size_t pos = partial_text.find("USER:");
            if (pos == std::string::npos) pos = partial_text.find("User:");
            if (pos == std::string::npos) pos = partial_text.find("user:");
            if (pos != std::string::npos) {
                partial_text = partial_text.substr(0, pos);
                LOGI("nativeChat最终回调截断反提示，保留 %zu 字符", pos);
            }
        }
        // 清理 UTF-8 字符，避免 JNI 崩溃
        std::string cleaned_text = cleanUTF8String(partial_text);
        LOGI("nativeChat最终回调文本长度: %zu", cleaned_text.length());
        streamingCallback(cleaned_text);
    } else {
        LOGI("nativeChat没有生成任何token，跳过最终回调");
    }

    std::string generated_text;
    char buffer[256];
    for (size_t i = 0; i < generated_tokens.size(); i++) {
        int32_t n = llama_token_to_piece(vocab, generated_tokens[i], buffer, sizeof(buffer), 0, true);
        if (n > 0) generated_text.append(buffer, n);
    }
    // 最终文本也截断反提示
    if (containsAntiprompt(generated_text)) {
        size_t pos = generated_text.find("USER:");
        if (pos == std::string::npos) pos = generated_text.find("User:");
        if (pos == std::string::npos) pos = generated_text.find("user:");
        if (pos != std::string::npos) {
            generated_text = generated_text.substr(0, pos);
            LOGI("nativeChat返回文本截断反提示，保留 %zu 字符", pos);
        }
    }

    g_conversationTokens.insert(g_conversationTokens.end(), generated_tokens.begin(), generated_tokens.end());

    // 在回答末尾添加换行符，确保下一轮对话格式正确
    // 对话格式: USER: xxx\nASSISTANT: yyy\n
    std::string newlineStr = "\n";
    std::vector<llama_token> newlineTokens;
    newlineTokens.resize(2);
    int32_t newlineN = llama_tokenize(vocab, newlineStr.c_str(), newlineStr.size(), newlineTokens.data(), newlineTokens.size(), false, false);
    if (newlineN > 0) {
        newlineTokens.resize(newlineN);
        g_conversationTokens.insert(g_conversationTokens.end(), newlineTokens.begin(), newlineTokens.end());
        LOGI("nativeChat 已在回答末尾添加换行符");
    }

    std::string cleaned = cleanUTF8String(generated_text);
    return env->NewStringUTF(cleaned.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeResetChatHistory(JNIEnv *, jobject) {
    g_conversationTokens.clear();
    LOGI("对话历史已重置");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_erlab_actuaware_MainActivity_nativeRestoreContext(
        JNIEnv *env, jobject, jstring historyJson) {

    if (!isModelLoaded || !g_ctx || !g_model) {
        return env->NewStringUTF("模型未加载");
    }

    if (!historyJson) {
        return env->NewStringUTF("历史记录为空");
    }

    const char *historyCStr = env->GetStringUTFChars(historyJson, nullptr);
    std::string historyText(historyCStr);
    env->ReleaseStringUTFChars(historyJson, historyCStr);

    // 清空当前对话历史
    g_conversationTokens.clear();

    // 解析历史记录，每行一条
    std::istringstream iss(historyText);
    std::string line;
    const llama_vocab * vocab = llama_model_get_vocab(g_model);

    while (std::getline(iss, line)) {
        if (line.empty()) continue;

        // 将历史记录转换为 tokens
        std::vector<llama_token> lineTokens;
        lineTokens.resize(line.size() * 2);
        int32_t n = llama_tokenize(vocab, line.c_str(), line.size(), lineTokens.data(), lineTokens.size(), false, false);
        if (n > 0) {
            lineTokens.resize(n);
            g_conversationTokens.insert(g_conversationTokens.end(), lineTokens.begin(), lineTokens.end());
        }
    }

    // 检查是否超过上下文大小
    while (g_conversationTokens.size() > (size_t)(g_context_size - 512)) {
        g_conversationTokens.erase(g_conversationTokens.begin());
    }

    LOGI("对话历史已恢复，共 %zu 个 tokens", g_conversationTokens.size());
    return env->NewStringUTF("对话历史恢复成功");
}

extern "C" JNIEXPORT void JNICALL
Java_com_erlab_actuaware_MainActivity_nativeFreeModel(JNIEnv *, jobject) {
    LOGI("释放模型资源");
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_mtmd_ctx) { mtmd_free(g_mtmd_ctx); g_mtmd_ctx = nullptr; }
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_conversationTokens.clear();
    g_systemPrompt = "你是一个有用的助手。";
    g_supportsMultimodalInModel = false;
    isModelLoaded = false;
    LOGI("模型已释放");
}
