/**
 * mnn_jni.cpp
 * ===========
 * JNI bridge for MNN 2.9.0 — RehabGuardian on-device inference.
 *
 * Exposed to Kotlin via: com.healthai.ankle.inference.RGMNNBridge
 *
 * Design notes:
 *   - Net handles  = jlong storing MNN::Interpreter* (caller owns lifecycle)
 *   - Sess handles = jlong storing MNN::Session*     (owned by Interpreter)
 *   - ALL pointers validated before use; null → JNI_FALSE / null return
 *   - Thread safety: MNN Session is not thread-safe. Kotlin AtomicBoolean
 *     single-flight guard in RGPhaseAEngine ensures serial access.
 *   - No JNI exceptions thrown; errors surfaced as false / 0 / null returns
 *     so Kotlin can log and fallback gracefully.
 *
 * MNN 2.9.0 C++ API used:
 *   MNN::Interpreter::createFromFile(path)     → Interpreter*
 *   interpreter->createSession(schedConfig)    → Session*
 *   interpreter->getSessionInput(sess, name)   → Tensor*   (owned by session)
 *   interpreter->getSessionOutput(sess, name)  → Tensor*
 *   MNN::Tensor hostTensor(deviceTensor, CAFFE) → host copy
 *   deviceTensor->copyFromHostTensor(&host)    → write input
 *   interpreter->runSession(sess)              → execute graph
 *   deviceTensor->copyToHostTensor(&host)      → read output
 *   interpreter->releaseSession(sess)
 *   MNN::Interpreter::destroy(interpreter)
 */

#include <jni.h>
#include <android/log.h>

#include <cstring>
#include <cstdlib>
#include <string>
#include <algorithm>

// MNN 2.9.0 headers (extracted by setup_mnn_290.sh)
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ErrorCode.hpp"

#define LOG_TAG  "RGMNNJni"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

// ── Handle casts ─────────────────────────────────────────────────────────────
static inline MNN::Interpreter* toNet(jlong h)  { return reinterpret_cast<MNN::Interpreter*>(h); }
static inline MNN::Session*     toSess(jlong h) { return reinterpret_cast<MNN::Session*>(h); }

// ── JNI_OnLoad ───────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    LOGI("JNI_OnLoad: MNN 2.9.0 JNI bridge loaded ✓");
    return JNI_VERSION_1_6;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. nativeIsJniLoaded() → boolean
//    Runtime self-check: confirms we're in real JNI mode, not a no-op stub.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jboolean JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeIsJniLoaded(
        JNIEnv*, jobject) {
    return JNI_TRUE;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. nativeGetVersion() → String
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jstring JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeGetVersion(
        JNIEnv* env, jobject) {
    // MNN version string is defined in MNNDefine.h as MNN_VERSION
    return env->NewStringUTF(MNN_VERSION);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. nativeCreateNet(modelPath: String) → Long  (0 on failure)
//    Loads .mnn file. The .mnn.weight external data file must be in the
//    same directory and will be loaded automatically by MNN.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jlong JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeCreateNet(
        JNIEnv* env, jobject, jstring modelPath) {
    if (!modelPath) { LOGE("nativeCreateNet: null path"); return 0L; }

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("nativeCreateNet: loading %s", path);

    MNN::Interpreter* net = MNN::Interpreter::createFromFile(path);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!net) {
        LOGE("nativeCreateNet: MNN::Interpreter::createFromFile failed");
        return 0L;
    }
    LOGI("nativeCreateNet: success, handle=%p", net);
    return reinterpret_cast<jlong>(net);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. nativeCreateSession(netHandle, numThreads, forwardType) → Long (0 fail)
//    forwardType: 0=CPU, 3=OpenCL, 7=Vulkan
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jlong JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeCreateSession(
        JNIEnv*, jobject, jlong netHandle, jint numThreads, jint forwardType) {
    MNN::Interpreter* net = toNet(netHandle);
    if (!net) { LOGE("nativeCreateSession: invalid net handle"); return 0L; }

    MNN::ScheduleConfig schedConfig;
    schedConfig.numThread = static_cast<int>(numThreads);
    schedConfig.type      = static_cast<MNNForwardType>(forwardType);

    // BackendConfig: precision HIGH = FP32 to avoid accumulated FP16 errors
    // on STGCN (error budget 1e-5 target). FNO/Risk can use AUTO.
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    backendConfig.power     = MNN::BackendConfig::Power_Normal;
    schedConfig.backendConfig = &backendConfig;

    MNN::Session* sess = net->createSession(schedConfig);
    if (!sess) {
        LOGE("nativeCreateSession: createSession failed");
        return 0L;
    }
    LOGI("nativeCreateSession: success, sess=%p threads=%d", sess, (int)numThreads);
    return reinterpret_cast<jlong>(sess);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. nativeSetInput(netHandle, sessHandle, name, data) → Boolean
//    Writes float[] data into the named input tensor.
//    Uses CAFFE (NCHW) layout which matches all three model exports.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jboolean JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeSetInput(
        JNIEnv* env, jobject,
        jlong netHandle, jlong sessHandle, jstring name, jfloatArray data) {
    MNN::Interpreter* net  = toNet(netHandle);
    MNN::Session*     sess = toSess(sessHandle);
    if (!net || !sess || !name || !data) {
        LOGE("nativeSetInput: null argument");
        return JNI_FALSE;
    }

    const char* cName = env->GetStringUTFChars(name, nullptr);
    MNN::Tensor* deviceTensor = net->getSessionInput(sess, cName);
    env->ReleaseStringUTFChars(name, cName);

    if (!deviceTensor) {
        LOGE("nativeSetInput: tensor '%s' not found in session", cName);
        return JNI_FALSE;
    }

    // Create host tensor with same shape/type as the device tensor
    MNN::Tensor hostTensor(deviceTensor, MNN::Tensor::CAFFE);

    jsize   javaLen  = env->GetArrayLength(data);
    jfloat* javaPtr  = env->GetFloatArrayElements(data, nullptr);
    int     hostElems = hostTensor.elementSize();
    int     copyElems = std::min((int)javaLen, hostElems);

    if (javaLen < hostElems) {
        LOGW("nativeSetInput: data size %d < tensor size %d, zero-padding tail",
             (int)javaLen, hostElems);
        memset(hostTensor.host<float>(), 0, hostElems * sizeof(float));
    }
    memcpy(hostTensor.host<float>(), javaPtr, copyElems * sizeof(float));
    env->ReleaseFloatArrayElements(data, javaPtr, JNI_ABORT);

    // Push to device tensor
    deviceTensor->copyFromHostTensor(&hostTensor);
    return JNI_TRUE;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. nativeRun(netHandle, sessHandle) → Boolean
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jboolean JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeRun(
        JNIEnv*, jobject, jlong netHandle, jlong sessHandle) {
    MNN::Interpreter* net  = toNet(netHandle);
    MNN::Session*     sess = toSess(sessHandle);
    if (!net || !sess) { LOGE("nativeRun: null handle"); return JNI_FALSE; }

    MNN::ErrorCode code = net->runSession(sess);
    if (code != MNN::NO_ERROR) {
        LOGE("nativeRun: runSession failed, code=%d", (int)code);
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. nativeGetOutput(netHandle, sessHandle, name) → FloatArray? (null on fail)
//    Reads the named output tensor into a new Java float[].
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeGetOutput(
        JNIEnv* env, jobject,
        jlong netHandle, jlong sessHandle, jstring name) {
    MNN::Interpreter* net  = toNet(netHandle);
    MNN::Session*     sess = toSess(sessHandle);
    if (!net || !sess || !name) {
        LOGE("nativeGetOutput: null argument");
        return nullptr;
    }

    const char* cName = env->GetStringUTFChars(name, nullptr);
    MNN::Tensor* deviceTensor = net->getSessionOutput(sess, cName);
    env->ReleaseStringUTFChars(name, cName);

    if (!deviceTensor) {
        LOGE("nativeGetOutput: tensor '%s' not found", cName);
        return nullptr;
    }

    MNN::Tensor hostTensor(deviceTensor, MNN::Tensor::CAFFE);
    deviceTensor->copyToHostTensor(&hostTensor);

    int elems = hostTensor.elementSize();
    if (elems <= 0) {
        LOGE("nativeGetOutput: elementSize=%d for '%s'", elems, cName);
        return nullptr;
    }

    jfloatArray result = env->NewFloatArray(elems);
    if (!result) { LOGE("nativeGetOutput: NewFloatArray failed"); return nullptr; }
    env->SetFloatArrayRegion(result, 0, elems, hostTensor.host<float>());
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. nativeReleaseSession(netHandle, sessHandle)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT void JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeReleaseSession(
        JNIEnv*, jobject, jlong netHandle, jlong sessHandle) {
    MNN::Interpreter* net  = toNet(netHandle);
    MNN::Session*     sess = toSess(sessHandle);
    if (net && sess) {
        net->releaseSession(sess);
        LOGI("nativeReleaseSession: released sess=%p", sess);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. nativeReleaseNet(netHandle)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT void JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeReleaseNet(
        JNIEnv*, jobject, jlong netHandle) {
    MNN::Interpreter* net = toNet(netHandle);
    if (net) {
        MNN::Interpreter::destroy(net);
        LOGI("nativeReleaseNet: destroyed net=%p", net);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. nativeGetTensorShape(netHandle, sessHandle, name, isInput) → int[]
//     Returns shape array for validation; null if tensor not found.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jintArray JNICALL
Java_com_healthai_ankle_inference_RGMNNBridge_nativeGetTensorShape(
        JNIEnv* env, jobject,
        jlong netHandle, jlong sessHandle, jstring name, jboolean isInput) {
    MNN::Interpreter* net  = toNet(netHandle);
    MNN::Session*     sess = toSess(sessHandle);
    if (!net || !sess || !name) return nullptr;

    const char* cName = env->GetStringUTFChars(name, nullptr);
    MNN::Tensor* t = isInput
        ? net->getSessionInput(sess, cName)
        : net->getSessionOutput(sess, cName);
    env->ReleaseStringUTFChars(name, cName);

    if (!t) return nullptr;

    const auto& dims = t->shape();
    jintArray result = env->NewIntArray((jsize)dims.size());
    std::vector<jint> jdims(dims.begin(), dims.end());
    env->SetIntArrayRegion(result, 0, (jsize)dims.size(), jdims.data());
    return result;
}
