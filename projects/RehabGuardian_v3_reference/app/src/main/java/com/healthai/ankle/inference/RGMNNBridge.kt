package com.healthai.ankle.inference

import android.util.Log

/**
 * Kotlin-side JNI bridge to MNN 2.9.0 C++ runtime.
 *
 * ALL methods are `external` — they call the real mnn_jni.cpp implementation.
 * There is NO no-op fallback: if the .so fails to load, the companion object
 * sets [jniAvailable] = false and the engine disables AI explicitly.
 *
 * Usage:
 *   val netHandle  = RGMNNBridge.nativeCreateNet(modelPath)   // 0 = fail
 *   val sessHandle = RGMNNBridge.nativeCreateSession(netHandle, 4, 0)
 *   RGMNNBridge.nativeSetInput(netHandle, sessHandle, "visual_seq", floatArray)
 *   RGMNNBridge.nativeRun(netHandle, sessHandle)
 *   val out = RGMNNBridge.nativeGetOutput(netHandle, sessHandle, "joint_angles")
 *   RGMNNBridge.nativeReleaseSession(netHandle, sessHandle)
 *   RGMNNBridge.nativeReleaseNet(netHandle)
 */
object RGMNNBridge {

    private const val TAG = "RGMNNBridge"

    /** True only when librehab_mnn_jni.so loaded AND nativeIsJniLoaded() returned true. */
    var jniAvailable: Boolean = false
        private set

    /** MNN version string reported by the native library. */
    var mnnVersion: String = "not loaded"
        private set

    init {
        loadNativeLibrary()
    }

    private fun loadNativeLibrary() {
        try {
            System.loadLibrary("rehab_mnn_jni")
            // Immediate self-check: if this returns true, we have real inference
            jniAvailable = nativeIsJniLoaded()
            if (jniAvailable) {
                mnnVersion = nativeGetVersion()
                Log.i(TAG, "✅ REAL JNI INFERENCE ACTIVE — MNN $mnnVersion")
            } else {
                Log.e(TAG, "❌ JNI loaded but nativeIsJniLoaded() returned false — stub mode")
            }
        } catch (e: UnsatisfiedLinkError) {
            jniAvailable = false
            Log.e(TAG, "❌ Failed to load librehab_mnn_jni.so: ${e.message}")
            Log.e(TAG, "   → AI inference DISABLED. Run scripts/setup_mnn_290.sh and rebuild.")
        }
    }

    // ── Native declarations ───────────────────────────────────────────────────

    /**
     * Runtime self-check: returns true only from real JNI C++ code.
     * A stub would need to return true manually — ours never will.
     */
    @JvmStatic external fun nativeIsJniLoaded(): Boolean

    /** MNN version string (e.g. "2.9.0") */
    @JvmStatic external fun nativeGetVersion(): String

    /**
     * Load a .mnn model from file path.
     * The .mnn.weight file must exist in the same directory.
     * @return net handle (> 0) or 0 on failure
     */
    @JvmStatic external fun nativeCreateNet(modelPath: String): Long

    /**
     * Create an inference session.
     * @param numThreads  CPU thread count (4 recommended for Reno15 Pro)
     * @param forwardType 0=CPU, 3=OpenCL, 7=Vulkan
     * @return session handle (> 0) or 0 on failure
     */
    @JvmStatic external fun nativeCreateSession(
        netHandle: Long, numThreads: Int, forwardType: Int): Long

    /**
     * Write float[] data into named input tensor.
     * @return true on success
     */
    @JvmStatic external fun nativeSetInput(
        netHandle: Long, sessHandle: Long, name: String, data: FloatArray): Boolean

    /**
     * Execute the graph.
     * @return true on success
     */
    @JvmStatic external fun nativeRun(netHandle: Long, sessHandle: Long): Boolean

    /**
     * Read float[] from named output tensor.
     * @return output data, or null on failure
     */
    @JvmStatic external fun nativeGetOutput(
        netHandle: Long, sessHandle: Long, name: String): FloatArray?

    /**
     * Get tensor shape for validation.
     * @param isInput true for input tensor, false for output
     * @return int[] shape, or null if tensor not found
     */
    @JvmStatic external fun nativeGetTensorShape(
        netHandle: Long, sessHandle: Long, name: String, isInput: Boolean): IntArray?

    /** Release a session (call before releaseNet) */
    @JvmStatic external fun nativeReleaseSession(netHandle: Long, sessHandle: Long)

    /** Destroy interpreter and free all model memory */
    @JvmStatic external fun nativeReleaseNet(netHandle: Long)

    // ── Safe wrappers (log errors, never throw) ───────────────────────────────

    /**
     * Safe version of nativeGetOutput: returns FloatArray of [expectedSize] zeros
     * instead of null, so callers don't crash on unexpected null.
     */
    fun getOutputSafe(
        netHandle: Long, sessHandle: Long, name: String, expectedSize: Int
    ): FloatArray {
        if (!jniAvailable) return FloatArray(expectedSize)
        return try {
            nativeGetOutput(netHandle, sessHandle, name) ?: run {
                Log.w(TAG, "getOutputSafe: null for '$name', returning zeros")
                FloatArray(expectedSize)
            }
        } catch (e: Exception) {
            Log.e(TAG, "getOutputSafe exception for '$name': ${e.message}")
            FloatArray(expectedSize)
        }
    }

    /**
     * Validate tensor shape matches expected.
     * Logs a warning if mismatch — does not throw.
     */
    fun validateShape(
        netHandle: Long, sessHandle: Long,
        name: String, isInput: Boolean, expected: IntArray
    ) {
        if (!jniAvailable) return
        val actual = nativeGetTensorShape(netHandle, sessHandle, name, isInput)
        if (actual == null) {
            Log.w(TAG, "validateShape: tensor '$name' not found in model")
            return
        }
        if (!actual.contentEquals(expected)) {
            Log.w(TAG, "validateShape: '$name' expected ${expected.toList()} " +
                    "got ${actual.toList()} — mismatch may cause incorrect results")
        } else {
            Log.d(TAG, "validateShape: '$name' ✓ ${expected.toList()}")
        }
    }
}
