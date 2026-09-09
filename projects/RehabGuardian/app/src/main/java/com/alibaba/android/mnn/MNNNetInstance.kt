package com.alibaba.android.mnn

/**
 * Stub class for MNN NetInstance.
 *
 * The real MNN Android library (com.alibaba.android:mnn) is not published to Maven.
 * This stub provides the minimal API surface so the project compiles.
 * At runtime, if MNN is not present, RGPhaseAEngine.init() catches the error and
 * gracefully degrades (returns lastResult from infer()).
 */
class MNNNetInstance private constructor() {

    fun getSessionInput(session: Session?, name: String): MNNTable {
        throw UnsupportedOperationException("MNN not available — AI inference disabled")
    }

    fun getSessionOutput(session: Session?, name: String): MNNTable {
        throw UnsupportedOperationException("MNN not available — AI inference disabled")
    }

    fun runSession(session: Session?) {
        throw UnsupportedOperationException("MNN not available — AI inference disabled")
    }

    fun releaseSession(session: Session?) {
        // no-op
    }

    fun release() {
        // no-op
    }

    fun createSession(config: Config): Session {
        throw UnsupportedOperationException("MNN not available — AI inference disabled")
    }

    companion object {
        fun createFromFile(path: String): MNNNetInstance? {
            // Return null to signal MNN unavailable; caller handles null gracefully
            return null
        }
    }

    /**
     * Stub table wrapper around a FloatArray.
     */
    class MNNTable internal constructor(var floatData: FloatArray = FloatArray(0)) {
        fun setInputFloatData(data: FloatArray) {
            this.floatData = data
        }
    }

    /**
     * Stub session placeholder.
     */
    class Session

    /**
     * Stub config.
     */
    class Config {
        var numThread: Int = 4
        var forwardType: Int = 0
    }
}
