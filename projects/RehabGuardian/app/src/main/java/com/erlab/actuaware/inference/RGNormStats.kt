package com.healthai.ankle.inference

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * Loads and sanitizes normalization statistics from assets/norm_stats.json.
 * Handles "NaN" strings and Inf values with safe fallbacks.
 */
class RGNormStats(context: Context) {

    val jointAnglesMean: FloatArray   // [23]
    val jointAnglesStd: FloatArray    // [23]
    val grfLeftMean: FloatArray?      // [6]  optional
    val grfLeftStd: FloatArray?       // [6]  optional
    val grfRightMean: FloatArray?     // [6]  optional
    val grfRightStd: FloatArray?      // [6]  optional

    init {
        val json = runCatching {
            context.assets.open("norm_stats.json")
                .bufferedReader().use { it.readText() }
                .let { JSONObject(it) }
        }.getOrNull()

        jointAnglesMean = json?.parseArray("joint_angles_mean", 23) ?: FloatArray(23) { 0f }
        jointAnglesStd  = json?.parseArray("joint_angles_std",  23)?.clampStd() ?: FloatArray(23) { 1f }
        grfLeftMean     = json?.parseArrayOrNull("grf_left_mean",  6)
        grfLeftStd      = json?.parseArrayOrNull("grf_left_std",   6)?.clampStd()
        grfRightMean    = json?.parseArrayOrNull("grf_right_mean", 6)
        grfRightStd     = json?.parseArrayOrNull("grf_right_std",  6)?.clampStd()

        Log.d(TAG, "NormStats loaded: ja_mean[0]=${jointAnglesMean[0]}, ja_std[0]=${jointAnglesStd[0]}")
    }

    /** Normalize a value: (x - mean) / std */
    fun normalizeJointAngle(value: Float, idx: Int): Float {
        val v = if (value.isFinite()) value else 0f
        return (v - jointAnglesMean[idx]) / jointAnglesStd[idx]
    }

    /** Denormalize GRF left channel (returns raw if stats missing) */
    fun denormGrfLeft(normalized: FloatArray): FloatArray {
        if (grfLeftMean == null || grfLeftStd == null) return normalized
        return FloatArray(6) { i -> normalized[i] * grfLeftStd[i] + grfLeftMean[i] }
    }

    /** Denormalize GRF right channel (returns raw if stats missing) */
    fun denormGrfRight(normalized: FloatArray): FloatArray {
        if (grfRightMean == null || grfRightStd == null) return normalized
        return FloatArray(6) { i -> normalized[i] * grfRightStd[i] + grfRightMean[i] }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private fun JSONObject.parseArray(key: String, size: Int): FloatArray {
        val arr = FloatArray(size) { 0f }
        if (!has(key)) return arr
        val ja = getJSONArray(key)
        for (i in 0 until minOf(size, ja.length())) {
            arr[i] = parseSafeFloat(ja.getString(i), fallback = 0f)
        }
        return arr
    }

    private fun JSONObject.parseArrayOrNull(key: String, size: Int): FloatArray? {
        if (!has(key)) return null
        return parseArray(key, size)
    }

    private fun parseSafeFloat(s: String, fallback: Float): Float {
        if (s.equals("NaN", ignoreCase = true) || s.equals("nan", ignoreCase = true)) return fallback
        return s.toFloatOrNull()?.let { if (it.isFinite()) it else fallback } ?: fallback
    }

    private fun FloatArray.clampStd(): FloatArray =
        FloatArray(size) { i -> maxOf(this[i], 1e-6f) }

    companion object {
        private const val TAG = "RGNormStats"
    }
}