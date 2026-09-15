package com.healthai.ankle.inference

import kotlin.math.sqrt

/**
 * Stateless feature construction helpers.
 * All allocation is caller-owned; no internal state is kept here.
 */
object RGFeatureBuilder {

    // ── §3.2  STGCN window slicing ───────────────────────────────────────

    /**
     * Slice a 5-frame chunk from the 20-frame ring buffer.
     * Returns a flat FloatArray of shape [1,5,33,3] = 4950 elements.
     */
    fun sliceStgcnInput(frames20: Array<Array<FloatArray>>, chunkStart: Int): FloatArray {
        // frames20[t] = Array<FloatArray>(33) each FloatArray size 3
        val out = FloatArray(1 * 5 * 33 * 3)
        var idx = 0
        for (t in chunkStart until chunkStart + 5) {
            val frame = frames20[t]
            for (lm in frame) {
                out[idx++] = sanitize(lm[0])
                out[idx++] = sanitize(lm[1])
                out[idx++] = if (lm.size > 2) sanitize(lm[2]) else 0f
            }
        }
        return out
    }

    /**
     * Given STGCN outputs for 4 chunks (each [23]), build jaSeq[20][23]
     * by repeating each chunk result across its 5 frames.
     */
    fun buildJaSeq(chunkResults: Array<FloatArray>): Array<FloatArray> {
        // chunkResults[c] = FloatArray(23) for chunk c=0..3
        val jaSeq = Array(20) { FloatArray(23) }
        for (c in 0 until 4) {
            val ja = chunkResults[c]
            for (t in c * 5 until (c + 1) * 5) {
                jaSeq[t] = ja.copyOf()
            }
        }
        return jaSeq
    }

    // ── §3.3  bio_seq construction [1,20,72] ─────────────────────────────

    /**
     * Build bio_seq from jaSeq.
     * Per timestep t: [ja(23) | vel(23) | acc(23) | com(3)] = 72 dims.
     */
    fun buildBioSeq(jaSeq: Array<FloatArray>, frames20: Array<Array<FloatArray>>): FloatArray {
        val out = FloatArray(1 * 20 * 72)
        val prevVel = FloatArray(23) { 0f }

        for (t in 0 until 20) {
            val base = t * 72
            val ja  = jaSeq[t]
            val vel = if (t == 0) FloatArray(23) else FloatArray(23) { i -> ja[i] - jaSeq[t - 1][i] }
            val acc = if (t == 0) FloatArray(23) else FloatArray(23) { i -> vel[i] - prevVel[i] }
            val com = computeComVel(frames20, t)

            System.arraycopy(sanitizeAll(ja),  0, out, base,      23)
            System.arraycopy(sanitizeAll(vel), 0, out, base + 23, 23)
            System.arraycopy(sanitizeAll(acc), 0, out, base + 46, 23)
            System.arraycopy(sanitizeAll(com), 0, out, base + 69,  3)

            if (t > 0) System.arraycopy(vel, 0, prevVel, 0, 23)
        }
        return out
    }

    // ── §3.5  risk_seq construction [1,20,35] ────────────────────────────

    /**
     * Build risk_seq.
     * Per timestep: [ja_norm(23) | grf0(12)] = 35 dims.
     * grf0 is the first-timestep slice of FNO grf_seq output (shape [1,10,12]).
     * All 20 timesteps share the same grf0.
     */
    fun buildRiskSeq(
        jaSeq: Array<FloatArray>,
        grf0: FloatArray,        // [12]
        normStats: RGNormStats
    ): FloatArray {
        val out = FloatArray(1 * 20 * 35)
        for (t in 0 until 20) {
            val base = t * 35
            val ja = jaSeq[t]
            for (i in 0 until 23) {
                out[base + i] = normStats.normalizeJointAngle(ja[i], i)
            }
            System.arraycopy(sanitizeAll(grf0), 0, out, base + 23, 12)
        }
        return out
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /**
     * Simplified center-of-mass velocity using hip and shoulder midpoints.
     * Returns 3D velocity vector relative to previous frame.
     * Falls back to zero if frame out of range.
     */
    private fun computeComVel(frames: Array<Array<FloatArray>>, t: Int): FloatArray {
        if (t == 0) return FloatArray(3)
        val cur  = comPosition(frames[t])
        val prev = comPosition(frames[t - 1])
        return FloatArray(3) { i -> sanitize(cur[i] - prev[i]) }
    }

    private fun comPosition(frame: Array<FloatArray>): FloatArray {
        // Use hip and shoulder midpoints as CoM proxy
        val lHip  = frame.getOrNull(RGMarkerMapper.L_HIP)  ?: FloatArray(3)
        val rHip  = frame.getOrNull(RGMarkerMapper.R_HIP)  ?: FloatArray(3)
        val lSh   = frame.getOrNull(RGMarkerMapper.L_SHOULDER) ?: FloatArray(3)
        val rSh   = frame.getOrNull(RGMarkerMapper.R_SHOULDER) ?: FloatArray(3)
        return FloatArray(3) { i ->
            (lHip[i] + rHip[i] + lSh[i] + rSh[i]) / 4f
        }
    }

    private fun sanitize(v: Float) = if (v.isFinite()) v else 0f

    private fun sanitizeAll(arr: FloatArray): FloatArray =
        FloatArray(arr.size) { i -> sanitize(arr[i]) }
}
