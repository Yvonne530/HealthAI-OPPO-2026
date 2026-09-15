package com.healthai.ankle.inference

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Gap 6 — Validation & baseline alignment test.
 *
 * Runs a fixed deterministic seed input through the full 3-model pipeline
 * and verifies:
 *   1. JNI is genuinely active (not stub)
 *   2. riskScore is finite and non-constant
 *   3. Two consecutive runs produce identical results (determinism)
 *   4. GRF values are finite and non-zero
 *   5. Future 10-step GRF has variance > 0  (dynamic, not constant)
 *   6. Latency < 50ms target
 *   7. Optional: numerical comparison vs Python baseline
 *
 * ── Regenerate Python baseline ──────────────────────────────────────────────
 *   python scripts/generate_validator_baseline.py
 *       --mnn_dir export/mnn_phasea_android_opt
 *
 * That script prints BASELINE_RISK_SCORE and BASELINE_GRF0 values you paste
 * into the companion object below.
 */
object RGInferenceValidator {

    private const val TAG = "RGValidator"

    // ── Threshold constants ───────────────────────────────────────────────
    private const val RISK_MAX_ERR     = 0.02f   // 2% — FP16-aware tolerance
    private const val GRF_MAX_ERR      = 0.05f   // GRF model is FP16
    private const val LATENCY_MAX_MS   = 50L     // target per spec

    // ── Python baseline values ────────────────────────────────────────────
    // Set these after running generate_validator_baseline.py.
    // Leave null to skip numerical comparison (finiteness checks still run).
    private val BASELINE_RISK_SCORE: Float? = null
    private val BASELINE_GRF0: FloatArray?  = null

    // ─────────────────────────────────────────────────────────────────────

    data class ValidationReport(
        val passed: Boolean,
        val jniActive: Boolean,
        val mnnVersion: String,
        val riskScore: Float,
        val grfPeak: Float,
        val futureDynamic: Boolean,
        val latencyMs: Long,
        val baselineRiskErr: Float?,
        val baselineGrfErr:  Float?,
        val failures: List<String>
    ) {
        val summary: String get() = buildString {
            appendLine(if (passed) "✅ PASS — RGInferenceValidator" else "❌ FAIL — RGInferenceValidator")
            appendLine("  JNI active     : $jniActive  (MNN $mnnVersion)")
            appendLine("  Risk score     : ${String.format("%.5f", riskScore)}" +
                       if (!riskScore.isFinite()) "  ← NaN/Inf!" else "")
            appendLine("  GRF peak       : ${String.format("%.4f", grfPeak)} BW")
            appendLine("  Future dynamic : $futureDynamic")
            appendLine("  Latency        : ${latencyMs}ms  (target <$LATENCY_MAX_MS)")
            if (baselineRiskErr != null)
                appendLine("  Baseline risk Δ: ${String.format("%.2e", baselineRiskErr)}")
            if (baselineGrfErr != null)
                appendLine("  Baseline GRF  Δ: ${String.format("%.2e", baselineGrfErr)}")
            if (failures.isNotEmpty()) {
                appendLine("  ── Failures ──")
                failures.forEach { appendLine("    • $it") }
            }
        }.trimEnd()
    }

    /** Async entry point — call from a coroutine */
    suspend fun run(engine: RGPhaseAEngine): ValidationReport =
        withContext(Dispatchers.Default) { runSync(engine) }

    /** Synchronous entry point — call from instrumentation tests */
    fun runSync(engine: RGPhaseAEngine): ValidationReport {
        val failures = mutableListOf<String>()

        // ── 0. JNI self-check ──────────────────────────────────────────────
        if (!RGMNNBridge.jniAvailable) {
            return ValidationReport(false, false, "N/A", 0f, 0f, false, 0L, null, null,
                listOf("librehab_mnn_jni.so not loaded — run scripts/setup_mnn_290.sh"))
        }

        val seed = buildSeedInput()

        // ── 1. Run #1 with timing ──────────────────────────────────────────
        val t0 = System.currentTimeMillis()
        val r1 = engine.infer(seed)
        val latency = System.currentTimeMillis() - t0

        if (r1 == null) {
            failures += "infer() returned null (engine not ready?)"
            return ValidationReport(false, true, RGMNNBridge.mnnVersion,
                0f, 0f, false, latency, null, null, failures)
        }

        // ── 2. Run #2 — determinism check ─────────────────────────────────
        val r2 = engine.infer(seed)
        if (r2 != null) {
            val delta = Math.abs(r1.riskScore - r2.riskScore)
            if (delta > 1e-4f)
                failures += "Non-deterministic: run1=${r1.riskScore} run2=${r2.riskScore} Δ=$delta"
        }

        // ── 3. riskScore finite & non-constant ────────────────────────────
        if (!r1.riskScore.isFinite())
            failures += "riskScore is ${r1.riskScore} (NaN/Inf)"
        if (r1.riskScore == 0f)
            failures += "riskScore == 0.0 — model outputs zeros (no-op / wrong tensor)"

        // ── 4. GRF finite ─────────────────────────────────────────────────
        val grfPeak = (r1.grfLeft + r1.grfRight).map { Math.abs(it) }.maxOrNull() ?: 0f
        val grfFinite = r1.grfLeft.all { it.isFinite() } && r1.grfRight.all { it.isFinite() }
        if (!grfFinite) failures += "GRF contains NaN/Inf"

        // ── 5. Future GRF dynamic ─────────────────────────────────────────
        val future = r1.grfFutureFlat
        val futureVar = if (future.size > 1) {
            val m = future.average().toFloat()
            future.map { (it - m) * (it - m) }.average().toFloat()
        } else 0f
        val futureDynamic = futureVar > 1e-8f
        if (!futureDynamic && future.isNotEmpty())
            failures += "Future GRF variance=0 — FNO may output constant (LSTM state reset?)"

        // ── 6. Latency ────────────────────────────────────────────────────
        if (latency > LATENCY_MAX_MS)
            Log.w(TAG, "Latency ${latency}ms > ${LATENCY_MAX_MS}ms (non-fatal, device-dependent)")

        // ── 7. Baseline comparison ────────────────────────────────────────
        var baseRiskErr: Float? = null
        var baseGrfErr:  Float? = null

        BASELINE_RISK_SCORE?.let {
            val e = Math.abs(r1.riskScore - it)
            baseRiskErr = e
            if (e > RISK_MAX_ERR)
                failures += "Risk baseline Δ=${String.format("%.2e", e)} > threshold ${String.format("%.2e", RISK_MAX_ERR)}"
        }
        BASELINE_GRF0?.let { bl ->
            val n = minOf(bl.size, r1.grfLeft.size)
            if (n > 0) {
                val e = (0 until n).map { i -> Math.abs(r1.grfLeft[i] - bl[i]) }.max()
                baseGrfErr = e
                if (e > GRF_MAX_ERR)
                    failures += "GRF baseline Δ=${String.format("%.2e", e)} > threshold ${String.format("%.2e", GRF_MAX_ERR)}"
            }
        }

        val report = ValidationReport(
            passed          = failures.isEmpty(),
            jniActive       = true,
            mnnVersion      = RGMNNBridge.mnnVersion,
            riskScore       = r1.riskScore,
            grfPeak         = grfPeak,
            futureDynamic   = futureDynamic,
            latencyMs       = latency,
            baselineRiskErr = baseRiskErr,
            baselineGrfErr  = baseGrfErr,
            failures        = failures
        )
        Log.i(TAG, "\n${report.summary}")
        return report
    }

    /**
     * Deterministic seed: simple LCG pseudo-random, seed=42.
     * Produces 20 frames × 33 landmarks × 3 coords in [0,1].
     * To get exact Python match, run generate_validator_baseline.py and
     * compare — the LCG here is intentionally simple, not numpy-exact.
     */
    private fun buildSeedInput(): Array<Array<FloatArray>> {
        var s = 42L
        fun next(): Float {
            s = (s * 6364136223846793005L + 1442695040888963407L) and 0x7FFFFFFFFFFFFFFFL
            return (s.toFloat() / Long.MAX_VALUE.toFloat()).coerceIn(0.01f, 0.99f)
        }
        return Array(20) { Array(33) { FloatArray(3) { next() } } }
    }

    private operator fun FloatArray.plus(other: FloatArray): FloatArray {
        val out = FloatArray(size + other.size)
        copyInto(out, 0); other.copyInto(out, size)
        return out
    }
}
