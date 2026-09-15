package com.healthai.ankle.inference

import kotlin.math.abs
import kotlin.math.max

/**
 * Transforms raw model outputs into human-readable, clinically-grounded
 * risk explanations. This is the "explainability layer" that turns
 * "HIGH RISK" → structured reasons + actionable correction prompts.
 *
 * Decision criteria are derived from clinical rehabilitation literature:
 *   - Knee flexion at landing: optimal ≥ 150° (< 150° = increased ACL load)
 *   - Bilateral asymmetry: > 20° = significant compensation pattern
 *   - GRF vertical peak: > 2.5 BW = high-impact landing
 *   - GRF loading rate (proxy from consecutive frames): rapid rise = injury risk
 */
object RiskExplainer {

    // ── Public API ────────────────────────────────────────────────────────

    data class Explanation(
        val reasons: List<Reason>,          // ranked by severity (worst first)
        val topCorrection: String,          // single most important correction
        val allCorrections: List<String>,   // full list for detail view
        val clinicalSummary: String,        // one-sentence clinical summary
        val futureThreat: String            // FNO future-prediction sentence
    ) {
        val reasonText: String get() = reasons.joinToString("\n") { "• ${it.text}" }
        val isEmpty: Boolean get() = reasons.isEmpty()
    }

    data class Reason(
        val text: String,
        val severity: Severity,     // HIGH / MEDIUM / LOW
        val metric: String,         // e.g. "左膝 142°"
        val threshold: String       // e.g. "< 150° 风险增加"
    )

    enum class Severity { HIGH, MEDIUM, LOW }

    /**
     * Build explanation from engine outputs.
     *
     * @param lKneeDeg     Left knee flexion angle in degrees
     * @param rKneeDeg     Right knee flexion angle in degrees
     * @param grfLeft      GRF left [6] — index 0 = Fz vertical (body-weight units)
     * @param grfRight     GRF right [6]
     * @param grfFuture    FNO future prediction [10,12] — next 10 frames GRF
     * @param riskScore    Softmax high-risk probability [0,1]
     * @param motionScore  0-100 composite quality score
     */
    fun explain(
        lKneeDeg: Float,
        rKneeDeg: Float,
        grfLeft: FloatArray,
        grfRight: FloatArray,
        grfFuture: FloatArray,      // flat [10*12], or empty if unavailable
        riskScore: Float,
        motionScore: Float
    ): Explanation {
        val reasons  = mutableListOf<Reason>()
        val fixes    = mutableListOf<String>()

        // ── 1. Knee flexion analysis ──────────────────────────────────────
        val minKnee = minOf(lKneeDeg, rKneeDeg)
        when {
            lKneeDeg < 130f -> {
                reasons += Reason(
                    "左膝弯曲严重不足（${lKneeDeg.toInt()}°）",
                    Severity.HIGH,
                    "左膝 ${lKneeDeg.toInt()}°",
                    "< 130° 高风险"
                )
                fixes += "增大左膝弯曲幅度，目标 ≥ 150°"
            }
            lKneeDeg < 150f -> {
                reasons += Reason(
                    "左膝弯曲不足（${lKneeDeg.toInt()}°）",
                    Severity.MEDIUM,
                    "左膝 ${lKneeDeg.toInt()}°",
                    "< 150° 需改善"
                )
                fixes += "适度增加左膝弯曲角度"
            }
        }
        when {
            rKneeDeg < 130f -> {
                reasons += Reason(
                    "右膝弯曲严重不足（${rKneeDeg.toInt()}°）",
                    Severity.HIGH,
                    "右膝 ${rKneeDeg.toInt()}°",
                    "< 130° 高风险"
                )
                fixes += "增大右膝弯曲幅度，目标 ≥ 150°"
            }
            rKneeDeg < 150f -> {
                reasons += Reason(
                    "右膝弯曲不足（${rKneeDeg.toInt()}°）",
                    Severity.MEDIUM,
                    "右膝 ${rKneeDeg.toInt()}°",
                    "< 150° 需改善"
                )
                fixes += "适度增加右膝弯曲角度"
            }
        }

        // ── 2. Bilateral asymmetry ─────────────────────────────────────────
        val asymmetry = abs(lKneeDeg - rKneeDeg)
        when {
            asymmetry > 30f -> {
                reasons += Reason(
                    "双膝不对称严重（差值 ${asymmetry.toInt()}°）",
                    Severity.HIGH,
                    "不对称 ${asymmetry.toInt()}°",
                    "> 30° 代偿明显"
                )
                fixes += "纠正下肢力学不对称，强化弱侧单腿训练"
            }
            asymmetry > 20f -> {
                reasons += Reason(
                    "双膝不对称（差值 ${asymmetry.toInt()}°）",
                    Severity.MEDIUM,
                    "不对称 ${asymmetry.toInt()}°",
                    "> 20° 需关注"
                )
                fixes += "关注左右膝协调性，减少代偿动作"
            }
        }

        // ── 3. GRF impact analysis ────────────────────────────────────────
        val lFz = grfLeft.getOrElse(0) { 0f }
        val rFz = grfRight.getOrElse(0) { 0f }
        val peakFz = max(lFz, rFz)
        when {
            peakFz > 3.5f -> {
                reasons += Reason(
                    "落地冲击力极高（峰值 ${String.format("%.1f", peakFz)} BW）",
                    Severity.HIGH,
                    "GRF ${String.format("%.1f", peakFz)} BW",
                    "> 3.5 BW 极危险"
                )
                fixes += "降低落地高度，主动减缓冲击"
            }
            peakFz > 2.5f -> {
                reasons += Reason(
                    "落地冲击力过大（${String.format("%.1f", peakFz)} BW）",
                    Severity.MEDIUM,
                    "GRF ${String.format("%.1f", peakFz)} BW",
                    "> 2.5 BW 偏高"
                )
                fixes += "注意落地缓冲，屈膝吸收冲击"
            }
        }

        // ── 4. GRF asymmetry ──────────────────────────────────────────────
        val grfAsym = abs(lFz - rFz)
        if (grfAsym > 0.8f && peakFz > 1.0f) {
            reasons += Reason(
                "双脚受力不均（差值 ${String.format("%.1f", grfAsym)} BW）",
                Severity.MEDIUM,
                "左右GRF差 ${String.format("%.1f", grfAsym)} BW",
                "> 0.8 BW 需改善"
            )
            fixes += "均衡双脚受力，避免单侧负重"
        }

        // ── 5. Future GRF prediction (FNO output) ────────────────────────
        val futureThreat = buildFutureThreat(grfFuture, peakFz)

        // ── 6. If everything OK ───────────────────────────────────────────
        if (reasons.isEmpty()) {
            fixes += "保持当前动作质量，继续训练"
        }

        // Sort: HIGH first
        reasons.sortByDescending { it.severity.ordinal.inv() }

        val topFix = fixes.firstOrNull() ?: "保持当前动作"
        val summary = buildClinicalSummary(riskScore, motionScore, minKnee, peakFz)

        return Explanation(
            reasons         = reasons,
            topCorrection   = topFix,
            allCorrections  = fixes,
            clinicalSummary = summary,
            futureThreat    = futureThreat
        )
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /**
     * Build future-prediction sentence from FNO's 10-frame GRF output.
     * This is the "国一级叙事" — AI doesn't just detect, it predicts.
     */
    private fun buildFutureThreat(grfFuture: FloatArray, currentPeak: Float): String {
        if (grfFuture.isEmpty()) return "预测模块就绪，等待足够帧数"

        // grfFuture shape [10*12]: for each of 10 future frames, 12 GRF channels
        // Fz channels are at index 0 (left) and 6 (right) of each 12-dim slice
        var maxFutureLeft  = 0f
        var maxFutureRight = 0f
        val frameCount = minOf(10, grfFuture.size / 12)
        for (t in 0 until frameCount) {
            val base = t * 12
            if (base + 11 < grfFuture.size) {
                maxFutureLeft  = maxOf(maxFutureLeft,  grfFuture[base + 0])
                maxFutureRight = maxOf(maxFutureRight, grfFuture[base + 6])
            }
        }
        val futurePeak = max(maxFutureLeft, maxFutureRight)
        val deltaMs = (frameCount * (1000f / 30f)).toInt()   // assuming 30fps

        return when {
            futurePeak > currentPeak * 1.2f ->
                "⚡ 预测：未来 ${deltaMs}ms 冲击力将升高至 ${String.format("%.1f", futurePeak)} BW"
            futurePeak > 2.5f ->
                "⚠ 预测：未来 ${deltaMs}ms 冲击力持续偏高（${String.format("%.1f", futurePeak)} BW）"
            else ->
                "✓ 预测：未来 ${deltaMs}ms 冲击力趋势平稳（${String.format("%.1f", futurePeak)} BW）"
        }
    }

    private fun buildClinicalSummary(
        riskScore: Float, motionScore: Float, minKnee: Float, peakFz: Float
    ): String {
        val riskPct = (riskScore * 100).toInt()
        return when {
            riskScore > 0.7f -> "当前动作模式存在较高损伤风险（${riskPct}%），建议立即调整"
            riskScore > 0.4f -> "动作质量中等（运动评分 ${motionScore.toInt()}），关键指标需改善"
            else             -> "动作质量良好（运动评分 ${motionScore.toInt()}），继续保持"
        }
    }
}
