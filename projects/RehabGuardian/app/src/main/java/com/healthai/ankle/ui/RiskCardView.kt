package com.healthai.ankle.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.healthai.ankle.inference.RiskExplainer
import kotlin.math.min

/**
 * The visual centrepiece of the app.
 * Displays: risk level badge | confidence arc | top 3 reasons | future prediction line.
 *
 * Three-state color system:
 *   LOW    (#22C55E green)
 *   MEDIUM (#F59E0B amber)
 *   HIGH   (#EF4444 red)
 */
class RiskCardView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    // State
    private var riskLabel    = -1       // -1 = waiting
    private var riskScore    = 0f
    private var confidence   = 0f
    private var motionScore  = 0f
    private var explanation: RiskExplainer.Explanation? = null
    private var futureLine   = ""

    // Animation
    private var animatedArc  = 0f      // animated confidence arc sweep
    private val handler = android.os.Handler(android.os.Looper.getMainLooper())

    // Colors
    private val COL_LOW    = Color.parseColor("#22C55E")
    private val COL_MEDIUM = Color.parseColor("#F59E0B")
    private val COL_HIGH   = Color.parseColor("#EF4444")
    private val COL_BG     = Color.parseColor("#0D1B1E")
    private val COL_CARD   = Color.parseColor("#131F23")
    private val COL_TRACK  = Color.parseColor("#1E3A3A")
    private val COL_TEXT   = Color.WHITE
    private val COL_SUB    = Color.parseColor("#94A3B8")
    private val COL_REASON_HIGH   = Color.parseColor("#FCA5A5")
    private val COL_REASON_MEDIUM = Color.parseColor("#FDE68A")
    private val COL_REASON_LOW    = Color.parseColor("#86EFAC")

    private val bgPaint   = Paint().apply { color = COL_BG;   style = Paint.Style.FILL }
    private val cardPaint = Paint().apply { color = COL_CARD; style = Paint.Style.FILL }

    private fun arcTrackPaint() = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = COL_TRACK; style = Paint.Style.STROKE; strokeWidth = 16f
        strokeCap = Paint.Cap.ROUND
    }
    private fun arcFillPaint(color: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color = color; style = Paint.Style.STROKE; strokeWidth = 16f
        strokeCap = Paint.Cap.ROUND
    }
    private fun textPaint(size: Float, color: Int = COL_TEXT, bold: Boolean = false) =
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; textSize = size
            if (bold) typeface = Typeface.DEFAULT_BOLD
            textAlign = Paint.Align.CENTER
        }
    private fun leftTextPaint(size: Float, color: Int = COL_TEXT, bold: Boolean = false) =
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; textSize = size
            if (bold) typeface = Typeface.DEFAULT_BOLD
        }

    // ── Public API ────────────────────────────────────────────────────────

    fun update(
        label: Int, score: Float, conf: Float,
        motion: Float, expl: RiskExplainer.Explanation, future: String
    ) {
        riskLabel   = label
        riskScore   = score
        confidence  = conf
        motionScore = motion
        explanation = expl
        futureLine  = future
        animateArc(conf)
        invalidate()
    }

    fun setWaiting() {
        riskLabel = -1; explanation = null; futureLine = ""; invalidate()
    }

    // ── Draw ──────────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        canvas.drawRoundRect(0f, 0f, w, h, 20f, 20f, bgPaint)

        if (riskLabel < 0) {
            drawWaiting(canvas, w, h); return
        }

        val riskColor = riskColor()

        // ── Top section: arc + risk badge ────────────────────────────────
        val arcR = minOf(w * 0.18f, 72f)
        val arcCx = w * 0.18f + 24f; val arcCy = 20f + arcR + 12f
        drawConfidenceArc(canvas, arcCx, arcCy, arcR, riskColor)

        // Risk percentage in arc center
        canvas.drawText(
            "${(riskScore * 100).toInt()}%",
            arcCx, arcCy + 14f, textPaint(arcR * 0.55f, riskColor, bold = true)
        )
        canvas.drawText("高风险", arcCx, arcCy + arcR * 0.82f + 14f, textPaint(22f, COL_SUB))

        // Risk level badge (right of arc)
        val badgeX = arcCx + arcR + 20f
        val badgeY = arcCy - arcR * 0.2f
        drawRiskBadge(canvas, riskLabel, riskColor, badgeX, badgeY, w - badgeX - 12f)

        // Motion score
        canvas.drawText(
            "运动评分  ${motionScore.toInt()}",
            w - 80f, arcCy + arcR * 0.5f,
            leftTextPaint(24f, if (motionScore >= 70) COL_LOW else COL_MEDIUM, bold = true)
                .also { it.textAlign = Paint.Align.RIGHT }
        )

        // ── Divider ───────────────────────────────────────────────────────
        val divY = arcCy + arcR + 22f
        val dp = Paint().apply { color = COL_TRACK; strokeWidth = 1f }
        canvas.drawLine(16f, divY, w - 16f, divY, dp)

        // ── Reasons list ──────────────────────────────────────────────────
        val expl = explanation
        var lineY = divY + 32f
        if (expl == null || expl.isEmpty) {
            canvas.drawText("动作质量良好 ✓", w / 2f, lineY + 12f, textPaint(28f, COL_LOW))
        } else {
            expl.reasons.take(3).forEach { reason ->
                val rc = when (reason.severity) {
                    RiskExplainer.Severity.HIGH   -> COL_REASON_HIGH
                    RiskExplainer.Severity.MEDIUM -> COL_REASON_MEDIUM
                    RiskExplainer.Severity.LOW    -> COL_REASON_LOW
                }
                // Severity dot
                canvas.drawCircle(20f, lineY + 4f, 7f, Paint().also { it.color = rc })
                // Reason text
                canvas.drawText(reason.text, 36f, lineY + 12f, leftTextPaint(24f, rc))
                // Metric small
                canvas.drawText(
                    reason.metric, w - 12f, lineY + 12f,
                    leftTextPaint(20f, COL_SUB).also { it.textAlign = Paint.Align.RIGHT }
                )
                lineY += 36f
            }
        }

        // ── Correction hint ───────────────────────────────────────────────
        val fixText = expl?.topCorrection ?: ""
        if (fixText.isNotBlank()) {
            lineY += 6f
            val hintBg = Paint().apply { color = Color.parseColor("#1A2F3D"); style = Paint.Style.FILL }
            canvas.drawRoundRect(12f, lineY, w - 12f, lineY + 42f, 8f, 8f, hintBg)
            canvas.drawText("建议: $fixText", 24f, lineY + 28f, leftTextPaint(23f, Color.parseColor("#FDE68A")))
            lineY += 54f
        }

        // ── Future prediction line ────────────────────────────────────────
        if (futureLine.isNotBlank()) {
            canvas.drawText(futureLine, 16f, lineY + 20f, leftTextPaint(21f, Color.parseColor("#7DD3FC")))
        }
    }

    // ── Sub-draws ─────────────────────────────────────────────────────────

    private fun drawConfidenceArc(canvas: Canvas, cx: Float, cy: Float, r: Float, color: Int) {
        val rect = RectF(cx - r, cy - r, cx + r, cy + r)
        // Track (full circle background)
        canvas.drawArc(rect, 135f, 270f, false, arcTrackPaint())
        // Filled arc up to animatedArc
        val sweep = animatedArc * 270f
        canvas.drawArc(rect, 135f, sweep, false, arcFillPaint(color))
    }

    private fun drawRiskBadge(
        canvas: Canvas, label: Int, color: Int,
        x: Float, y: Float, maxW: Float
    ) {
        val emoji = when (label) { 0 -> "🐱"; 1 -> "😿"; else -> "🙀" }
        val text  = when (label) { 0 -> "低风险"; 1 -> "中风险"; else -> "高风险" }

        // Badge background pill
        val bh = 56f; val bw = minOf(maxW, 200f)
        val bp = Paint().apply { this.color = Color.argb(50, Color.red(color), Color.green(color), Color.blue(color)); style = Paint.Style.FILL }
        canvas.drawRoundRect(x, y, x + bw, y + bh, bh / 2, bh / 2, bp)

        // Border
        val border = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; style = Paint.Style.STROKE; strokeWidth = 2f
        }
        canvas.drawRoundRect(x, y, x + bw, y + bh, bh / 2, bh / 2, border)

        // Text
        val tp = textPaint(30f, color, bold = true)
        canvas.drawText("$emoji $text", x + bw / 2, y + 38f, tp)
    }

    private fun drawWaiting(canvas: Canvas, w: Float, h: Float) {
        canvas.drawText("🐱 等待检测…", w / 2f, h / 2f, textPaint(32f, COL_SUB))
        canvas.drawText("需要 20 帧后开始推理", w / 2f, h / 2f + 40f, textPaint(24f, COL_SUB))
    }

    // ── Arc animation ─────────────────────────────────────────────────────

    private fun animateArc(target: Float) {
        val step = (target - animatedArc) * 0.25f
        if (kotlin.math.abs(step) < 0.002f) { animatedArc = target; return }
        animatedArc += step
        invalidate()
        handler.postDelayed({ animateArc(target) }, 16)
    }

    private fun riskColor() = when (riskLabel) {
        0 -> COL_LOW; 1 -> COL_MEDIUM; else -> COL_HIGH
    }
}
