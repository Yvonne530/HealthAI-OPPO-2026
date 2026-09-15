package com.healthai.ankle.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import java.util.ArrayDeque

/**
 * Real-time risk sparkline chart.
 * Maintains a rolling window of up to HISTORY_SIZE risk scores and draws
 * an orange anti-aliased line on a cream background with a gradient fill.
 */
class RiskChartView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    companion object {
        private const val HISTORY_SIZE = 60   // ~6 seconds at 10 fps
    }

    private val history = ArrayDeque<Float>(HISTORY_SIZE)

    // ── Paints ────────────────────────────────────────────────────────────

    private val bgPaint = Paint().apply {
        color = Color.parseColor("#FFF7ED")
        style = Paint.Style.FILL
    }

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F97316")
        style = Paint.Style.STROKE
        strokeWidth = 3f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FED7AA")
        style = Paint.Style.STROKE
        strokeWidth = 1f
        pathEffect = DashPathEffect(floatArrayOf(6f, 6f), 0f)
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#78716C")
        textSize = 22f   // in px; ~9sp
    }

    private val thresholdPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#EF4444")
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
        pathEffect = DashPathEffect(floatArrayOf(8f, 4f), 0f)
    }

    // ── Data ──────────────────────────────────────────────────────────────

    fun push(riskScore: Float) {
        if (history.size >= HISTORY_SIZE) history.pollFirst()
        history.addLast(riskScore.coerceIn(0f, 1f))
        invalidate()
    }

    fun clear() {
        history.clear()
        invalidate()
    }

    // ── Draw ──────────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        val padL = 36f; val padR = 8f; val padT = 8f; val padB = 20f
        val chartW = w - padL - padR
        val chartH = h - padT - padB

        // Background
        canvas.drawRoundRect(0f, 0f, w, h, 16f, 16f, bgPaint)

        // Grid lines at 0.25, 0.50, 0.75
        for (level in listOf(0.25f, 0.5f, 0.75f)) {
            val y = padT + chartH * (1f - level)
            canvas.drawLine(padL, y, w - padR, y, gridPaint)
            canvas.drawText("${(level * 100).toInt()}%", 2f, y + 6f, labelPaint)
        }

        // High-risk threshold at 0.70 (red dashed)
        val thresholdY = padT + chartH * 0.30f
        canvas.drawLine(padL, thresholdY, w - padR, thresholdY, thresholdPaint)

        if (history.size < 2) return

        val pts = history.toFloatArray()
        val n = pts.size

        // Build path
        val linePath = Path()
        val fillPath = Path()
        pts.forEachIndexed { i, v ->
            val x = padL + i.toFloat() / (n - 1) * chartW
            val y = padT + chartH * (1f - v)
            if (i == 0) { linePath.moveTo(x, y); fillPath.moveTo(x, padT + chartH) }
            else linePath.lineTo(x, y)
            if (i == n - 1) {
                fillPath.lineTo(x, y)   // tip of data
            } else {
                fillPath.lineTo(x, y)
            }
        }
        // Close fill path to bottom
        val lastX = padL + chartW
        fillPath.lineTo(lastX, padT + chartH)
        fillPath.lineTo(padL, padT + chartH)
        fillPath.close()

        // Gradient fill (orange → transparent)
        fillPaint.shader = LinearGradient(
            0f, padT, 0f, padT + chartH,
            Color.parseColor("#60F97316"), Color.parseColor("#00F97316"),
            Shader.TileMode.CLAMP
        )
        canvas.drawPath(fillPath, fillPaint)
        canvas.drawPath(linePath, linePaint)
    }
}
