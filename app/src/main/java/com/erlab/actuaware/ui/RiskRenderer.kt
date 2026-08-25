package com.rehabguardian.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.*

class RiskRenderer @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val CONNECTIONS = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 7,
        0 to 4, 4 to 5, 5 to 6, 6 to 8,
        9 to 10, 11 to 12, 11 to 13, 13 to 15,
        12 to 14, 14 to 16, 11 to 23, 12 to 24,
        23 to 24, 23 to 25, 24 to 26, 25 to 27,
        26 to 28, 27 to 29, 28 to 30, 29 to 31,
        30 to 32,
    )

    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val anglePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.YELLOW
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 36f
        typeface = Typeface.DEFAULT_BOLD
        color = Color.WHITE
    }
    private val warningPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 48f
        typeface = Typeface.DEFAULT_BOLD
        color = Color.RED
    }
    private val overlayPaint = Paint().apply {
        color = Color.argb(120, 255, 0, 0)
    }
    private val riskBarPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val latencyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 24f
        color = Color.WHITE
        typeface = Typeface.DEFAULT_BOLD
    }

    private var currentResult: PoseLandmarkerResult? = null
    private var riskScore = 0f
    private var riskLabel = 0
    private var kneeAngleL = 160f
    private var kneeAngleR = 160f
    private var latencyMs = 0f
    private var imageWidth = 640
    private var imageHeight = 480
    private var scaleX = 1f
    private var scaleY = 1f

    fun update(
        result: PoseLandmarkerResult,
        risk: Float,
        label: Int,
        kneL: Float,
        kneR: Float,
        imgW: Int,
        imgH: Int,
        latency: Float = 0f
    ) {
        currentResult = result
        riskScore = risk
        riskLabel = label
        kneeAngleL = kneL
        kneeAngleR = kneR
        imageWidth = imgW
        imageHeight = imgH
        latencyMs = latency
        scaleX = width.toFloat() / imageWidth
        scaleY = height.toFloat() / imageHeight
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val result = currentResult ?: return
        val landmarks = result.landmarks()
        if (landmarks.isEmpty()) return
        val lms = landmarks[0]

        fun lmX(i: Int) = (lms.getOrNull(i)?.x() ?: 0f) * width
        fun lmY(i: Int) = (lms.getOrNull(i)?.y() ?: 0f) * height
        fun visible(i: Int) = (lms.getOrNull(i)?.visibility() ?: 0f) > 0.5f

        val boneColor = when (riskLabel) {
            0 -> Color.rgb(76, 175, 80)
            1 -> Color.rgb(255, 193, 7)
            else -> Color.rgb(244, 67, 54)
        }
        bonePaint.color = boneColor
        jointPaint.color = boneColor

        // 骨骼连线
        CONNECTIONS.forEach { (a, b) ->
            if (visible(a) && visible(b)) {
                canvas.drawLine(lmX(a), lmY(a), lmX(b), lmY(b), bonePaint)
            }
        }

        // 关节点
        for (i in 0 until minOf(lms.size, 33)) {
            if (visible(i)) {
                val r = if (i in listOf(11, 12, 13, 14, 23, 24, 25, 26)) 10f else 6f
                canvas.drawCircle(lmX(i), lmY(i), r, jointPaint)
            }
        }

        // 膝关节角度
        drawKneeAngle(canvas, lmX(23), lmY(23), lmX(25), lmY(25), lmX(27), lmY(27), kneeAngleL, "L")
        drawKneeAngle(canvas, lmX(24), lmY(24), lmX(26), lmY(26), lmX(28), lmY(28), kneeAngleR, "R")

        drawRiskBar(canvas, riskScore, riskLabel)
        drawLatency(canvas, latencyMs)

        if (riskLabel == 2) {
            canvas.drawRect(0f, 0f, width.toFloat(), 80f, overlayPaint)
            canvas.drawText("⚠ 检测到损伤风险", 20f, 60f, warningPaint)
        }
    }

    private fun drawKneeAngle(
        canvas: Canvas,
        hipX: Float, hipY: Float,
        kneeX: Float, kneeY: Float,
        ankleX: Float, ankleY: Float,
        angleDeg: Float,
        side: String,
    ) {
        val r = 40f
        val startAngle = atan2(hipY - kneeY, hipX - kneeX)
        val sweepAngle = Math.toRadians(180.0 - angleDeg).toFloat()
        val rectF = RectF(kneeX - r, kneeY - r, kneeX + r, kneeY + r)
        val color = if (angleDeg > 175f || angleDeg < 90f) Color.RED else Color.YELLOW
        anglePaint.color = color
        canvas.drawArc(rectF, Math.toDegrees(startAngle.toDouble()).toFloat(),
            Math.toDegrees(sweepAngle.toDouble()).toFloat(), false, anglePaint)

        textPaint.color = color
        canvas.drawText("$side: ${angleDeg.toInt()}°", kneeX + 45f, kneeY, textPaint)

        if (angleDeg > 175f) {
            canvas.drawCircle(kneeX, kneeY, 20f, Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 6f
            })
        }
    }

    private fun drawRiskBar(canvas: Canvas, score: Float, label: Int) {
        val barWidth = 200f
        val barHeight = 30f
        val barX = 20f
        val barY = height - 80f

        val bgPaint = Paint().apply { color = Color.argb(100, 0, 0, 0) }
        canvas.drawRect(barX, barY, barX + barWidth, barY + barHeight, bgPaint)

        val progress = (score * barWidth).coerceIn(0f, barWidth)
        riskBarPaint.color = when (label) {
            0 -> Color.GREEN
            1 -> Color.parseColor("#FF9800")
            else -> Color.RED
        }
        canvas.drawRect(barX, barY, barX + progress, barY + barHeight, riskBarPaint)

        textPaint.textSize = 24f
        textPaint.color = Color.WHITE
        canvas.drawText("Risk: ${"%.2f".format(score)}", barX + barWidth + 10f, barY + 22f, textPaint)
    }

    private fun drawLatency(canvas: Canvas, latency: Float) {
        val color = when {
            latency < 20 -> Color.GREEN
            latency < 50 -> Color.YELLOW
            else -> Color.RED
        }
        latencyPaint.color = color
        canvas.drawText("Latency: ${"%.1f".format(latency)}ms", width - 150f, 40f, latencyPaint)
    }
}