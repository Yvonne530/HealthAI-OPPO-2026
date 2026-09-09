package com.erlab.actuaware.ui

import android.graphics.*
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.erlab.actuaware.analysis.RiskStateMachine
import kotlin.math.*

class SkeletonRenderer {

    private val CONNECTIONS = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 7,
        0 to 4, 4 to 5, 5 to 6, 6 to 8,
        9 to 10, 11 to 12, 11 to 13, 13 to 15,
        12 to 14, 14 to 16, 11 to 23, 12 to 24,
        23 to 24, 23 to 25, 24 to 26, 25 to 27,
        26 to 28, 27 to 29, 28 to 30, 29 to 31,
        30 to 32,
    )

    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 4f; style = Paint.Style.STROKE }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val anglePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 36f; typeface = Typeface.DEFAULT_BOLD }
    private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3f; color = Color.CYAN }
    private val warningPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.RED; style = Paint.Style.FILL }
    private val grfPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val grfBgPaint = Paint().apply { color = Color.argb(120, 0, 0, 0); style = Paint.Style.FILL }

    private var animAlpha = 0f

    fun draw(
        canvas: Canvas,
        landmarks: List<NormalizedLandmark>,
        riskState: RiskStateMachine.State,
        riskScore: Float,
        width: Int,
        height: Int,
        kneeAngleL: Float = 180f,
        kneeAngleR: Float = 180f,
        grfLeft: FloatArray? = null,
        grfRight: FloatArray? = null
    ) {
        if (landmarks.size < 33) return

        val boneColor = riskStateColor(riskState, riskScore)
        bonePaint.color = boneColor
        jointPaint.color = boneColor

        for ((a, b) in CONNECTIONS) {
            if (a >= landmarks.size || b >= landmarks.size) continue
            val lmA = landmarks[a]; val lmB = landmarks[b]
            if (lmA.visibility().orElse(0f) < 0.5f || lmB.visibility().orElse(0f) < 0.5f) continue
            canvas.drawLine(lmA.x() * width, lmA.y() * height, lmB.x() * width, lmB.y() * height, bonePaint)
        }

        for (lm in landmarks) {
            if (lm.visibility().orElse(0f) < 0.5f) continue
            canvas.drawCircle(lm.x() * width, lm.y() * height, 6f, jointPaint)
        }

        drawKneeAngle(canvas, landmarks, width, height, kneeAngleL, 25, 23, 27, "L")
        drawKneeAngle(canvas, landmarks, width, height, kneeAngleR, 26, 24, 28, "R")

        if (riskState == RiskStateMachine.State.DANGER) {
            animAlpha = (animAlpha + 0.1f) % 1f
            val alpha = (sin(animAlpha * Math.PI) * 200).toInt().coerceIn(50, 200)
            warningPaint.alpha = alpha
            for (idx in listOf(25, 26)) {
                if (idx < landmarks.size) {
                    val lm = landmarks[idx]
                    canvas.drawCircle(lm.x() * width, lm.y() * height, 24f, warningPaint)
                }
            }
        }

        if (grfLeft != null && grfRight != null) {
            drawGRFBar(canvas, grfLeft, grfRight, width, height)
        }
    }

    private fun drawKneeAngle(
        canvas: Canvas,
        lms: List<NormalizedLandmark>,
        w: Int, h: Int,
        angle: Float,
        kneeIdx: Int, hipIdx: Int, ankleIdx: Int,
        side: String
    ) {
        if (kneeIdx >= lms.size || hipIdx >= lms.size || ankleIdx >= lms.size) return
        val knee = lms[kneeIdx]; val hip = lms[hipIdx]; val ankle = lms[ankleIdx]
        if (knee.visibility().orElse(0f) < 0.5f) return
        val kx = knee.x() * w; val ky = knee.y() * h
        val radius = 40f
        val angleColor = when {
            angle > 175f -> Color.RED
            angle > 150f -> Color.YELLOW
            else -> Color.CYAN
        }
        arcPaint.color = angleColor
        val rect = RectF(kx - radius, ky - radius, kx + radius, ky + radius)
        canvas.drawArc(rect, -90f, -(180f - angle), false, arcPaint)
        anglePaint.color = angleColor
        canvas.drawText("${angle.toInt()}°", kx + radius + 4f, ky + 12f, anglePaint)
        if (angle > 175f) {
            anglePaint.color = Color.RED
            canvas.drawText("⚠过伸", kx - 40f, ky - radius - 8f, anglePaint)
        }
    }

    private fun drawGRFBar(canvas: Canvas, grfLeft: FloatArray, grfRight: FloatArray, w: Int, h: Int) {
        val maxH = h * 0.3f
        val fzL = grfLeft.getOrElse(2) { 0f }.coerceIn(0f, 2000f)
        val fzR = grfRight.getOrElse(2) { 0f }.coerceIn(0f, 2000f)
        val bwN = 70f * 9.81f
        canvas.drawRect(w - 100f, h - maxH - 20f, w.toFloat(), h.toFloat(), grfBgPaint)
        val hL = (fzL / 2000f * maxH).coerceIn(0f, maxH)
        grfPaint.color = if (fzL > 1.5f * bwN) Color.RED else Color.GREEN
        canvas.drawRect(w - 90f, h - hL, w - 55f, h.toFloat(), grfPaint)
        val hR = (fzR / 2000f * maxH).coerceIn(0f, maxH)
        grfPaint.color = if (fzR > 1.5f * bwN) Color.RED else Color.GREEN
        canvas.drawRect(w - 45f, h - hR, w - 10f, h.toFloat(), grfPaint)
        anglePaint.color = Color.WHITE; anglePaint.textSize = 24f
        canvas.drawText("L R", w - 90f, h - maxH - 30f, anglePaint)
        canvas.drawText("${(fzL / bwN).format(1)}BW", w - 90f, h - hL - 6f, anglePaint)
        canvas.drawText("${(fzR / bwN).format(1)}BW", w - 45f, h - hR - 6f, anglePaint)
    }

    private fun riskStateColor(state: RiskStateMachine.State, score: Float): Int {
        return when (state) {
            RiskStateMachine.State.SAFE -> Color.GREEN
            RiskStateMachine.State.WARNING -> Color.parseColor("#FF9800")
            RiskStateMachine.State.DANGER -> Color.RED
        }
    }

    private fun Float.format(decimals: Int) = "%.${decimals}f".format(this)
}