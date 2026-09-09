package com.erlab.actuaware.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.erlab.actuaware.analysis.RiskStateMachine

class RiskOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val renderer = SkeletonRenderer()
    private var landmarks: List<NormalizedLandmark> = emptyList()
    private var riskState = RiskStateMachine.State.SAFE
    private var riskScore = 0f
    private var kneeAngleL = 180f
    private var kneeAngleR = 180f
    private var grfLeft: FloatArray? = null
    private var grfRight: FloatArray? = null

    private val infoPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 48f
        typeface = Typeface.DEFAULT_BOLD
        color = Color.WHITE
    }
    private val infoBgPaint = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val warningBannerPaint = Paint().apply {
        color = Color.argb(200, 220, 0, 0)
        style = Paint.Style.FILL
    }

    fun update(
        lms: List<NormalizedLandmark>,
        state: RiskStateMachine.State,
        score: Float,
        kneeL: Float,
        kneeR: Float,
        grfL: FloatArray? = null,
        grfR: FloatArray? = null
    ) {
        landmarks = lms
        riskState = state
        riskScore = score
        kneeAngleL = kneeL
        kneeAngleR = kneeR
        grfLeft = grfL
        grfRight = grfR
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width == 0 || height == 0) return

        renderer.draw(canvas, landmarks, riskState, riskScore, width, height,
            kneeAngleL, kneeAngleR, grfLeft, grfRight)
        drawStatusCard(canvas)
        if (riskState == RiskStateMachine.State.DANGER) {
            drawDangerBanner(canvas)
        }
    }

    private fun drawStatusCard(canvas: Canvas) {
        val cardRight = width - 20f
        val cardTop = 20f
        val cardW = 280f
        val cardH = 130f
        val rect = RectF(cardRight - cardW, cardTop, cardRight, cardTop + cardH)
        canvas.drawRoundRect(rect, 16f, 16f, infoBgPaint)

        val color = when (riskState) {
            RiskStateMachine.State.SAFE -> Color.GREEN
            RiskStateMachine.State.WARNING -> Color.parseColor("#FF9800")
            RiskStateMachine.State.DANGER -> Color.RED
        }
        infoPaint.color = color
        infoPaint.textSize = 36f
        canvas.drawText(riskState.label, cardRight - cardW + 16f, cardTop + 48f, infoPaint)

        infoPaint.color = Color.WHITE
        infoPaint.textSize = 28f
        canvas.drawText("Risk: ${"%.2f".format(riskScore)}", cardRight - cardW + 16f, cardTop + 88f, infoPaint)
        canvas.drawText("L:${kneeAngleL.toInt()}° R:${kneeAngleR.toInt()}°", cardRight - cardW + 16f, cardTop + 120f, infoPaint)
    }

    private fun drawDangerBanner(canvas: Canvas) {
        val bannerH = 80f
        canvas.drawRect(0f, height - bannerH, width.toFloat(), height.toFloat(), warningBannerPaint)
        infoPaint.color = Color.WHITE
        infoPaint.textSize = 38f
        canvas.drawText("⚠ 检测到高风险动作", 30f, height - bannerH / 4, infoPaint)
    }
}