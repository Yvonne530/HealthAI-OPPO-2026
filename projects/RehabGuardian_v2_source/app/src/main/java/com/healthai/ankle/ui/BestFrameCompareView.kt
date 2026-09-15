package com.healthai.ankle.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Side-by-side skeleton comparison:
 *   LEFT  = current frame landmarks
 *   RIGHT = best-recorded frame landmarks (lowest risk)
 *
 * This is the "冠军级功能" — shows the user exactly what they need to fix.
 * Each skeleton drawn as joints + bones with color-coded knee joints.
 */
class BestFrameCompareView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    // MediaPipe 33-point skeleton connections (anatomically meaningful subset)
    private val BODY_CONNECTIONS = listOf(
        11 to 12,  // shoulders
        11 to 13, 13 to 15,  // left arm
        12 to 14, 14 to 16,  // right arm
        11 to 23, 12 to 24, 23 to 24,  // torso
        23 to 25, 25 to 27, 27 to 29, 29 to 31,  // left leg
        24 to 26, 26 to 28, 28 to 30, 30 to 32   // right leg
    )

    data class SkeletonData(
        val landmarks: Array<FloatArray>,   // [33][3]
        val lKneeDeg: Float,
        val rKneeDeg: Float,
        val motionScore: Float,
        val label: String
    )

    private var current: SkeletonData? = null
    private var best: SkeletonData?    = null

    // Paints
    private val bgPaint = Paint().apply { color = Color.parseColor("#0D1B1E"); style = Paint.Style.FILL }
    private val dividerPaint = Paint().apply {
        color = Color.parseColor("#334155"); strokeWidth = 1.5f; style = Paint.Style.STROKE
        pathEffect = DashPathEffect(floatArrayOf(6f, 4f), 0f)
    }
    private fun bonePaint(color: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color = color; style = Paint.Style.STROKE; strokeWidth = 3f
        strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
    }
    private fun jointPaint(color: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color = color; style = Paint.Style.FILL
    }
    private fun labelPaint(color: Int = Color.parseColor("#94A3B8"), bold: Boolean = false) =
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; textSize = 28f
            if (bold) typeface = Typeface.DEFAULT_BOLD
        }

    // Colors
    private val COL_CURRENT_BONE = Color.parseColor("#F97316")    // orange
    private val COL_BEST_BONE    = Color.parseColor("#22C55E")    // green
    private val COL_JOINT        = Color.parseColor("#FFF3E0")
    private val COL_KNEE_BAD     = Color.parseColor("#EF4444")
    private val COL_KNEE_GOOD    = Color.parseColor("#22C55E")
    private val COL_KNEE_WARN    = Color.parseColor("#F59E0B")

    // ── Public API ────────────────────────────────────────────────────────

    fun updateCurrent(data: SkeletonData) { current = data; invalidate() }
    fun updateBest(data: SkeletonData)    { best = data;    invalidate() }

    fun clear() { current = null; best = null; invalidate() }

    // ── Draw ──────────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        canvas.drawRoundRect(0f, 0f, w, h, 16f, 16f, bgPaint)

        val midX = w / 2f
        canvas.drawLine(midX, 8f, midX, h - 8f, dividerPaint)

        // Left panel: current
        current?.let { drawPanel(canvas, it, 0f, w / 2f, h, COL_CURRENT_BONE, isLeft = true) }
            ?: drawEmpty(canvas, "当前动作", 0f, w / 2f, h)

        // Right panel: best
        best?.let { drawPanel(canvas, it, w / 2f, w, h, COL_BEST_BONE, isLeft = false) }
            ?: drawEmpty(canvas, "最佳动作\n（训练后显示）", w / 2f, w, h)
    }

    private fun drawPanel(
        canvas: Canvas, data: SkeletonData,
        x0: Float, x1: Float, h: Float, boneColor: Int, isLeft: Boolean
    ) {
        val panelW = x1 - x0
        val padTop = 40f; val padBot = 46f
        val skelH  = h - padTop - padBot
        val skelW  = panelW * 0.85f
        val skelX0 = x0 + (panelW - skelW) / 2f

        val lm = data.landmarks
        fun px(i: Int): Float = skelX0 + lm[i][0] * skelW
        fun py(i: Int): Float = padTop + lm[i][1] * skelH

        // Bones
        val bp = bonePaint(boneColor)
        for ((a, b) in BODY_CONNECTIONS) {
            if (a < lm.size && b < lm.size)
                canvas.drawLine(px(a), py(a), px(b), py(b), bp)
        }

        // Joints — highlight knees
        val kneeIndices = setOf(25, 26)
        for (i in lm.indices) {
            val jColor = when {
                i == 25 -> kneeColor(data.lKneeDeg)   // left knee
                i == 26 -> kneeColor(data.rKneeDeg)   // right knee
                i in kneeIndices -> COL_KNEE_WARN
                else -> COL_JOINT
            }
            val radius = if (i == 25 || i == 26) 9f else 5f
            canvas.drawCircle(px(i), py(i), radius, jointPaint(jColor))
        }

        // Knee angle labels next to knee joints
        if (25 < lm.size) {
            canvas.drawText("L:${data.lKneeDeg.toInt()}°",
                px(25) + 10f, py(25), labelPaint(kneeColor(data.lKneeDeg), bold = true))
        }
        if (26 < lm.size) {
            canvas.drawText("R:${data.rKneeDeg.toInt()}°",
                px(26) - 56f, py(26), labelPaint(kneeColor(data.rKneeDeg), bold = true))
        }

        // Panel header
        val headerColor = if (isLeft) COL_CURRENT_BONE else COL_BEST_BONE
        canvas.drawText(data.label, x0 + 8f, 28f, labelPaint(headerColor, bold = true))

        // Motion score at bottom
        val scoreColor = when {
            data.motionScore >= 80 -> COL_KNEE_GOOD
            data.motionScore >= 60 -> COL_KNEE_WARN
            else -> COL_KNEE_BAD
        }
        canvas.drawText(
            "评分 ${data.motionScore.toInt()}",
            x0 + 8f, h - 8f, labelPaint(scoreColor, bold = true)
        )
    }

    private fun drawEmpty(canvas: Canvas, msg: String, x0: Float, x1: Float, h: Float) {
        val p = labelPaint(Color.parseColor("#475569"))
        p.textAlign = Paint.Align.CENTER
        val lines = msg.split("\n")
        lines.forEachIndexed { i, line ->
            canvas.drawText(line, (x0 + x1) / 2f, h / 2f + i * 34f, p)
        }
    }

    private fun kneeColor(deg: Float) = when {
        deg >= 150f -> COL_KNEE_GOOD
        deg >= 130f -> COL_KNEE_WARN
        else -> COL_KNEE_BAD
    }
}
