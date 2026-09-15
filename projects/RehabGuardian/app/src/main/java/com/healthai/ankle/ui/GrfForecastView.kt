package com.healthai.ankle.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import java.util.ArrayDeque

/**
 * Dual-channel GRF (Ground Reaction Force) chart with:
 *   - Left / Right Fz history (rolling 60-frame buffer)
 *   - Future prediction overlay from FNO output (10 frames)
 *   - 2.5 BW danger threshold line
 *   - "预测未来" label — the "国一级叙事" visual anchor
 *
 * Layout: history region (left ~70%) | prediction region (right ~30%)
 */
class GrfForecastView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    companion object {
        private const val HISTORY_SIZE     = 60
        private const val DANGER_THRESHOLD = 2.5f
        private const val Y_MAX            = 4f
        private const val Y_MIN            = -0.3f
        // Colors
        private val COL_LEFT_HIST  = Color.parseColor("#22C55E")   // green
        private val COL_RIGHT_HIST = Color.parseColor("#F97316")   // orange
        private val COL_LEFT_PRED  = Color.parseColor("#86EFAC")   // light green
        private val COL_RIGHT_PRED = Color.parseColor("#FDBA74")   // light orange
        private val COL_DANGER     = Color.parseColor("#EF4444")
        private val COL_BG         = Color.parseColor("#0D1B1E")   // deep dark bg
        private val COL_GRID       = Color.parseColor("#1E3A3A")
        private val COL_TEXT       = Color.parseColor("#94A3B8")
        private val COL_DIVIDER    = Color.parseColor("#334155")
    }

    // History buffers
    private val leftHistory  = ArrayDeque<Float>(HISTORY_SIZE)
    private val rightHistory = ArrayDeque<Float>(HISTORY_SIZE)

    // Future prediction (10 frames)
    private var leftFuture  = FloatArray(10)
    private var rightFuture = FloatArray(10)

    // ── Paints ────────────────────────────────────────────────────────────

    private val bgPaint = Paint().apply { color = COL_BG; style = Paint.Style.FILL }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = COL_GRID; style = Paint.Style.STROKE; strokeWidth = 1f
    }
    private val dangerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = COL_DANGER; style = Paint.Style.STROKE; strokeWidth = 1.5f
        pathEffect = DashPathEffect(floatArrayOf(8f, 4f), 0f)
    }
    private val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = COL_DIVIDER; style = Paint.Style.STROKE; strokeWidth = 1.5f
        pathEffect = DashPathEffect(floatArrayOf(4f, 4f), 0f)
    }
    private fun linePaint(color: Int, alpha: Int = 255, width: Float = 2f) =
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; this.alpha = alpha
            style = Paint.Style.STROKE; strokeWidth = width
            strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
        }
    private fun fillPaint(color: Int, alpha: Int = 60) =
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; this.alpha = alpha
            style = Paint.Style.FILL
        }
    private fun labelPaint(color: Int = COL_TEXT, size: Float = 24f) =
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; textSize = size
        }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Push one frame of GRF data.
     * @param leftFz   Left foot vertical GRF in body-weight units
     * @param rightFz  Right foot vertical GRF in body-weight units
     * @param future   FNO prediction flat array [10*12]; Fz at [t*12+0] and [t*12+6]
     */
    fun push(leftFz: Float, rightFz: Float, future: FloatArray) {
        if (leftHistory.size  >= HISTORY_SIZE) leftHistory.pollFirst()
        if (rightHistory.size >= HISTORY_SIZE) rightHistory.pollFirst()
        leftHistory.addLast(leftFz.coerceIn(Y_MIN, Y_MAX))
        rightHistory.addLast(rightFz.coerceIn(Y_MIN, Y_MAX))

        // Extract Fz channels from future flat [10*12]
        val nFuture = minOf(10, future.size / 12)
        leftFuture  = FloatArray(nFuture) { t -> future.getOrElse(t * 12 + 0) { 0f }.coerceIn(Y_MIN, Y_MAX) }
        rightFuture = FloatArray(nFuture) { t -> future.getOrElse(t * 12 + 6) { 0f }.coerceIn(Y_MIN, Y_MAX) }

        invalidate()
    }

    fun clear() {
        leftHistory.clear(); rightHistory.clear()
        leftFuture = FloatArray(10); rightFuture = FloatArray(10)
        invalidate()
    }

    // ── Draw ──────────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        val pad  = 6f
        val padL = 40f; val padR = pad; val padT = 22f; val padB = 28f
        val chartW = w - padL - padR
        val chartH = h - padT - padB

        // Split: 70% history | 30% future
        val histW = chartW * 0.70f
        val predW = chartW * 0.30f
        val divX  = padL + histW

        // Background
        canvas.drawRoundRect(0f, 0f, w, h, 12f, 12f, bgPaint)

        // Grid lines at 0, 1, 2, 3 BW
        val gridLevels = listOf(0f, 1f, 2f, 2.5f, 3f)
        for (level in gridLevels) {
            val y = yPx(level, padT, chartH)
            val gp = if (level == DANGER_THRESHOLD) dangerPaint else gridPaint
            canvas.drawLine(padL, y, w - padR, y, gp)
            canvas.drawText("${level.toInt()}",  4f, y + 7f, labelPaint(size = 22f))
        }

        // Y-axis label
        canvas.save()
        canvas.rotate(-90f, 12f, h / 2)
        canvas.drawText("GRF (BW)", -h / 2 + 10f, 22f, labelPaint(size = 22f))
        canvas.restore()

        // Divider between history and prediction
        canvas.drawLine(divX, padT, divX, padT + chartH, dividerPaint)

        // "预测" label
        canvas.drawText("← 历史", padL + 4f, padT - 4f, labelPaint(COL_TEXT, 20f))
        canvas.drawText("预测 →", divX + 4f, padT - 4f, labelPaint(COL_RIGHT_PRED, 20f))

        // ── Draw history ─────────────────────────────────────────────────
        drawSeriesHistory(canvas, leftHistory,  padL, padT, histW, chartH, COL_LEFT_HIST)
        drawSeriesHistory(canvas, rightHistory, padL, padT, histW, chartH, COL_RIGHT_HIST)

        // ── Draw future prediction ────────────────────────────────────────
        drawSeriesFuture(canvas, leftFuture,  divX, padT, predW, chartH, COL_LEFT_PRED)
        drawSeriesFuture(canvas, rightFuture, divX, padT, predW, chartH, COL_RIGHT_PRED)

        // ── Legend ───────────────────────────────────────────────────────
        val legendY = padT + chartH + 18f
        canvas.drawLine(padL, legendY, padL + 20f, legendY, linePaint(COL_LEFT_HIST))
        canvas.drawText("左脚", padL + 24f, legendY + 6f, labelPaint(COL_LEFT_HIST, 22f))
        canvas.drawLine(padL + 60f, legendY, padL + 80f, legendY, linePaint(COL_RIGHT_HIST))
        canvas.drawText("右脚", padL + 84f, legendY + 6f, labelPaint(COL_RIGHT_HIST, 22f))
        canvas.drawText("危险线", w - 80f, legendY + 6f, labelPaint(COL_DANGER, 20f))
    }

    private fun drawSeriesHistory(
        canvas: Canvas, data: ArrayDeque<Float>,
        originX: Float, originY: Float, areaW: Float, areaH: Float, color: Int
    ) {
        val pts = data.toFloatArray()
        if (pts.size < 2) return
        val path = Path(); val fill = Path()
        pts.forEachIndexed { i, v ->
            val x = originX + i.toFloat() / (pts.size - 1) * areaW
            val y = yPx(v, originY, areaH)
            if (i == 0) { path.moveTo(x, y); fill.moveTo(x, yPx(0f, originY, areaH)) }
            path.lineTo(x, y); fill.lineTo(x, y)
        }
        fill.lineTo(originX + areaW, yPx(0f, originY, areaH))
        fill.lineTo(originX, yPx(0f, originY, areaH)); fill.close()
        canvas.drawPath(fill, fillPaint(color, 40))
        canvas.drawPath(path, linePaint(color, 220, 2.5f))
    }

    private fun drawSeriesFuture(
        canvas: Canvas, data: FloatArray,
        originX: Float, originY: Float, areaW: Float, areaH: Float, color: Int
    ) {
        if (data.size < 2) return
        val path = Path()
        data.forEachIndexed { i, v ->
            val x = originX + i.toFloat() / (data.size - 1) * areaW
            val y = yPx(v, originY, areaH)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        canvas.drawPath(path, linePaint(color, 200, 2f).also {
            it.pathEffect = DashPathEffect(floatArrayOf(6f, 3f), 0f)
        })
    }

    /** Convert a value in [Y_MIN, Y_MAX] to pixel Y coordinate */
    private fun yPx(v: Float, originY: Float, areaH: Float): Float {
        val norm = (v - Y_MIN) / (Y_MAX - Y_MIN)
        return originY + areaH * (1f - norm)
    }
}
