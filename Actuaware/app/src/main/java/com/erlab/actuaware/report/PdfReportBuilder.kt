package com.healthai.ankle.report

import android.content.Context
import android.graphics.*
import android.graphics.pdf.PdfDocument
import android.util.Log
import com.healthai.ankle.db.FrameEntity
import com.healthai.ankle.db.SessionEntity
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.max
import kotlin.math.min

/**
 * Generates an A4 PDF report (595×842 pts) for a rehabilitation session.
 */
object PdfReportBuilder {

    private const val A4_W = 595
    private const val A4_H = 842
    private const val MARGIN = 40f
    private const val CHART_H = 80f

    // Palette
    private val COLOR_ORANGE  = Color.parseColor("#F97316")
    private val COLOR_CREAM   = Color.parseColor("#FFF3E0")
    private val COLOR_RED     = Color.parseColor("#EF4444")
    private val COLOR_YELLOW  = Color.parseColor("#F59E0B")
    private val COLOR_GREEN   = Color.parseColor("#22C55E")
    private val COLOR_DARK    = Color.parseColor("#1C1917")
    private val COLOR_GRAY    = Color.parseColor("#78716C")

    fun build(
        context: Context,
        session: SessionEntity,
        frames: List<FrameEntity>,
        outputFile: File
    ): File {
        val doc = PdfDocument()
        val pageInfo = PdfDocument.PageInfo.Builder(A4_W, A4_H, 1).create()
        val page = doc.startPage(pageInfo)
        val canvas = page.canvas

        var y = drawHeader(canvas, session)
        y = drawScoreSummary(canvas, session, frames, y)
        y = drawRiskCurve(canvas, frames, y)
        y = drawKneeCurves(canvas, frames, y)
        y = drawGrfCurve(canvas, frames, y)
        drawSuggestions(canvas, session, frames, y)

        doc.finishPage(page)

        outputFile.parentFile?.mkdirs()
        FileOutputStream(outputFile).use { doc.writeTo(it) }
        doc.close()
        Log.i("PdfReport", "Saved to ${outputFile.absolutePath}")
        return outputFile
    }

    private fun drawHeader(canvas: Canvas, session: SessionEntity): Float {
        val bgPaint = Paint().apply { color = COLOR_ORANGE; style = Paint.Style.FILL }
        canvas.drawRect(0f, 0f, A4_W.toFloat(), 80f, bgPaint)

        val titlePaint = textPaint(24f, Color.WHITE, bold = true)
        canvas.drawText("RehabGuardian 康复报告", MARGIN, 48f, titlePaint)

        val datePaint = textPaint(10f, Color.WHITE)
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
        canvas.drawText("生成时间: ${fmt.format(Date(session.startTimeMs))}", MARGIN, 68f, datePaint)

        return 100f
    }

    private fun drawScoreSummary(canvas: Canvas, session: SessionEntity, frames: List<FrameEntity>, startY: Float): Float {
        var y = startY
        sectionTitle(canvas, "综合评分", y); y += 28f

        val boxes = listOf(
            Triple("运动评分", "${session.avgMotionScore.toInt()}/100", COLOR_ORANGE),
            Triple("最高风险", riskText(session.maxRiskLabel), riskColor(session.maxRiskLabel)),
            Triple("分析帧数", "${session.totalFrames}", COLOR_GRAY),
            Triple("体重(kg)", "${session.userWeightKg.toInt()}", COLOR_GRAY)
        )
        val bw = (A4_W - MARGIN * 2 - 18f) / 4f
        boxes.forEachIndexed { i, (label, value, color) ->
            val bx = MARGIN + i * (bw + 6f)
            val boxPaint = Paint().apply { this.color = COLOR_CREAM; style = Paint.Style.FILL }
            val accentPaint = Paint().apply { this.color = color; style = Paint.Style.FILL }
            canvas.drawRoundRect(RectF(bx, y, bx + bw, y + 60f), 8f, 8f, boxPaint)
            canvas.drawRoundRect(RectF(bx, y, bx + 4f, y + 60f), 4f, 4f, accentPaint)
            canvas.drawText(label, bx + 10f, y + 22f, textPaint(9f, COLOR_GRAY))
            canvas.drawText(value, bx + 10f, y + 48f, textPaint(18f, COLOR_DARK, bold = true))
        }
        return y + 80f
    }

    private fun drawRiskCurve(canvas: Canvas, frames: List<FrameEntity>, startY: Float): Float {
        if (frames.isEmpty()) return startY + 20f
        var y = startY
        sectionTitle(canvas, "风险曲线 (高风险概率)", y); y += 24f
        drawLineChart(
            canvas, frames.map { it.riskScore }, y, COLOR_RED,
            yMin = 0f, yMax = 1f
        )
        return y + CHART_H + 16f
    }

    private fun drawKneeCurves(canvas: Canvas, frames: List<FrameEntity>, startY: Float): Float {
        if (frames.isEmpty()) return startY + 20f
        var y = startY
        sectionTitle(canvas, "膝关节角度 (°)", y); y += 24f
        drawLineChart(canvas, frames.map { it.lKneeDeg }, y, COLOR_GREEN, label = "左膝",
            yMin = 90f, yMax = 180f)
        drawLineChart(canvas, frames.map { it.rKneeDeg }, y, COLOR_ORANGE, label = "右膝",
            yMin = 90f, yMax = 180f, alpha = 180)
        drawLegend(canvas, y + CHART_H - 16f,
            listOf("左膝" to COLOR_GREEN, "右膝" to COLOR_ORANGE))
        return y + CHART_H + 16f
    }

    private fun drawGrfCurve(canvas: Canvas, frames: List<FrameEntity>, startY: Float): Float {
        if (frames.isEmpty()) return startY + 20f
        var y = startY
        sectionTitle(canvas, "地面反作用力 Fz (BW)", y); y += 24f
        drawLineChart(canvas, frames.map { it.grfLeftFz },  y, COLOR_GREEN, label = "左脚",
            yMin = -0.5f, yMax = 3f)
        drawLineChart(canvas, frames.map { it.grfRightFz }, y, COLOR_ORANGE, label = "右脚",
            yMin = -0.5f, yMax = 3f, alpha = 180)
        drawLegend(canvas, y + CHART_H - 16f,
            listOf("左脚" to COLOR_GREEN, "右脚" to COLOR_ORANGE))
        return y + CHART_H + 16f
    }

    private fun drawSuggestions(canvas: Canvas, session: SessionEntity, frames: List<FrameEntity>, startY: Float) {
        if (startY > A4_H - 60f) return
        var y = startY
        sectionTitle(canvas, "改善建议", y); y += 24f

        val suggestions = buildSuggestions(session, frames)
        for (s in suggestions) {
            if (y > A4_H - 30f) break
            canvas.drawText("• $s", MARGIN + 8f, y, textPaint(10f, COLOR_DARK))
            y += 18f
        }
    }

    // ── chart primitives ──────────────────────────────────────────────────

    private fun drawLineChart(
        canvas: Canvas, values: List<Float>, y: Float, color: Int,
        label: String = "", yMin: Float = 0f, yMax: Float = 1f, alpha: Int = 255
    ) {
        if (values.size < 2) return
        val chartW = A4_W - MARGIN * 2
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color; this.alpha = alpha
            style = Paint.Style.STROKE; strokeWidth = 2f
        }
        val bg = Paint().apply { this.color = COLOR_CREAM; style = Paint.Style.FILL }
        canvas.drawRect(MARGIN, y, MARGIN + chartW, y + CHART_H, bg)

        val path = Path()
        values.forEachIndexed { i, v ->
            val px = MARGIN + i.toFloat() / (values.size - 1) * chartW
            val norm = ((v - yMin) / (yMax - yMin)).coerceIn(0f, 1f)
            val py = y + CHART_H - norm * CHART_H
            if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        canvas.drawPath(path, paint)

        // Axis lines
        val axisPaint = Paint().apply { this.color = COLOR_GRAY; strokeWidth = 0.5f }
        canvas.drawLine(MARGIN, y, MARGIN, y + CHART_H, axisPaint)
        canvas.drawLine(MARGIN, y + CHART_H, MARGIN + chartW, y + CHART_H, axisPaint)
    }

    private fun drawLegend(canvas: Canvas, y: Float, items: List<Pair<String, Int>>) {
        var x = MARGIN + 8f
        for ((label, color) in items) {
            val lp = Paint().apply { this.color = color; style = Paint.Style.FILL }
            canvas.drawRect(x, y, x + 16f, y + 8f, lp)
            canvas.drawText(label, x + 20f, y + 8f, textPaint(9f, COLOR_DARK))
            x += 60f
        }
    }

    private fun sectionTitle(canvas: Canvas, title: String, y: Float) {
        canvas.drawText(title, MARGIN, y, textPaint(12f, COLOR_ORANGE, bold = true))
        val lp = Paint().apply { color = COLOR_ORANGE; strokeWidth = 1f }
        canvas.drawLine(MARGIN, y + 4f, A4_W - MARGIN, y + 4f, lp)
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private fun textPaint(size: Float, color: Int, bold: Boolean = false) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color = color
        textSize = size * 2.5f   // PdfDocument uses 72dpi units; scale for readability
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun riskText(label: Int) = when (label) { 0 -> "低" 1 -> "中" else -> "高" }
    private fun riskColor(label: Int) = when (label) { 0 -> COLOR_GREEN 1 -> COLOR_YELLOW else -> COLOR_RED }

    private fun buildSuggestions(session: SessionEntity, frames: List<FrameEntity>): List<String> {
        val s = mutableListOf<String>()
        val avgScore = session.avgMotionScore
        if (avgScore < 60f) s.add("运动质量偏低，建议降低训练强度，专注于膝关节控制。")
        val avgKneeAsym = frames.map { kotlin.math.abs(it.lKneeDeg - it.rKneeDeg) }.average().toFloat()
        if (avgKneeAsym > 15f) s.add("双膝对称性较差 (平均偏差 ${avgKneeAsym.toInt()}°)，建议加强单腿稳定性训练。")
        val highRiskRatio = frames.count { it.riskLabel == 2 }.toFloat() / frames.size.coerceAtLeast(1)
        if (highRiskRatio > 0.2f) s.add("高风险帧占比 ${(highRiskRatio * 100).toInt()}%，建议咨询康复师。")
        if (s.isEmpty()) s.add("运动表现良好，请保持当前训练节奏。")
        return s
    }
}