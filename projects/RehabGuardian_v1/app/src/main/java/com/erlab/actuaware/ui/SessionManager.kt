package com.erlab.actuaware.ui

import android.content.Context
import com.erlab.actuaware.analysis.RiskThresholds.advice
import com.erlab.actuaware.analysis.RiskThresholds.riskLevel
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.erlab.actuaware.data.FrameData
import com.erlab.actuaware.data.HistoryRecord
import com.erlab.actuaware.data.Session
import java.text.SimpleDateFormat
import java.util.*

class SessionManager(private val context: Context) {

    private val prefs = context.getSharedPreferences("rg_history", Context.MODE_PRIVATE)
    private val gson = Gson()
    private val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())

    private var currentSession: Session? = null
    val isRecording: Boolean get() = currentSession != null

    private var fps = 30f
    private val replayBufferSize: Int get() = (fps * 10).toInt()
    private val replayBuffer = ArrayDeque<FrameData>()
    val replayFrames: List<FrameData> get() = replayBuffer.toList()

    fun setFps(actualFps: Float) {
        fps = actualFps.coerceIn(15f, 60f)
        while (replayBuffer.size > replayBufferSize) replayBuffer.removeFirst()
    }

    fun startSession() {
        currentSession = Session()
        replayBuffer.clear()
    }

    fun addFrame(frame: FrameData) {
        currentSession?.frames?.add(frame)
        if (replayBuffer.size >= replayBufferSize) replayBuffer.removeFirst()
        replayBuffer.addLast(frame)
    }

    fun stopSession(): Session? {
        val session = currentSession ?: return null
        session.endMs = System.currentTimeMillis()
        saveToHistory(session)
        currentSession = null
        return session
    }

    private fun saveToHistory(session: Session) {
        val records = loadHistory().toMutableList()
        records.add(0, HistoryRecord(
            sessionId = session.id,
            date = sdf.format(Date(session.startMs)),
            durationSec = session.durationSec,
            avgRisk = session.avgRisk,
            maxRisk = session.maxRisk,
            isHighRisk = session.isHighRisk,
            anomalies = session.topAnomalies(),
        ))
        prefs.edit().putString("history", gson.toJson(records.take(50))).apply()
    }

    fun loadHistory(): List<HistoryRecord> {
        val json = prefs.getString("history", "[]") ?: "[]"
        return gson.fromJson(json, object : TypeToken<List<HistoryRecord>>() {}.type)
    }

    enum class TrendAnalysis { Rising, Falling, Stable }

    fun getTrend(): TrendAnalysis {
        val frames = currentSession?.frames ?: return TrendAnalysis.Stable
        if (frames.size < 30) return TrendAnalysis.Stable
        val firstHalf = frames.take(frames.size / 2).map { it.riskScore }.average()
        val secondHalf = frames.takeLast(frames.size / 2).map { it.riskScore }.average()
        return when {
            secondHalf - firstHalf > 0.1 -> TrendAnalysis.Rising
            firstHalf - secondHalf > 0.1 -> TrendAnalysis.Falling
            else -> TrendAnalysis.Stable
        }
    }

    fun generateReport(session: Session): String {
        val riskText = session.riskLevelText
        val anomalies = session.topAnomalies().joinToString("\n • ", prefix = " • ")
        val trendText = when (getTrend()) {
            TrendAnalysis.Rising -> "⚠️ 风险呈上升趋势，请注意"
            TrendAnalysis.Falling -> "📉 风险呈下降趋势，继续保持"
            else -> "➡️ 风险水平稳定"
        }
        val advice = advice(session.avgRisk)
        return """
            🧾 ACL 损伤风险分析报告
            ━━━━━━━━━━━━━━━━━━━━━━━━
            📅 时间: ${sdf.format(Date(session.startMs))}
            ⏱ 时长: ${"%.1f".format(session.durationSec)} 秒
            📊 风险指标
            • 平均风险: ${"%.2f".format(session.avgRisk)} ($riskText)
            • 峰值风险: ${"%.2f".format(session.maxRisk)}
            • 高风险帧: ${session.frames.count { it.riskLabel == 2 }}帧
            📈 趋势: $trendText
            ⚠️ 异常动作 $anomalies
            💡 专业建议 $advice
            ━━━━━━━━━━━━━━━━━━━━━━━━
            由 RehabGuardian AI 生成
        """.trimIndent()
    }
}