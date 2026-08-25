package com.erlab.actuaware.data

import com.erlab.actuaware.analysis.RiskThresholds
import com.google.gson.Gson
import kotlin.math.abs

data class FrameData(
    val timestampMs: Long,
    val keypoints: Array<FloatArray>, // [33][3]
    val riskScore: Float,
    val riskLabel: Int,
    val grfLeft: FloatArray,
    val grfRight: FloatArray,
    val kneeAngleL: Float,
    val kneeAngleR: Float,
) {
    override fun equals(other: Any?) = other is FrameData && timestampMs == other.timestampMs
    override fun hashCode() = timestampMs.hashCode()
}

data class Session(
    val id: String = java.util.UUID.randomUUID().toString(),
    val startMs: Long = System.currentTimeMillis(),
    var endMs: Long = 0L,
    val frames: MutableList<FrameData> = mutableListOf(),
) {
    val durationSec: Float get() = (endMs - startMs) / 1000f
    val avgRisk: Float get() = if (frames.isEmpty()) 0f else frames.map { it.riskScore }.average().toFloat()
    val maxRisk: Float get() = frames.maxOfOrNull { it.riskScore } ?: 0f
    val isHighRisk: Boolean get() = frames.count { it.riskLabel == 2 } > frames.size * 0.1f
    val riskLevelText: String get() = RiskThresholds.riskLevel(avgRisk)

    fun topAnomalies(): List<String> {
        val a = mutableListOf<String>()
        if (frames.count { it.kneeAngleL > 175f || it.kneeAngleR > 175f } > 3) a += "膝关节过伸"
        if (frames.count { abs(it.kneeAngleL - it.kneeAngleR) > 20f } > frames.size * 0.2f) a += "左右步态不对称"
        if (frames.count { it.grfLeft[2] + it.grfRight[2] > 1200f } > 3) a += "落地冲击力过大"
        return a.ifEmpty { listOf("未检测到明显异常") }
    }

    fun toJson(): String = Gson().toJson(this)
}

data class HistoryRecord(
    val sessionId: String,
    val date: String,
    val durationSec: Float,
    val avgRisk: Float,
    val maxRisk: Float,
    val isHighRisk: Boolean,
    val anomalies: List<String>,
)