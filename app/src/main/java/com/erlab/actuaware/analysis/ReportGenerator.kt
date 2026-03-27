package com.rehabguardian.analysis

import com.rehabguardian.data.FrameData
import com.rehabguardian.data.SessionRecord

object ReportGenerator {
    data class Report(
        val avgRisk: Float,
        val maxRisk: Float,
        val riskLevel: String,
        val peakFrame: FrameData?,
        val abnormalities: List<String>,
        val suggestions: List<String>,
        val fullText: String
    )

    fun generate(frames: List<FrameData>, session: SessionRecord? = null): Report {
        if (frames.isEmpty()) return emptyReport()

        val avgRisk = frames.map { it.riskScore }.average().toFloat()
        val maxRisk = frames.maxOf { it.riskScore }
        val peakFrame = frames.maxByOrNull { it.riskScore }
        val riskLevel = when {
            avgRisk > 0.7f -> "高风险"
            avgRisk > 0.4f -> "中风险"
            else -> "低风险"
        }

        val abnormalities = mutableListOf<String>()
        val overextFrames = frames.count { it.kneeAngleL > 175f || it.kneeAngleR > 175f }
        val asymmFrames = frames.count { abs(it.kneeAngleL - it.kneeAngleR) > 20f }

        if (overextFrames > frames.size * 0.05) abnormalities.add("膝关节过伸（占${(overextFrames * 100f / frames.size).toInt()}%帧）")
        if (asymmFrames > frames.size * 0.10) abnormalities.add("双腿运动不对称（占${(asymmFrames * 100f / frames.size).toInt()}%帧）")

        val suggestions = buildSuggestions(avgRisk, abnormalities)
        val avgKL = frames.map { it.kneeAngleL }.average()
        val avgKR = frames.map { it.kneeAngleR }.average()

        val fullText = buildString {
            appendLine("🧾 运动分析报告")
            appendLine("─────────────────")
            appendLine("• 平均风险评分：${"%.2f".format(avgRisk)}（$riskLevel）")
            appendLine("• 峰值风险：${"%.2f".format(maxRisk)}")
            appendLine("• 左膝平均角度：${"%.1f".format(avgKL)}°")
            appendLine("• 右膝平均角度：${"%.1f".format(avgKR)}°")
            if (abnormalities.isNotEmpty()) {
                appendLine("─────────────────")
                appendLine("⚠️ 检测到的问题：")
                abnormalities.forEach { appendLine(" • $it") }
            }
            appendLine("─────────────────")
            appendLine("💡 建议：")
            suggestions.forEach { appendLine(" • $it") }
        }

        return Report(avgRisk, maxRisk, riskLevel, peakFrame, abnormalities, suggestions, fullText)
    }

    private fun buildSuggestions(avgRisk: Float, issues: List<String>): List<String> {
        val s = mutableListOf<String>()
        if (avgRisk > 0.7f) s.add("建议停止当前运动，充分休息")
        if (avgRisk > 0.4f) s.add("降低运动强度，注意膝关节保护")
        if (issues.any { it.contains("过伸") }) {
            s.add("加强股四头肌和腘绳肌的力量训练")
            s.add("落地时保持膝关节微屈（15-30°）")
        }
        if (issues.any { it.contains("不对称") }) {
            s.add("进行单腿平衡练习，改善双侧协调性")
        }
        if (s.isEmpty()) s.add("保持当前运动模式，注意适当休息")
        return s
    }

    private fun emptyReport() = Report(
        avgRisk = 0f,
        maxRisk = 0f,
        riskLevel = "暂无数据",
        peakFrame = null,
        abnormalities = emptyList(),
        suggestions = listOf("开始运动以获取分析报告"),
        fullText = "暂无运动数据"
    )
}