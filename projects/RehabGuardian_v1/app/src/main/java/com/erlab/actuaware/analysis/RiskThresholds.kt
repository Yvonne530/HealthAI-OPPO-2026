package com.erlab.actuaware.analysis

/**
 * Centralized risk thresholds to ensure consistency across all components.
 */
object RiskThresholds {
    const val DANGER = 0.7f
    const val WARNING = 0.4f

    fun riskLevel(score: Float): String = when {
        score > DANGER -> "高风险"
        score > WARNING -> "中风险"
        else -> "低风险"
    }

    fun riskLabel(score: Float): String = when {
        score > DANGER -> "High"
        score > WARNING -> "Medium"
        else -> "Low"
    }

    fun advice(score: Float): String = when {
        score > DANGER -> "检测到高风险动作，建议停止运动，咨询专业康复师。"
        score > WARNING -> "存在轻微风险，建议适当降低运动强度，注意落地姿势。"
        else -> "运动状态良好，继续保持当前运动强度。"
    }
}
