package com.erlab.actuaware.analysis

object RiskStateMachine {

    enum class State(val label: String) {
        SAFE("SAFE"),
        WARNING("WARNING"),
        DANGER("DANGER");
    }

    var currentScore = 0f
        private set

    private var stateHistory = mutableListOf<Float>()
    private val HISTORY_SIZE = 10

    fun update(
        rawScore: Float,
        confidence: Float = 0f,
        heartRate: Float = 75f,
        sleepScore: Float = 80f
    ): State {
        currentScore = rawScore
        stateHistory.add(rawScore)
        if (stateHistory.size > HISTORY_SIZE) stateHistory.removeAt(0)

        val avgScore = stateHistory.average().toFloat()

        return when {
            avgScore > RiskThresholds.DANGER -> State.DANGER
            avgScore > RiskThresholds.WARNING -> State.WARNING
            else -> State.SAFE
        }
    }

    fun reset() {
        currentScore = 0f
        stateHistory.clear()
    }
}
