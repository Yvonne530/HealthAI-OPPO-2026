package com.healthai.ankle.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.healthai.ankle.db.RehabDatabase
import com.healthai.ankle.db.SessionEntity
import com.healthai.ankle.inference.InferenceResult
import com.healthai.ankle.inference.RGPhaseAEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class RehabViewModel(app: Application) : AndroidViewModel(app) {

    val engine = RGPhaseAEngine()
    private val db = RehabDatabase.getInstance(app)

    // Live result for UI observation
    private val _latestResult = MutableLiveData<InferenceResult?>()
    val latestResult: LiveData<InferenceResult?> = _latestResult

    // All sessions for history screen
    val allSessions: StateFlow<List<SessionEntity>> = db.dao().allSessions()
        .stateIn(viewModelScope, SharingStarted.Lazily, emptyList())

    // Session state
    private val _sessionId = MutableLiveData<Long>(-1L)
    val sessionId: LiveData<Long> = _sessionId

    private var _frameCount = 0
    val frameCount get() = _frameCount

    init {
        viewModelScope.launch(Dispatchers.IO) {
            engine.init(app)
        }
    }

    fun startSession(weightKg: Float) {
        engine.userWeightKg = weightKg
        viewModelScope.launch(Dispatchers.IO) {
            val id = db.dao().insertSession(
                SessionEntity(userWeightKg = weightKg)
            )
            withContext(Dispatchers.Main) { _sessionId.value = id }
        }
    }

    fun publishResult(result: InferenceResult) {
        _frameCount++
        _latestResult.postValue(result)
    }

    fun stopSession(
        avgMotion: Float,
        maxRisk: Int,
        totalFrames: Int
    ) {
        val sid = _sessionId.value ?: return
        viewModelScope.launch(Dispatchers.IO) {
            db.dao().updateSession(SessionEntity(
                id             = sid,
                endTimeMs      = System.currentTimeMillis(),
                userWeightKg   = engine.userWeightKg,
                avgMotionScore = avgMotion,
                maxRiskLabel   = maxRisk,
                totalFrames    = totalFrames
            ))
        }
        _sessionId.value = -1L
        _frameCount = 0
    }

    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch(Dispatchers.IO) { engine.release() }
    }
}
