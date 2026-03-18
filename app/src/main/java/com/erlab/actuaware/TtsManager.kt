package com.erlab.actuaware

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

class TtsManager(private val context: Context) : TextToSpeech.OnInitListener {
    
    private var tts: TextToSpeech? = null
    private var isInitialized = false
    private var onStartCallback: (() -> Unit)? = null
    private var onDoneCallback: (() -> Unit)? = null
    private var onErrorCallback: ((String) -> Unit)? = null
    
    // TTS 参数
    private var speechRate = 1.0f  // 语速: 0.5-2.0
    private var pitch = 1.0f       // 音调: 0.5-2.0
    private var currentLanguage = Locale.CHINESE

    // 初始化 TTS
    fun init() {
        tts = TextToSpeech(context, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            isInitialized = true
            
            // 设置中文语音
            val result = tts?.setLanguage(currentLanguage)
            if (result == TextToSpeech.LANG_MISSING_DATA || 
                result == TextToSpeech.LANG_NOT_SUPPORTED) {
                // 如果中文不可用，使用英文
                tts?.setLanguage(Locale.US)
                currentLanguage = Locale.US
            }
            
            // 设置语速和音调
            tts?.setSpeechRate(speechRate)
            tts?.setPitch(pitch)
            
            // 设置播放完成监听
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    onStartCallback?.invoke()
                }
                
                override fun onDone(utteranceId: String?) {
                    onDoneCallback?.invoke()
                }
                
                override fun onError(utteranceId: String?) {
                    onErrorCallback?.invoke("TTS Error")
                }
                
                override fun onError(utteranceId: String?, errorCode: Int) {
                    onErrorCallback?.invoke("TTS Error: $errorCode")
                }
            })
        }
    }

    // 播放文本
    fun speak(text: String, queueMode: Int = TextToSpeech.QUEUE_FLUSH) {
        if (!isInitialized) {
            init()
            return
        }
        
        tts?.speak(text, queueMode, null, "utterance_${System.currentTimeMillis()}")
    }

    // 停止播放
    fun stop() {
        tts?.stop()
    }

    // 暂停播放（Android TTS 原生不支持暂停，使用 stop 替代）
    fun pause() {
        tts?.stop()
    }

    // 设置语速 (0.5 - 2.0, 默认 1.0)
    fun setSpeechRate(rate: Float) {
        speechRate = rate.coerceIn(0.5f, 2.0f)
        tts?.setSpeechRate(speechRate)
    }

    // 设置音调 (0.5 - 2.0, 默认 1.0)
    fun setPitch(pitch: Float) {
        this.pitch = pitch.coerceIn(0.5f, 2.0f)
        tts?.setPitch(this.pitch)
    }

    // 设置语言
    fun setLanguage(locale: Locale) {
        currentLanguage = locale
        tts?.setLanguage(locale)
    }

    // 获取可用语言
    fun getAvailableLanguages(): List<Locale> {
        return tts?.availableLanguages?.toList() ?: emptyList()
    }

    // 检查是否支持指定语言
    fun isLanguageSupported(locale: Locale): Boolean {
        val result = tts?.isLanguageAvailable(locale)
        return result == TextToSpeech.LANG_AVAILABLE ||
               result == TextToSpeech.LANG_COUNTRY_AVAILABLE ||
               result == TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE
    }

    // 设置回调
    fun setOnStartListener(callback: () -> Unit) {
        onStartCallback = callback
    }

    fun setOnDoneListener(callback: () -> Unit) {
        onDoneCallback = callback
    }

    fun setOnErrorListener(callback: (String) -> Unit) {
        onErrorCallback = callback
    }

    // 释放资源
    fun release() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
