package com.erlab.actuaware

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log

/**
 * 性能管理器 - 自动检测设备性能并自适应配置
 */
class PerformanceManager(context: Context) {
    
    companion object {
        private const val TAG = "PerformanceManager"
        
        // 性能等级
        const val PERFORMANCE_LOW = 1
        const val PERFORMANCE_MEDIUM = 2
        const val PERFORMANCE_HIGH = 3
        
        // 最小可用内存阈值（MB）
        private const val MIN_AVAILABLE_MEMORY_HIGH = 2048
        private const val MIN_AVAILABLE_MEMORY_MEDIUM = 1024
    }
    
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    
    // 性能等级
    val performanceLevel: Int
        get() {
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)
            
            val availableMemory = memoryInfo.availMem / (1024 * 1024)
            val totalMemory = memoryInfo.totalMem / (1024 * 1024)
            val memoryPercent = (availableMemory.toFloat() / totalMemory * 100).toInt()
            
            val cpuCount = Runtime.getRuntime().availableProcessors()
            val isHighEndDevice = cpuCount >= 8 && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
            
            Log.d(TAG, "CPU核心数: $cpuCount, 可用内存: ${availableMemory}MB (${memoryPercent}%), 总内存: ${totalMemory}MB")
            
            return when {
                isHighEndDevice && availableMemory >= MIN_AVAILABLE_MEMORY_HIGH -> {
                    Log.d(TAG, "检测到高性能设备")
                    PERFORMANCE_HIGH
                }
                cpuCount >= 4 && availableMemory >= MIN_AVAILABLE_MEMORY_MEDIUM -> {
                    Log.d(TAG, "检测到中性能设备")
                    PERFORMANCE_MEDIUM
                }
                else -> {
                    Log.d(TAG, "检测到低性能设备")
                    PERFORMANCE_LOW
                }
            }
        }
    
    /**
     * 获取推荐的线程数
     */
    fun getRecommendedThreads(): Int {
        val cpuCount = Runtime.getRuntime().availableProcessors()
        return when (performanceLevel) {
            PERFORMANCE_HIGH -> minOf(cpuCount - 1, 12)
            PERFORMANCE_MEDIUM -> minOf(cpuCount - 1, 8)
            else -> minOf(cpuCount - 1, 4)
        }
    }
    
    /**
     * 获取推荐的批处理大小
     */
    fun getRecommendedBatchSize(): Int {
        return when (performanceLevel) {
            PERFORMANCE_HIGH -> 1024
            PERFORMANCE_MEDIUM -> 512
            else -> 256
        }
    }
    
    /**
     * 获取推荐的上下文大小
     */
    fun getRecommendedContextSize(): Int {
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val availableMemory = memoryInfo.availMem / (1024 * 1024)
        
        return when {
            availableMemory >= 4096 -> 16384
            availableMemory >= 2048 -> 12288
            availableMemory >= 1024 -> 8192
            else -> 4096
        }
    }
    
    /**
     * 获取推荐的摄像头帧间隔（毫秒）
     */
    fun getRecommendedCameraInterval(): Long {
        return when (performanceLevel) {
            PERFORMANCE_HIGH -> 50   // 20 FPS
            PERFORMANCE_MEDIUM -> 100 // 10 FPS
            else -> 150               // ~6.7 FPS
        }
    }
    
    /**
     * 获取推荐的图像压缩质量
     */
    fun getRecommendedImageQuality(): Int {
        return when (performanceLevel) {
            PERFORMANCE_HIGH -> 85
            PERFORMANCE_MEDIUM -> 70
            else -> 55
        }
    }
    
    /**
     * 获取推荐的图像最大尺寸
     */
    fun getRecommendedImageMaxDimension(): Int {
        return when (performanceLevel) {
            PERFORMANCE_HIGH -> 1280
            PERFORMANCE_MEDIUM -> 1024
            else -> 800
        }
    }
    
    /**
     * 获取推荐的GPU层数
     */
    fun getRecommendedGpuLayers(): Int {
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val availableMemory = memoryInfo.availMem / (1024 * 1024)
        
        return when {
            availableMemory >= 4096 -> 999
            availableMemory >= 2048 -> 500
            availableMemory >= 1024 -> 200
            else -> 0
        }
    }
    
    /**
     * 获取是否启用Flash Attention
     */
    fun shouldEnableFlashAttention(): Boolean {
        return performanceLevel >= PERFORMANCE_MEDIUM
    }
    
    /**
     * 获取设备信息摘要
     */
    fun getDeviceInfo(): String {
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val cpuCount = Runtime.getRuntime().availableProcessors()
        
        return """
            设备信息:
            - CPU核心数: $cpuCount
            - 可用内存: ${memoryInfo.availMem / (1024 * 1024)}MB
            - 总内存: ${memoryInfo.totalMem / (1024 * 1024)}MB
            - Android版本: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})
            - 性能等级: ${getPerformanceLevelName()}
            - 推荐线程数: ${getRecommendedThreads()}
            - 推荐批处理大小: ${getRecommendedBatchSize()}
            - 推荐上下文大小: ${getRecommendedContextSize()}
            - 推荐摄像头间隔: ${getRecommendedCameraInterval()}ms
        """.trimIndent()
    }
    
    private fun getPerformanceLevelName(): String {
        return when (performanceLevel) {
            PERFORMANCE_HIGH -> "高性能"
            PERFORMANCE_MEDIUM -> "中性能"
            else -> "低性能"
        }
    }
}