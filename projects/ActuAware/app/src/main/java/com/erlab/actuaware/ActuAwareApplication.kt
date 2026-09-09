package com.erlab.actuaware

import android.app.Application
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * 应用程序入口类
 * 负责初始化全局组件
 */
class ActuAwareApplication : Application() {

    companion object {
        private const val TAG = "ActuAwareApplication"
        lateinit var instance: ActuAwareApplication
            private set
    }

    override fun onCreate() {
        super.onCreate()
        instance = this

        Log.d(TAG, "Application onCreate 开始")

        // 异步初始化非关键组件
        asyncInitNonCriticalComponents()
    }

    /**
     * 异步初始化非关键组件
     * 这些组件的初始化不会阻塞应用启动
     */
    private fun asyncInitNonCriticalComponents() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.d(TAG, "开始异步初始化非关键组件")

                // MMKV 初始化（如果需要自定义配置）
                // 注意：MMKV 会在第一次使用时自动初始化
                // 这里只是确保它在后台初始化，避免第一次使用时的延迟
                initMMKV()

                // 其他可以异步初始化的组件
                initOtherComponents()

                Log.d(TAG, "异步初始化非关键组件完成")
            } catch (e: Exception) {
                Log.e(TAG, "异步初始化失败", e)
            }
        }
    }

    /**
     * 初始化 MMKV
     * MMKV 会在第一次使用时自动初始化，这里只是预热
     */
    private fun initMMKV() {
        try {
            Log.d(TAG, "预热 MMKV...")
            // MMKV 会在第一次访问时自动初始化
            // 这里只是读取一个简单的键值来触发初始化
            val prefs = getSharedPreferences("mmkv_warmup", MODE_PRIVATE)
            prefs.edit().putBoolean("initialized", true).apply()
            Log.d(TAG, "MMKV 预热完成")
        } catch (e: Exception) {
            Log.e(TAG, "MMKV 预热失败", e)
        }
    }

    /**
     * 初始化其他组件
     */
    private fun initOtherComponents() {
        // 这里可以添加其他需要异步初始化的组件
        // 例如：网络库、数据库、分析工具等
    }
}