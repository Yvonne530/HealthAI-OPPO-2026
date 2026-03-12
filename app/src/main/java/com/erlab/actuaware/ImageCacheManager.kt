package com.erlab.actuaware

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import android.util.LruCache
import java.io.ByteArrayOutputStream

/**
 * 图像缓存管理器 - 优化图像处理性能
 */
class ImageCacheManager {
    
    companion object {
        private const val TAG = "ImageCacheManager"
        
        // 缓存大小（基于内存的1/8）
        private const val MEMORY_CACHE_DIVIDER = 8
        
        @Volatile
        private var instance: ImageCacheManager? = null
        
        fun getInstance(): ImageCacheManager {
            return instance ?: synchronized(this) {
                instance ?: ImageCacheManager().also { instance = it }
            }
        }
    }
    
    // 内存缓存
    private val memoryCache: LruCache<String, Bitmap>
    
    // 图像压缩质量缓存
    private val qualityCache = HashMap<String, Int>()
    
    init {
        // 计算缓存大小（最大使用应用内存的1/8）
        val maxMemory = Runtime.getRuntime().maxMemory() / 1024
        val cacheSize = (maxMemory / MEMORY_CACHE_DIVIDER).toInt()
        
        memoryCache = object : LruCache<String, Bitmap>(cacheSize) {
            override fun sizeOf(key: String, bitmap: Bitmap): Int {
                return (bitmap.byteCount / 1024).toInt()
            }
            
            override fun entryRemoved(
                evicted: Boolean,
                key: String,
                oldValue: Bitmap,
                newValue: Bitmap?
            ) {
                if (evicted && !oldValue.isRecycled) {
                    oldValue.recycle()
                }
            }
        }
        
        Log.d(TAG, "图像缓存初始化完成，缓存大小: ${cacheSize}KB")
    }
    
    /**
     * 添加图像到缓存
     */
    fun putBitmap(key: String, bitmap: Bitmap) {
        memoryCache.put(key, bitmap)
    }
    
    /**
     * 从缓存获取图像
     */
    fun getBitmap(key: String): Bitmap? {
        return memoryCache.get(key)
    }
    
    /**
     * 移除缓存的图像
     */
    fun removeBitmap(key: String) {
        memoryCache.remove(key)
    }
    
    /**
     * 清空所有缓存
     */
    fun clearCache() {
        memoryCache.evictAll()
        qualityCache.clear()
        Log.d(TAG, "图像缓存已清空")
    }
    
    /**
     * 获取缓存大小
     */
    fun getCacheSize(): Int {
        return memoryCache.size().toInt()
    }
    
    /**
     * 获取缓存命中数
     */
    fun getHitCount(): Int {
        return memoryCache.hitCount()
    }
    
    /**
     * 获取缓存未命中数
     */
    fun getMissCount(): Int {
        return memoryCache.missCount()
    }
    
    /**
     * 压缩Bitmap为字节数组（带缓存）
     */
    fun compressBitmap(bitmap: Bitmap, quality: Int, key: String? = null): ByteArray {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        return outputStream.toByteArray()
    }
    
    /**
     * 解压字节数组为Bitmap（带缓存）
     */
    fun decompressBitmap(bytes: ByteArray, key: String): Bitmap? {
        // 先尝试从缓存获取
        val cached = memoryCache.get(key)
        if (cached != null && !cached.isRecycled) {
            Log.d(TAG, "从缓存获取图像: $key")
            return cached
        }
        
        // 解码图像
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        
        // 添加到缓存
        if (bitmap != null) {
            memoryCache.put(key, bitmap)
        }
        
        return bitmap
    }
    
    /**
     * 缩放Bitmap到指定尺寸（保持宽高比）
     */
    fun scaleBitmap(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        if (width <= maxDimension && height <= maxDimension) {
            return bitmap
        }
        
        val ratio = maxDimension.toFloat() / maxOf(width, height)
        val newWidth = (width * ratio).toInt()
        val newHeight = (height * ratio).toInt()
        
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
    
    /**
     * 获取缓存统计信息
     */
    fun getCacheStats(): String {
        val maxMemory = Runtime.getRuntime().maxMemory() / 1024
        val usedMemory = Runtime.getRuntime().totalMemory() / 1024
        val freeMemory = Runtime.getRuntime().freeMemory() / 1024
        
        return """
            图像缓存统计:
            - 缓存大小: ${memoryCache.size().toInt()}KB / ${memoryCache.maxSize().toInt()}KB
            - 缓存命中: ${memoryCache.hitCount()}次
            - 缓存未命中: ${memoryCache.missCount()}次
            - 命中率: ${getHitRate()}%
            - 应用内存使用: ${usedMemory}KB / ${maxMemory}KB
            - 应用可用内存: ${freeMemory}KB
        """.trimIndent()
    }
    
    private fun getHitRate(): Float {
        val hits = memoryCache.hitCount().toFloat()
        val misses = memoryCache.missCount().toFloat()
        val total = hits + misses
        return if (total > 0) (hits / total * 100) else 0f
    }
}