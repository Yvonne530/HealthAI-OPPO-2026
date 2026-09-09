package com.erlab.actuaware

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * 历史记录管理助手
 * 使用文件系统存储历史记录到 files/history 子文件夹
 */
class HistoryHelper(private val context: Context) {

    private val historyDir: File
        get() {
            val dir = File(context.filesDir, "history")
            if (!dir.exists()) {
                dir.mkdirs()
            }
            return dir
        }

    /**
     * 获取所有历史记录
     */
    fun getHistoryList(): List<HistoryItem> {
        val historyItems = mutableListOf<HistoryItem>()

        try {
            val files: Array<File> = historyDir.listFiles()?.sortedByDescending { it.lastModified() }?.toTypedArray() ?: emptyArray()

            for (file in files) {
                if (file.isFile && file.extension == "json") {
                    try {
                        val jsonContent = file.readText()
                        val jsonObject = JSONObject(jsonContent)
                        
                        val historyItem = HistoryItem(
                            id = file.nameWithoutExtension,
                            title = jsonObject.getString("title"),
                            date = jsonObject.getString("date"),
                            preview = jsonObject.getString("preview"),
                            data = jsonObject.getString("data"),
                            // 模型字段已弃用，但仍用 optString 读取以保持向后兼容
                            enableNetwork = jsonObject.optBoolean("enableNetwork", false),
                            systemPrompt = jsonObject.optString("systemPrompt", "你是一个有用的助手。")
                        )
                        historyItems.add(historyItem)
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return historyItems
    }

    /**
     * 保存历史记录
     */
    fun saveHistory(historyItem: HistoryItem): Boolean {
        return try {
            val file = File(historyDir, "${historyItem.id}.json")
            val jsonObject = JSONObject().apply {
                put("id", historyItem.id)
                put("title", historyItem.title)
                put("date", historyItem.date)
                put("preview", historyItem.preview)
                put("data", historyItem.data)
                // 不再保存模型相关字段
                put("enableNetwork", historyItem.enableNetwork)
                put("systemPrompt", historyItem.systemPrompt)
            }
                            file.writeText(jsonObject.toString())
                            true
                        } catch (e: Exception) {
                            e.printStackTrace()
                            false
                        }
                    }
                
                    /**
                     * 删除历史记录
                     */
                    fun deleteHistory(historyId: String): Boolean {
                        return try {
                            val file = File(historyDir, "$historyId.json")
                            if (file.exists()) {
                                file.delete()
                            }
                            true
                        } catch (e: Exception) {
                            e.printStackTrace()
                            false
                        }
                    }
                
                    /**
                     * 根据ID获取历史记录
                     */
                    fun getHistoryById(historyId: String): HistoryItem? {
                        return try {
                            val file = File(historyDir, "$historyId.json")
                            if (!file.exists()) {
                                return null
                            }
                            
                            val jsonContent = file.readText()
                            val jsonObject = JSONObject(jsonContent)
                            
                            HistoryItem(
                                id = jsonObject.getString("id"),
                                title = jsonObject.getString("title"),
                                date = jsonObject.getString("date"),
                                preview = jsonObject.getString("preview"),
                                data = jsonObject.getString("data"),
                                // 模型字段已弃用，但仍用 optString 读取以保持向后兼容
                                enableNetwork = jsonObject.optBoolean("enableNetwork", false),
                                systemPrompt = jsonObject.optString("systemPrompt", "你是一个有用的助手。")
                            )
                        } catch (e: Exception) {
                            e.printStackTrace()
                            null
                        }
                    }
                
    /**
     * 清空所有历史记录
     */
    fun clearAllHistory(): Boolean {
        return try {
            val files = historyDir.listFiles() ?: emptyArray()
            for (file in files) {
                file.delete()
            }
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
}