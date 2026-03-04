package com.erlab.actuaware

import java.io.File

/**
 * 模型文件数据类
 */
data class ModelFile(
    val file: File,
    val type: ModelType
) {
    enum class ModelType {
        MODEL,      // 主模型文件
        MMPROJ      // 多模态文件
    }

    val name: String
        get() = file.name

    val size: String
        get() {
            val bytes = file.length()
            val kb = bytes / 1024.0
            val mb = kb / 1024.0
            val gb = mb / 1024.0
            return when {
                gb >= 1 -> String.format("%.2f GB", gb)
                mb >= 1 -> String.format("%.2f MB", mb)
                kb >= 1 -> String.format("%.2f KB", kb)
                else -> "$bytes B"
            }
        }

    val path: String
        get() = file.absolutePath

    val typeLabel: String
        get() = if (type == ModelType.MODEL) "模型" else "多模态"
}