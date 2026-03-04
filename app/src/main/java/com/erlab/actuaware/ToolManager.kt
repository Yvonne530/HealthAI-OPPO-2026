package com.erlab.actuaware

import android.content.Context
import android.util.Log
import java.io.File

/**
 * 工具管理器
 * 管理 AI 可以调用的工具函数
 */
class ToolManager(private val context: Context) {
    
    companion object {
        private const val TAG = "ToolManager"
        private const val MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB
    }
    
    // 工具调用请求
    data class ToolCall(
        val tool_name: String,
        val arguments: Map<String, Any>
    )
    
    // 工具执行结果
    data class ToolResult(
        val tool_name: String,
        val result: String,
        val success: Boolean
    )
    
    // 执行工具调用
    fun executeToolCall(toolCall: ToolCall): ToolResult {
        return try {
            Log.d(TAG, "执行工具调用: ${toolCall.tool_name}")
            
            when (toolCall.tool_name) {
                "create_file" -> executeCreateFile(toolCall.arguments)
                "read_file" -> executeReadFile(toolCall.arguments)
                "modify_file" -> executeModifyFile(toolCall.arguments)
                "delete_file" -> executeDeleteFile(toolCall.arguments)
                else -> ToolResult(
                    tool_name = toolCall.tool_name,
                    result = "❌ 未知工具: ${toolCall.tool_name}",
                    success = false
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "工具执行失败", e)
            ToolResult(
                tool_name = toolCall.tool_name,
                result = "❌ 执行失败: ${e.message}",
                success = false
            )
        }
    }
    
    // 创建新文件
    private fun executeCreateFile(args: Map<String, Any>): ToolResult {
        return try {
            val filename = args["filename"] as? String
                ?: return ToolResult("create_file", "❌ 缺少 filename 参数", false)
            val content = args["content"] as? String
                ?: return ToolResult("create_file", "❌ 缺少 content 参数", false)

            // 安全验证
            if (!validateFilePath(filename)) {
                return ToolResult("create_file", "❌ 文件路径非法", false)
            }

            if (!validateFileExtension(filename)) {
                return ToolResult("create_file", "❌ 不支持的文件类型", false)
            }

            if (!validateFileSize(content)) {
                return ToolResult("create_file", "❌ 文件内容过大（最大 10MB）", false)
            }

            val file = File(context.filesDir, filename)

            if (file.exists()) {
                return ToolResult("create_file", "❌ 文件已存在，请使用 modify_file 工具来修改文件", false)
            }

            // 确保目录存在
            file.parentFile?.mkdirs()

            file.writeText(content)

            Log.d(TAG, "文件创建成功: ${file.absolutePath}")

            ToolResult(
                tool_name = "create_file",
                result = "✅ 文件已创建: $filename\n路径: ${file.absolutePath}\n大小: ${file.length()} bytes",
                success = true
            )
        } catch (e: Exception) {
            Log.e(TAG, "创建文件失败", e)
            ToolResult("create_file", "❌ 创建失败: ${e.message}", false)
        }
    }

    // 修改已存在的文件
    private fun executeModifyFile(args: Map<String, Any>): ToolResult {
        return try {
            val filename = args["filename"] as? String
                ?: return ToolResult("modify_file", "❌ 缺少 filename 参数", false)
            val content = args["content"] as? String
                ?: return ToolResult("modify_file", "❌ 缺少 content 参数", false)

            // 安全验证
            if (!validateFilePath(filename)) {
                return ToolResult("modify_file", "❌ 文件路径非法", false)
            }

            if (!validateFileExtension(filename)) {
                return ToolResult("modify_file", "❌ 不支持的文件类型", false)
            }

            if (!validateFileSize(content)) {
                return ToolResult("modify_file", "❌ 文件内容过大（最大 10MB）", false)
            }

            val file = File(context.filesDir, filename)

            if (!file.exists()) {
                return ToolResult("modify_file", "❌ 文件不存在，请使用 create_file 工具来创建文件", false)
            }

            file.writeText(content)

            Log.d(TAG, "文件修改成功: ${file.absolutePath}")

            ToolResult(
                tool_name = "modify_file",
                result = "✅ 文件已修改: $filename\n路径: ${file.absolutePath}\n大小: ${file.length()} bytes",
                success = true
            )
        } catch (e: Exception) {
            Log.e(TAG, "修改文件失败", e)
            ToolResult("modify_file", "❌ 修改失败: ${e.message}", false)
        }
    }
    
    // 读取文件
    private fun executeReadFile(args: Map<String, Any>): ToolResult {
        return try {
            val filename = args["filename"] as? String
                ?: return ToolResult("read_file", "❌ 缺少 filename 参数", false)

            // 安全验证
            if (!validateFilePath(filename)) {
                return ToolResult("read_file", "❌ 文件路径非法", false)
            }

            val file = File(context.filesDir, filename)

            if (!file.exists()) {
                return ToolResult("read_file", "❌ 文件不存在: $filename", false)
            }

            // 验证文件大小，避免读取过大文件
            if (file.length() > MAX_FILE_SIZE) {
                return ToolResult("read_file", "❌ 文件过大（最大 10MB）", false)
            }

            val content = file.readText()

            // 限制返回内容长度
            val maxLength = 10000
            val displayContent = if (content.length > maxLength) {
                content.substring(0, maxLength) + "\n\n...(内容过长，已截断)"
            } else {
                content
            }

            Log.d(TAG, "文件读取成功: $filename")

            ToolResult(
                tool_name = "read_file",
                result = "📄 文件内容 ($filename):\n```\n$displayContent\n```",
                success = true
            )
        } catch (e: Exception) {
            Log.e(TAG, "读取文件失败", e)
            ToolResult("read_file", "❌ 读取失败: ${e.message}", false)
        }
    }
    
    // 删除文件
    private fun executeDeleteFile(args: Map<String, Any>): ToolResult {
        return try {
            val filename = args["filename"] as? String 
                ?: return ToolResult("delete_file", "❌ 缺少 filename 参数", false)
            
            // 安全验证
            if (!validateFilePath(filename)) {
                return ToolResult("delete_file", "❌ 文件路径非法", false)
            }
            
            val file = File(context.filesDir, filename)
            
            if (!file.exists()) {
                return ToolResult("delete_file", "❌ 文件不存在: $filename", false)
            }
            
            if (file.isDirectory) {
                return ToolResult("delete_file", "❌ 不能删除目录，只能删除文件", false)
            }
            
            val deleted = file.delete()
            
            if (!deleted) {
                return ToolResult("delete_file", "❌ 删除失败", false)
            }
            
            Log.d(TAG, "文件删除成功: $filename")
            
            ToolResult(
                tool_name = "delete_file",
                result = "✅ 文件已删除: $filename",
                success = true
            )
        } catch (e: Exception) {
            Log.e(TAG, "删除文件失败", e)
            ToolResult("delete_file", "❌ 删除失败: ${e.message}", false)
        }
    }
    
    // 验证文件路径（防止路径遍历攻击）
    private fun validateFilePath(filename: String): Boolean {
        return try {
            val baseDir = context.filesDir.normalize()
            val fullPath = File(context.filesDir, filename).normalize()
            
            // 确保路径在允许的目录内
            fullPath.startsWith(baseDir)
        } catch (e: Exception) {
            Log.e(TAG, "路径验证失败", e)
            false
        }
    }
    
    // 验证文件扩展名
    private fun validateFileExtension(filename: String): Boolean {
        val allowedExtensions = listOf(
            "txt", "md", "json", "py", "js", "ts", "html", "css", "xml",
            "yaml", "yml", "csv", "log", "ini", "cfg", "conf", "properties",
            "kt", "java", "c", "cpp", "h", "hpp", "rs", "go"
        )

        val extension = filename.substringAfterLast('.', "").lowercase()
        
        // 如果没有扩展名，则不允许
        if (extension.isEmpty() || filename.endsWith('.')) {
            return false
        }
        
        return allowedExtensions.contains(extension)
    }
    
    // 验证文件大小
    private fun validateFileSize(content: String): Boolean {
        return content.toByteArray().size <= MAX_FILE_SIZE
    }
}