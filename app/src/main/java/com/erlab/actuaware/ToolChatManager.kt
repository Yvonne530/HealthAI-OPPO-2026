package com.erlab.actuaware

import android.util.Log
import org.json.JSONObject
import org.json.JSONArray

/**
 * 工具聊天管理器
 * 处理 AI 的工具调用循环
 */
class ToolChatManager(
    private val toolManager: ToolManager,
    private val nativeChat: (String, Boolean) -> String
) {
    
    companion object {
        private const val TAG = "ToolChatManager"
        private const val MAX_ITERATIONS = 10 // 最大工具调用次数，防止无限循环
    }
    
    // 工具系统提示词
    private val systemPrompt = buildSystemPrompt()
    
    // 对话历史
    private var conversationHistory = mutableListOf<String>()
    
    /**
     * 带工具的聊天
     */
    fun chatWithTools(userMessage: String): String {
        conversationHistory.add("[用户]|||$userMessage")
        
        var response = nativeChat(conversationHistory.joinToString("\n"), false)
        conversationHistory.add("[助手]|||$response")
        
        Log.d(TAG, "AI 初始回复: $response")
        
        // 循环处理工具调用
        var iterations = 0
        
        while (iterations < MAX_ITERATIONS && containsToolCall(response)) {
            Log.d(TAG, "检测到工具调用，第 ${iterations + 1} 次迭代")
            
            // 解析工具调用
            val toolCalls = parseToolCalls(response)
            
            if (toolCalls.isEmpty()) {
                Log.d(TAG, "未能解析到有效的工具调用")
                break
            }

            // 执行工具调用
            val toolResults = mutableListOf<String>()
            for (toolCall in toolCalls) {
                Log.d(TAG, "执行工具: ${toolCall.tool_name}")
                val result = toolManager.executeToolCall(toolCall)
                toolResults.add("## ${result.tool_name} 的执行结果:\n${result.result}")
                conversationHistory.add("[系统]|||工具 ${result.tool_name} 执行结果: ${result.result}")
            }

            // 将工具结果发送给 AI
            val combinedResults = toolResults.joinToString("\n\n")
            conversationHistory.add("[用户]|||$combinedResults")
            
            // AI 基于工具结果继续生成回复
            response = nativeChat(conversationHistory.joinToString("\n"), false)
            conversationHistory.add("[助手]|||$response")
            Log.d(TAG, "AI 基于工具结果的回复: $response")
            
            iterations++
        }
        
        if (iterations >= MAX_ITERATIONS) {
            Log.w(TAG, "达到最大迭代次数 $MAX_ITERATIONS")
        }
        
        return response
    }
    
    /**
     * 清空对话历史
     */
    fun clearHistory() {
        conversationHistory.clear()
    }
    
    /**
     * 构建系统提示词
     */
    private fun buildSystemPrompt(): String {
        return """
你是一个 AI 助手，可以帮助用户回答问题和执行各种任务。

你可以使用以下工具来执行文件操作：
1. create_file - 创建新文件。参数：filename (文件名), content (文件内容)
2. read_file - 读取文件内容。参数：filename (文件名)
3. modify_file - 修改已存在的文件。参数：filename (文件名), content (新内容)
4. delete_file - 删除指定的文件。参数：filename (文件名)

当需要使用工具时，请按照以下格式回复：

\`\`\`json
{
  "tool_calls": [
    {
      "tool_name": "工具名称",
      "arguments": {
        "参数名": "参数值"
      }
    }
  ]
}
\`\`\`

注意事项：
- 只在需要时使用工具
- 每次只能调用一个工具
- 如果工具执行失败，请尝试其他方法或向用户说明
- 文件操作只能在应用沙盒内进行
- 支持的文件类型包括：txt, md, json, py, js, ts, html, css, xml, yaml, yml, csv, log, ini, cfg, conf, properties, kt, java, c, cpp, h, hpp, rs, go
- 文件大小最大为 10MB
        """.trimIndent()
    }
    
    /**
     * 检查回复中是否包含工具调用
     */
    private fun containsToolCall(response: String): Boolean {
        return response.contains("tool_calls") && 
               (response.contains("create_file") || 
                response.contains("read_file") || 
                response.contains("modify_file") || 
                response.contains("delete_file"))
    }
    
    /**
     * 解析工具调用
     */
    private fun parseToolCalls(response: String): List<ToolManager.ToolCall> {
        val toolCalls = mutableListOf<ToolManager.ToolCall>()
        
        try {
            // 查找 JSON 代码块
            val jsonPattern = Regex("""```json\s*(\{.*?\})\s*```""", RegexOption.DOT_MATCHES_ALL)
            val match = jsonPattern.find(response)
            
            if (match != null) {
                val jsonStr = match.groupValues[1]
                val jsonObject = JSONObject(jsonStr)
                
                if (jsonObject.has("tool_calls")) {
                    val toolCallsArray = jsonObject.getJSONArray("tool_calls")
                    
                    for (i in 0 until toolCallsArray.length()) {
                        val toolCallObj = toolCallsArray.getJSONObject(i)
                        val toolName = toolCallObj.getString("tool_name")
                        val arguments = mutableMapOf<String, Any>()
                        
                        if (toolCallObj.has("arguments")) {
                            val argsObj = toolCallObj.getJSONObject("arguments")
                            val keys = argsObj.keys()
                            while (keys.hasNext()) {
                                val key = keys.next()
                                arguments[key] = argsObj.get(key)
                            }
                        }
                        
                        toolCalls.add(ToolManager.ToolCall(toolName, arguments))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "解析工具调用失败", e)
        }
        
        return toolCalls
    }
}