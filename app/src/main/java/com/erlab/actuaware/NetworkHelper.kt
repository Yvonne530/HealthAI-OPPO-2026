package com.erlab.actuaware

import android.util.Log
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import org.json.JSONArray
import org.json.JSONObject
import kotlin.concurrent.thread

class NetworkHelper {

    companion object {
        private const val TAG = "NetworkHelper"
        private const val USER_AGENT = "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36"
    }

    fun search(query: String, callback: (String?, Exception?) -> Unit) {
        thread {
            try {
                Log.d(TAG, "开始搜索: $query")

                // 使用 Bing 搜索（无需 API Key 的方式）
                val encodedQuery = URLEncoder.encode(query, "UTF-8")
                val url = URL("https://cn.bing.com/search?q=$encodedQuery&format=rss&count=5")

                val connection = url.openConnection() as HttpURLConnection
                connection.apply {
                    requestMethod = "GET"
                    connectTimeout = 15000
                    readTimeout = 15000
                    setRequestProperty("User-Agent", USER_AGENT)
                    setRequestProperty("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
                    setRequestProperty("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
                }

                val responseCode = connection.responseCode
                Log.d(TAG, "响应码: $responseCode")

                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    val results = parseBingResponse(response)
                    callback(results, null)
                } else {
                    val error = "HTTP 错误: $responseCode"
                    Log.e(TAG, error)
                    callback(null, Exception(error))
                }

                connection.disconnect()
            } catch (e: Exception) {
                Log.e(TAG, "搜索失败", e)
                callback(null, e)
            }
        }
    }

    private fun parseBingResponse(response: String): String {
        try {
            val results = StringBuilder()

            val items = Regex("<item[^>]*>(.*?)</item>", RegexOption.DOT_MATCHES_ALL).findAll(response)
            var count = 0

            for ((index, item) in items.withIndex()) {
                if (index >= 5) break

                val itemContent = item.groupValues[1]

                val titleMatch = Regex("<title>(.*?)</title>").find(itemContent)
                val title = titleMatch?.groupValues?.get(1)?.replace("<!\\[CDATA\\[", "")
                    ?.replace("]]>", "") ?: "未知标题"

                val descMatch = Regex("<description>(.*?)</description>").find(itemContent)
                var description = descMatch?.groupValues?.get(1)?.replace("<!\\[CDATA\\[", "")
                    ?.replace("]]>", "")?.replace(Regex("<[^>]+>"), "") ?: ""

                description = description
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&amp;", "&")
                    .replace("&quot;", "\"")
                    .replace("&apos;", "'")
                    .replace("&nbsp;", " ")

                if (description.length > 300) {
                    description = description.substring(0, 300) + "..."
                }

                results.append("${index + 1}. $title\n$description\n\n")
                count++
            }

            if (count == 0) {
                results.append("未找到相关搜索结果。请尝试使用更具体的关键词。")
            }

            return results.toString().trim()
        } catch (e: Exception) {
            Log.e(TAG, "解析响应失败", e)
            return "解析搜索结果时出错: ${e.message}"
        }
    }

    fun fetchUrl(urlStr: String, callback: (String?, Exception?) -> Unit) {
        thread {
            try {
                Log.d(TAG, "获取网页: $urlStr")

                val url = URL(urlStr)
                val connection = url.openConnection() as HttpURLConnection
                connection.apply {
                    requestMethod = "GET"
                    connectTimeout = 15000
                    readTimeout = 15000
                    setRequestProperty("User-Agent", USER_AGENT)
                    setRequestProperty("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
                    setRequestProperty("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
                }

                val responseCode = connection.responseCode
                Log.d(TAG, "响应码: $responseCode")

                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    val content = extractTextContent(response)
                    callback(content, null)
                } else {
                    val error = "HTTP 错误: $responseCode"
                    Log.e(TAG, error)
                    callback(null, Exception(error))
                }

                connection.disconnect()
            } catch (e: Exception) {
                Log.e(TAG, "获取网页失败", e)
                callback(null, e)
            }
        }
    }

    private fun extractTextContent(html: String): String {
        try {
            var text = html.replace(Regex("<script[^>]*>.*?</script>", RegexOption.DOT_MATCHES_ALL), "")
                .replace(Regex("<style[^>]*>.*?</style>", RegexOption.DOT_MATCHES_ALL), "")

            text = text.replace(Regex("<[^>]+>"), " ")

            text = text
                .replace("&nbsp;", " ")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
                .replace("&quot;", "\"")
                .replace("&apos;", "'")
                .replace("&#39;", "'")
                .replace("&mdash;", "—")
                .replace("&ndash;", "–")
                .replace("&hellip;", "…")

            text = text.replace(Regex("\\s+"), " ").trim()

            val maxLength = 5000
            return if (text.length > maxLength) {
                text.substring(0, maxLength) + "\n\n...(内容过长，已截断)"
            } else {
                text
            }
        } catch (e: Exception) {
            Log.e(TAG, "提取文本内容失败", e)
            return "提取网页内容时出错: ${e.message}"
        }
    }

    fun isNetworkAvailable(): Boolean {
        return try {
            val url = URL("https://www.baidu.com")
            val connection = url.openConnection() as HttpURLConnection
            connection.connectTimeout = 3000
            connection.disconnect()
            true
        } catch (e: Exception) {
            false
        }
    }
}