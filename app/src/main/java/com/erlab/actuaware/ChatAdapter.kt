package com.erlab.actuaware

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.ext.tasklist.TaskListPlugin
import java.io.File

class ChatAdapter : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val messages = mutableListOf<ChatMessage>()

    private lateinit var markwon: Markwon

    var userNickname: String = "用户"
        set(value) {
            field = value
            notifyDataSetChanged()
        }
    var userAvatarPath: String? = null
        set(value) {
            field = value
            notifyDataSetChanged()
        }
    var aiNickname: String = "AI助手"
        set(value) {
            field = value
            notifyDataSetChanged()
        }
    var aiAvatarPath: String? = null
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    companion object {
        private const val TYPE_USER = 1
        private const val TYPE_AI = 2
        private const val TYPE_SYSTEM = 3
    }

    sealed class ChatMessage(val type: Int) {
        data class UserMessage(val message: String, val imageUri: Uri? = null) : ChatMessage(TYPE_USER)
        data class AIMessage(
            val message: String,
            val tokenCount: Int = 0,
            val duration: Long = 0
        ) : ChatMessage(TYPE_AI)
        data class SystemMessage(val message: String) : ChatMessage(TYPE_SYSTEM)
    }

    inner class UserViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val textViewMessage: TextView = view.findViewById(R.id.textViewMessage)
        val imageView: ImageView = view.findViewById(R.id.imageView)
        val imageViewAvatar: ImageView = view.findViewById(R.id.imageViewAvatar)
        val textViewNickname: TextView = view.findViewById(R.id.textViewNickname)
        val linearLayoutContent: android.widget.LinearLayout = view.findViewById(R.id.linearLayoutContent)
    }

    inner class AIViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val textViewMessage: TextView = view.findViewById(R.id.textViewMessage)
        val textViewMeta: TextView = view.findViewById(R.id.textViewMeta)
        val buttonCopy: Button = view.findViewById(R.id.buttonCopy)
        val imageViewAvatar: ImageView = view.findViewById(R.id.imageViewAvatar)
        val textViewNickname: TextView = view.findViewById(R.id.textViewNickname)
    }

    inner class SystemViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val textViewMessage: TextView = view.findViewById(R.id.textViewMessage)
    }

    override fun getItemViewType(position: Int): Int {
        return messages[position].type
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        if (!::markwon.isInitialized) {
            markwon = Markwon.builder(parent.context)
                .usePlugin(StrikethroughPlugin.create())
                .usePlugin(TablePlugin.create(parent.context))
                .usePlugin(TaskListPlugin.create(parent.context))
                .build()
        }

        return when (viewType) {
            TYPE_USER -> {
                val view = LayoutInflater.from(parent.context)
                    .inflate(R.layout.item_chat_message_user, parent, false)
                UserViewHolder(view)
            }
            TYPE_AI -> {
                val view = LayoutInflater.from(parent.context)
                    .inflate(R.layout.item_chat_message_ai, parent, false)
                AIViewHolder(view)
            }
            TYPE_SYSTEM -> {
                val view = LayoutInflater.from(parent.context)
                    .inflate(R.layout.item_chat_message_system, parent, false)
                SystemViewHolder(view)
            }
            else -> throw IllegalArgumentException("Unknown view type: $viewType")
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]
        when (holder) {
            is UserViewHolder -> {
                val userMessage = message as ChatMessage.UserMessage
                holder.textViewMessage.text = userMessage.message
                holder.textViewNickname.text = userNickname

                loadAvatar(holder.imageViewAvatar, userAvatarPath)

                if (userMessage.imageUri != null) {
                    holder.imageView.visibility = View.VISIBLE
                    holder.imageView.setImageURI(userMessage.imageUri)
                } else {
                    holder.imageView.visibility = View.GONE
                    holder.imageView.setImageURI(null)
                }
            }
            is AIViewHolder -> {
                val aiMessage = message as ChatMessage.AIMessage
                holder.textViewNickname.text = aiNickname

                loadAvatar(holder.imageViewAvatar, aiAvatarPath)

                if (::markwon.isInitialized) {
                    markwon.setMarkdown(holder.textViewMessage, aiMessage.message)
                } else {
                    holder.textViewMessage.text = aiMessage.message
                }

                if (aiMessage.tokenCount > 0 || aiMessage.duration > 0) {
                    holder.textViewMeta.visibility = View.VISIBLE
                    val durationStr = if (aiMessage.duration > 0) {
                        String.format("%.2f秒", aiMessage.duration / 1000.0)
                    } else ""
                    holder.textViewMeta.text = "${aiMessage.tokenCount} tokens · $durationStr"

                    holder.buttonCopy.setOnClickListener {
                        val clipboard = holder.itemView.context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        val clip = ClipData.newPlainText("AI 回复", aiMessage.message)
                        clipboard.setPrimaryClip(clip)
                        Toast.makeText(holder.itemView.context, "已复制到剪贴板", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    holder.textViewMeta.visibility = View.GONE
                    holder.buttonCopy.visibility = View.GONE
                }
            }
            is SystemViewHolder -> {
                holder.textViewMessage.text = (message as ChatMessage.SystemMessage).message
            }
        }
    }

    private fun loadAvatar(imageView: ImageView, avatarPath: String?) {
        if (avatarPath.isNullOrEmpty()) {
            imageView.setImageResource(android.R.drawable.ic_menu_gallery)
            imageView.setBackgroundColor(0xFF333333.toInt())
        } else {
            try {
                val file = File(avatarPath)
                if (file.exists()) {
                    val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                    if (bitmap != null) {
                        imageView.setImageBitmap(bitmap)
                        imageView.setBackgroundColor(0x00000000)
                    } else {
                        imageView.setImageResource(android.R.drawable.ic_menu_gallery)
                        imageView.setBackgroundColor(0xFF333333.toInt())
                    }
                } else {
                    imageView.setImageResource(android.R.drawable.ic_menu_gallery)
                    imageView.setBackgroundColor(0xFF333333.toInt())
                }
            } catch (e: Exception) {
                imageView.setImageResource(android.R.drawable.ic_menu_gallery)
                imageView.setBackgroundColor(0xFF333333.toInt())
            }
        }
    }

    override fun getItemCount(): Int = messages.size

    fun addUserMessage(message: String, imageUri: Uri? = null) {
        messages.add(ChatMessage.UserMessage(message, imageUri))
        notifyItemInserted(messages.size - 1)
    }

    fun addAIMessage(message: String, tokenCount: Int = 0, duration: Long = 0) {
        messages.add(ChatMessage.AIMessage(message, tokenCount, duration))
        notifyItemInserted(messages.size - 1)
    }

    fun addSystemMessage(message: String) {
        messages.add(ChatMessage.SystemMessage(message))
        notifyItemInserted(messages.size - 1)
    }

    fun updateLastAIMessage(message: String) {
        if (messages.isNotEmpty() && messages.last() is ChatMessage.AIMessage) {
            val lastMessage = messages.last() as ChatMessage.AIMessage
            messages[messages.size - 1] = lastMessage.copy(message = message)
            notifyItemChanged(messages.size - 1)
        }
    }

    fun updateLastAIMessageMeta(tokenCount: Int, duration: Long) {
        if (messages.isNotEmpty() && messages.last() is ChatMessage.AIMessage) {
            val lastMessage = messages.last() as ChatMessage.AIMessage
            messages[messages.size - 1] = lastMessage.copy(tokenCount = tokenCount, duration = duration)
            notifyItemChanged(messages.size - 1)
        }
    }

    fun clearMessages() {
        messages.clear()
        notifyDataSetChanged()
    }
}