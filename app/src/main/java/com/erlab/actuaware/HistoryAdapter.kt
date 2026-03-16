package com.erlab.actuaware

import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

data class HistoryItem(
    val id: String,
    val title: String,
    val date: String,
    val preview: String,
    val data: String,
    val enableNetwork: Boolean = false,
    val systemPrompt: String = "你是一个有用的助手。"
)

class HistoryAdapter(
    private val onLoad: (HistoryItem) -> Unit,
    private val onDelete: (HistoryItem) -> Unit
) : RecyclerView.Adapter<HistoryAdapter.HistoryViewHolder>() {

    private val historyItems = mutableListOf<HistoryItem>()

    inner class HistoryViewHolder(view: android.view.View) : RecyclerView.ViewHolder(view) {
        val textViewTitle: TextView = view.findViewById(R.id.textViewTitle)
        val textViewDate: TextView = view.findViewById(R.id.textViewDate)
        val textViewPreview: TextView = view.findViewById(R.id.textViewPreview)
        val buttonDelete: Button = view.findViewById(R.id.buttonDelete)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): HistoryViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return HistoryViewHolder(view)
    }

    override fun onBindViewHolder(holder: HistoryViewHolder, position: Int) {
        val item = historyItems[position]
        holder.textViewTitle.text = item.title
        holder.textViewDate.text = item.date
        holder.textViewPreview.text = item.preview

        holder.itemView.setOnClickListener {
            onLoad(item)
        }

        holder.buttonDelete.setOnClickListener {
            onDelete(item)
        }
    }

    override fun getItemCount(): Int = historyItems.size

    fun updateItems(items: List<HistoryItem>) {
        historyItems.clear()
        historyItems.addAll(items)
        notifyDataSetChanged()
    }

    fun removeItem(item: HistoryItem) {
        val position = historyItems.indexOf(item)
        if (position >= 0) {
            historyItems.removeAt(position)
            notifyItemRemoved(position)
        }
    }
}