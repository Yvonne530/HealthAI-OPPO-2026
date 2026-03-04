package com.erlab.actuaware

import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class ModelAdapter : RecyclerView.Adapter<ModelAdapter.ModelViewHolder>() {

    private val models = mutableListOf<ModelFile>()

    inner class ModelViewHolder(view: android.view.View) : RecyclerView.ViewHolder(view) {
        val textViewModelName: TextView = view.findViewById(R.id.textViewModelName)
        val textViewModelType: TextView = view.findViewById(R.id.textViewModelType)
        val textViewModelSize: TextView = view.findViewById(R.id.textViewModelSize)
        val textViewModelPath: TextView = view.findViewById(R.id.textViewModelPath)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_model, parent, false)
        return ModelViewHolder(view)
    }

    override fun onBindViewHolder(holder: ModelViewHolder, position: Int) {
        val modelFile = models[position]

        holder.textViewModelName.text = modelFile.name
        holder.textViewModelType.text = modelFile.typeLabel
        holder.textViewModelSize.text = "大小: ${modelFile.size}"
        holder.textViewModelPath.text = modelFile.path

        // 设置不同类型标签的颜色
        if (modelFile.type == ModelFile.ModelType.MODEL) {
            holder.textViewModelType.setTextColor(0xFF4CAF50.toInt())
            holder.textViewModelType.setBackgroundColor(0x80E8F5E9.toInt())
        } else {
            holder.textViewModelType.setTextColor(0xFF9C27B0.toInt())
            holder.textViewModelType.setBackgroundColor(0x80F3E5F5.toInt())
        }
    }

    override fun getItemCount(): Int = models.size

    fun updateItems(newModels: List<ModelFile>) {
        models.clear()
        models.addAll(newModels)
        notifyDataSetChanged()
    }
}