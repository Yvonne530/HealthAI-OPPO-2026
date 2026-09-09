package com.erlab.actuaware

import android.app.AlertDialog
import android.content.Context
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView

class ModelLoadProgressDialog(context: Context) {
    private val dialog: AlertDialog
    private var textViewTitle: TextView? = null
    private var textViewMessage: TextView? = null
    private var progressBar: ProgressBar? = null
    private var textViewProgress: TextView? = null
    private var buttonCancel: Button? = null
    private var onCancelListener: (() -> Unit)? = null
    var isCancelled = false

    init {
        try {
            val builder = AlertDialog.Builder(context)
            val view = LayoutInflater.from(context).inflate(R.layout.dialog_progress, null)

            textViewTitle = view.findViewById(R.id.textViewTitle)
            textViewMessage = view.findViewById(R.id.textViewMessage)
            progressBar = view.findViewById(R.id.progressBar)
            textViewProgress = view.findViewById(R.id.textViewProgress)
            buttonCancel = view.findViewById(R.id.buttonCancel)

            buttonCancel?.setOnClickListener {
                isCancelled = true
                onCancelListener?.invoke()
                dismiss()
            }

            builder.setView(view)
            builder.setCancelable(false)

            dialog = builder.create()

            Log.d("ModelLoadProgressDialog", "Dialog initialized successfully")
        } catch (e: Exception) {
            Log.e("ModelLoadProgressDialog", "Failed to initialize dialog", e)
            throw e
        }
    }

    fun setTitle(title: String) {
        textViewTitle?.text = title
    }

    fun setMessage(message: String) {
        textViewMessage?.text = message
    }

    fun setProgress(progress: Int) {
        progressBar?.progress = progress
        textViewProgress?.text = "$progress%"
    }

    fun setMax(max: Int) {
        progressBar?.max = max
    }

    fun setOnCancelListener(listener: () -> Unit) {
        onCancelListener = listener
    }

    fun show() {
        try {
            dialog.show()
            // 确保对话框有合适的宽度
            val window = dialog.window
            window?.setLayout(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.WRAP_CONTENT
            )
        } catch (e: Exception) {
            Log.e("ModelLoadProgressDialog", "Failed to show dialog", e)
        }
    }

    fun dismiss() {
        try {
            dialog.dismiss()
        } catch (e: Exception) {
            Log.e("ModelLoadProgressDialog", "Failed to dismiss dialog", e)
        }
    }
}