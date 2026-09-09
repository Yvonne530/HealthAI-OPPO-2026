package com.healthai.ankle.ui

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.healthai.ankle.db.FrameEntity
import com.healthai.ankle.db.RehabDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Displays the "best frame" for a session:
 *   - Best = lowest risk score among frames with both knees ≥ 150°
 *   - Fallback = global lowest-risk frame in session
 */
class BestFrameActivity : AppCompatActivity() {

    private val db by lazy { RehabDatabase.getInstance(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        supportActionBar?.title = "🐾 最佳动作帧"

        val sessionId = intent.getLongExtra(EXTRA_SESSION_ID, -1L)
        if (sessionId < 0) { finish(); return }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 120, 48, 48)
            setBackgroundColor(Color.parseColor("#FFF7ED"))
        }
        setContentView(root)

        fun tv(text: String, size: Float = 14f, bold: Boolean = false) = TextView(this).apply {
            this.text = text
            textSize = size
            if (bold) typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(Color.parseColor("#1C1917"))
            setPadding(0, 8, 0, 8)
        }

        val titleTv = tv("加载最佳帧…", 20f, bold = true)
        root.addView(titleTv)

        lifecycleScope.launch {
            val frame: FrameEntity? = withContext(Dispatchers.IO) {
                db.dao().bestFrameForSession(sessionId)
                    ?: db.dao().lowestRiskFrame(sessionId)
            }

            if (frame == null) {
                titleTv.text = "暂无数据"
                return@launch
            }

            val riskEmoji = when (frame.riskLabel) { 0 -> "🐱" 1 -> "😿" else -> "🙀" }
            titleTv.text = "$riskEmoji 最佳动作帧"

            fun row(label: String, value: String) {
                root.addView(tv("$label: $value"))
            }

            row("风险分", String.format("%.1f%%", frame.riskScore * 100))
            row("运动评分", "${frame.motionScore.toInt()} / 100")
            row("左膝角度", "${frame.lKneeDeg.toInt()}°")
            row("右膝角度", "${frame.rKneeDeg.toInt()}°")
            row("左脚 GRF Fz", String.format("%.2f BW", frame.grfLeftFz))
            row("右脚 GRF Fz", String.format("%.2f BW", frame.grfRightFz))
            row("置信度", String.format("%.0f%%", frame.confidence * 100))

            if (frame.explanations.isNotEmpty()) {
                root.addView(tv("\n风险提示", 15f, bold = true))
                frame.explanations.split(";").forEach { reason ->
                    if (reason.isNotBlank()) root.addView(tv("• $reason"))
                }
            }

            // Visual accent bar (orange)
            val bar = android.view.View(this@BestFrameActivity).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, 6
                ).also { it.topMargin = 24 }
                setBackgroundColor(Color.parseColor("#F97316"))
            }
            root.addView(bar, 1)
        }
    }

    companion object {
        private const val EXTRA_SESSION_ID = "session_id"
        fun start(ctx: Context, sessionId: Long) {
            ctx.startActivity(Intent(ctx, BestFrameActivity::class.java).apply {
                putExtra(EXTRA_SESSION_ID, sessionId)
            })
        }
    }
}