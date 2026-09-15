package com.healthai.ankle.ui

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.*
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.healthai.ankle.R
import com.healthai.ankle.db.FrameEntity
import com.healthai.ankle.db.RehabDatabase
import com.healthai.ankle.db.SessionEntity
import com.healthai.ankle.report.PdfReportBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class SessionHistoryActivity : AppCompatActivity() {

    private lateinit var recycler: RecyclerView
    private val db by lazy { RehabDatabase.getInstance(this) }
    private val adapter = SessionAdapter(::onSessionClick, ::onExportClick)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Inline layout to avoid extra XML file
        recycler = RecyclerView(this).apply {
            layoutManager = LinearLayoutManager(this@SessionHistoryActivity)
            this.adapter = this@SessionHistoryActivity.adapter
            setPadding(0, 80, 0, 0)
        }
        setContentView(recycler)
        supportActionBar?.title = "🐾 历史训练记录"

        lifecycleScope.launch {
            db.dao().allSessions().collect { list ->
                adapter.submitList(list)
            }
        }
    }

    private fun onSessionClick(session: SessionEntity) {
        // Open best-frame replay
        lifecycleScope.launch {
            val bestFrame = withContext(Dispatchers.IO) {
                db.dao().bestFrameForSession(session.id)
                    ?: db.dao().lowestRiskFrame(session.id)
            }
            bestFrame?.let { BestFrameActivity.start(this@SessionHistoryActivity, session.id) }
        }
    }

    private fun onExportClick(session: SessionEntity) {
        lifecycleScope.launch {
            val frames = withContext(Dispatchers.IO) { db.dao().framesForSession(session.id) }
            val outFile = File(getExternalFilesDir("reports"), "rehab_${session.id}.pdf")
            withContext(Dispatchers.IO) {
                PdfReportBuilder.build(this@SessionHistoryActivity, session, frames, outFile)
            }
            android.widget.Toast.makeText(
                this@SessionHistoryActivity, "报告已保存", android.widget.Toast.LENGTH_SHORT
            ).show()
        }
    }

    companion object {
        fun start(ctx: Context) = ctx.startActivity(Intent(ctx, SessionHistoryActivity::class.java))
    }
}

// ── Adapter ───────────────────────────────────────────────────────────────────

class SessionAdapter(
    private val onClick: (SessionEntity) -> Unit,
    private val onExport: (SessionEntity) -> Unit
) : RecyclerView.Adapter<SessionAdapter.VH>() {

    private val items = mutableListOf<SessionEntity>()
    private val fmt = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())

    fun submitList(list: List<SessionEntity>) {
        items.clear(); items.addAll(list); notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(android.R.layout.two_line_list_item, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val s = items[position]
        val riskEmoji = when (s.maxRiskLabel) { 0 -> "🐱" 1 -> "😿" else -> "🙀" }
        holder.line1.text = "$riskEmoji  ${fmt.format(Date(s.startTimeMs))}  ·  运动评分 ${s.avgMotionScore.toInt()}"
        holder.line2.text = "帧数 ${s.totalFrames}  |  点击回放  |  长按导出报告"
        holder.itemView.setOnClickListener { onClick(s) }
        holder.itemView.setOnLongClickListener { onExport(s); true }
    }

    override fun getItemCount() = items.size

    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val line1: TextView = v.findViewById(android.R.id.text1)
        val line2: TextView = v.findViewById(android.R.id.text2)
    }
}
