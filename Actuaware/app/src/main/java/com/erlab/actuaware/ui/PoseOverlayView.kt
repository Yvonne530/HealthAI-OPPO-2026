package com.healthai.ankle.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.healthai.ankle.inference.RGMarkerMapper

/**
 * Transparent overlay that draws MediaPipe pose skeleton on top of the camera preview.
 */
class PoseOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var landmarks: Array<FloatArray>? = null

    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F97316")
        style = Paint.Style.FILL
        strokeWidth = 6f
    }
    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFF3E0")
        style = Paint.Style.STROKE
        strokeWidth = 3f
        alpha = 200
    }
    private val kneePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#22C55E")
        style = Paint.Style.FILL
    }

    // Skeleton connections (pairs of MediaPipe landmark indices)
    private val CONNECTIONS = listOf(
        11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16,  // arms
        11 to 23, 12 to 24, 23 to 24,                        // torso
        23 to 25, 24 to 26, 25 to 27, 26 to 28,              // legs
        27 to 29, 28 to 30, 29 to 31, 30 to 32               // feet
    )

    fun updateLandmarks(lm: Array<FloatArray>) {
        landmarks = lm
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val lm = landmarks ?: return
        if (lm.size < 33) return

        val w = width.toFloat()
        val h = height.toFloat()

        // Draw connections
        for ((a, b) in CONNECTIONS) {
            val ax = lm[a][0] * w; val ay = lm[a][1] * h
            val bx = lm[b][0] * w; val by = lm[b][1] * h
            canvas.drawLine(ax, ay, bx, by, bonePaint)
        }

        // Draw joints
        for (i in lm.indices) {
            val x = lm[i][0] * w
            val y = lm[i][1] * h
            val paint = if (i == RGMarkerMapper.L_KNEE || i == RGMarkerMapper.R_KNEE) kneePaint else jointPaint
            canvas.drawCircle(x, y, if (i == RGMarkerMapper.L_KNEE || i == RGMarkerMapper.R_KNEE) 10f else 6f, paint)
        }
    }
}