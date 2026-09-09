package com.healthai.ankle.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.lifecycle.lifecycleScope
import com.healthai.ankle.inference.InferenceResult
import com.healthai.ankle.inference.RGPhaseAEngine
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Video-based test activity.
 * Loads an MP4 from external storage, extracts frames, runs MediaPipe Pose + STGCN/FNO/Risk pipeline,
 * and outputs a detailed report with latency stats and risk distribution.
 */
class VideoTestActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "VideoTest"
        private const val REQUEST_CODE = 1001
        const val VIDEO_PATH = "/sdcard/Download/tmp_test_human.mp4"
    }

    private var poseLandmarker: PoseLandmarker? = null
    private val engine = RGPhaseAEngine()

    // Test statistics
    private var totalFrames = 0
    private var poseSuccessFrames = 0
    private var inferenceSuccessFrames = 0
    private val frameLatenciesMs = mutableListOf<Long>()
    private val poseLatenciesMs = mutableListOf<Long>()
    private val inferenceLatenciesMs = mutableListOf<Long>()
    private val riskResults = mutableListOf<InferenceResult>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Request storage permission
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                REQUEST_CODE
            )
            return
        }

        val scrollView = ScrollView(this)
        val tv = TextView(this).apply {
            setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 12f)
            setPadding(16, 16, 16, 16)
        }
        scrollView.addView(tv)
        setContentView(scrollView)

        tv.text = "Initializing test environment..."

        lifecycleScope.launch {
            try {
                // 1. Initialize MNN engine
                tv.text = "Loading MNN models..."
                engine.init(applicationContext)
                engine.userWeightKg = 70f
                Log.i(TAG, "RGPhaseAEngine initialized")

                // 2. Initialize MediaPipe Pose
                tv.text = "Loading MediaPipe Pose model..."
                initPoseLandmarker()
                Log.i(TAG, "Pose landmarker initialized")

                // 3. Load and extract frames from video
                tv.text = "Extracting frames from video..."
                val frameList = extractVideoFrames(VIDEO_PATH)
                Log.i(TAG, "Extracted ${frameList.size} frames from video")

                if (frameList.isEmpty()) {
                    tv.text = "ERROR: No frames extracted from video.\nCheck video path: $VIDEO_PATH"
                    return@launch
                }

                tv.text = "Extracted ${frameList.size} frames, starting inference pipeline..."

                // 4. Run inference pipeline
                val buffer20 = Array(20) { Array(33) { FloatArray(3) } }
                var bufferIdx = 0
                var warmupDone = false

                for ((idx, bitmap) in frameList.withIndex()) {
                    val frameStart = System.currentTimeMillis()

                    // Pose detection
                    val poseLatencyStart = System.currentTimeMillis()
                    val keypoints = detectPose(bitmap)
                    val poseLatency = System.currentTimeMillis() - poseLatencyStart
                    poseLatenciesMs.add(poseLatency)
                    bitmap.recycle()

                    if (keypoints != null) {
                        poseSuccessFrames++
                    }

                    if (keypoints == null) {
                        totalFrames++
                        tv.text = "Frame $idx/${frameList.size} — No pose detected"
                        continue
                    }

                    // Fill buffer
                    buffer20[bufferIdx] = keypoints
                    bufferIdx = (bufferIdx + 1) % 20

                    totalFrames++

                    // Need at least 20 frames for a full window
                    if (bufferIdx == 0) {
                        val infStart = System.currentTimeMillis()
                        val result = engine.infer(buffer20)
                        val infLatency = System.currentTimeMillis() - infStart
                        val frameTotal = System.currentTimeMillis() - frameStart

                        if (result != null) {
                            inferenceSuccessFrames++
                            riskResults.add(result)
                            inferenceLatenciesMs.add(infLatency)
                            frameLatenciesMs.add(frameTotal)

                            // Update UI every 5 inferences
                            if (inferenceSuccessFrames % 5 == 0) {
                                val riskLabel = result.riskLabelText
                                val kneeL = String.format("%.1f", result.kneeAnglesDeg.first)
                                val kneeR = String.format("%.1f", result.kneeAnglesDeg.second)
                                tv.text = String.format(
                                    "Frame %d/%d | Inferences: %d | Risk: %s | KneeL: %s° KneeR: %s° | Frame latency: %dms",
                                    idx + 1, frameList.size, inferenceSuccessFrames,
                                    riskLabel, kneeL, kneeR, frameTotal
                                )
                            }
                        }
                    }
                }

                // 5. Generate report
                val report = generateReport(frameList.size)
                withContext(Dispatchers.Main) {
                    tv.text = report
                }
                Log.i(TAG, report)
                Log.i(TAG, "=== Video Test Complete ===")

            } catch (e: Exception) {
                Log.e(TAG, "Test error: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    tv.text = "Test failed: ${e.message}\n\n${e.stackTraceToString()}"
                }
            }
        }
    }

    private fun initPoseLandmarker() {
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath("pose_landmarker_lite.task")
                    .build()
            )
            .setRunningMode(RunningMode.IMAGE)
            .setMinPoseDetectionConfidence(0.4f)
            .setMinPosePresenceConfidence(0.4f)
            .setMinTrackingConfidence(0.4f)
            .setErrorListener { e -> Log.e(TAG, "Pose error: $e") }
            .build()
        poseLandmarker = PoseLandmarker.createFromOptions(this, options)
    }

    private fun detectPose(bitmap: Bitmap): Array<FloatArray>? {
        return try {
            val mpImage = com.google.mediapipe.framework.image.BitmapImageBuilder(bitmap).build()
            val result = poseLandmarker?.detect(mpImage)
            mpImage.close()

            if (result != null && result.landmarks().isNotEmpty()) {
                val lms = result.landmarks()[0]
                Array(33) { i ->
                    val lm = lms.getOrNull(i)
                    val vis = lm?.visibility()?.orElse(0f) ?: 0f
                    if (vis < 0.3f) {
                        floatArrayOf(0f, 0f, 0f)
                    } else {
                        floatArrayOf(lm!!.x(), lm!!.y(), lm!!.z())
                    }
                }
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Pose detection error: ${e.message}")
            null
        }
    }

    private fun extractVideoFrames(videoPath: String): List<Bitmap> {
        val results = mutableListOf<Bitmap>()
        val retriever = MediaMetadataRetriever()

        try {
            retriever.setDataSource(videoPath)

            val durationUs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                ?.toIntOrNull() ?: 640
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                ?.toIntOrNull() ?: 480

            Log.i(TAG, "Video meta: durationUs=$durationUs, size=${width}x$height")

            // Extract 1 frame every 100ms (~10fps)
            val intervalUs = 100_000L
            val scaledW = minOf(width, 640)
            val scaledH = minOf(height, 480)

            var frameIndex = 0
            val maxFrames = 200 // safety limit
            val scanDuration = maxOf(durationUs, 10_000_000L) // fallback to 10s if duration is 0
            for (timestampUs in 0L until scanDuration step intervalUs) {
                if (frameIndex >= maxFrames) break
                var bitmap = retriever.getFrameAtTime(timestampUs, MediaMetadataRetriever.OPTION_CLOSEST)
                if (bitmap != null) {
                    Log.i(TAG, "  Frame $frameIndex at ${timestampUs}us: ${bitmap.width}x${bitmap.height} ${bitmap.config}")
                    val argbBitmap = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
                    else bitmap.copy(Bitmap.Config.ARGB_8888, false)
                    if (argbBitmap.width != scaledW || argbBitmap.height != scaledH) {
                        val scaled = Bitmap.createScaledBitmap(
                            argbBitmap,
                            scaledW, scaledH, true
                        )
                        results.add(scaled)
                    } else {
                        results.add(argbBitmap)
                    }
                    frameIndex++

                    if (frameIndex % 20 == 0) {
                        Log.i(TAG, "Extracted $frameIndex frames...")
                    }
                }
            }

            Log.i(TAG, "Total frames extracted: ${results.size}")

        } catch (e: Exception) {
            Log.e(TAG, "Video frame extraction error: ${e.message}", e)
        } finally {
            try { retriever.release() } catch (_: Exception) {}
        }

        return results
    }

    private fun generateReport(totalVideoFrames: Int): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

        fun avg(list: List<Long>) = if (list.isEmpty()) 0L else list.average().toLong()
        fun min(list: List<Long>) = if (list.isEmpty()) 0L else list.minOrNull() ?: 0L
        fun max(list: List<Long>) = if (list.isEmpty()) 0L else list.maxOrNull() ?: 0L
        fun p95(list: List<Long>): Long {
            if (list.isEmpty()) return 0L
            val sorted = list.sorted()
            return sorted[(sorted.size * 0.95).coerceAtMost((sorted.size - 1).toDouble()).toInt()]
        }
        fun p5(list: List<Long>): Long {
            if (list.isEmpty()) return 0L
            val sorted = list.sorted()
            return sorted[(sorted.size * 0.05).coerceAtMost((sorted.size - 1).toDouble()).toInt()]
        }

        val riskDist = mutableMapOf<String, Int>()
        var totalMotionScore = 0f
        var totalConfidence = 0f
        val lKnees = mutableListOf<Float>()
        val rKnees = mutableListOf<Float>()

        for (r in riskResults) {
            riskDist[r.riskLabelText] = riskDist.getOrElse(r.riskLabelText) { 0 } + 1
            totalMotionScore += r.motionScore
            totalConfidence += r.confidence
            lKnees.add(r.kneeAnglesDeg.first)
            rKnees.add(r.kneeAnglesDeg.second)
        }

        val successRate = if (totalFrames > 0) (poseSuccessFrames * 100.0 / totalFrames).toInt() else 0
        val inferenceRate = if (poseSuccessFrames >= 20)
            (inferenceSuccessFrames * 100.0 / (poseSuccessFrames - 19)).toInt()
        else 0

        return """
╔══════════════════════════════════════════════╗
║       RehabGuardian Video Test Report        ║
║       ${dateFormat.format(System.currentTimeMillis())}
╚══════════════════════════════════════════════╝

── Video Info ─────────────────────────────
  Source: $VIDEO_PATH
  Total video frames: $totalVideoFrames
  Extracted frames: ${poseSuccessFrames}
  Pose detection success: $successRate%

── Inference Summary ──────────────────────
  Full windows processed: $inferenceSuccessFrames
  Inference success rate: $inferenceRate%

── Latency Statistics ─────────────────────
  Frame (total):   avg=${avg(frameLatenciesMs)}ms  min=${min(frameLatenciesMs)}ms  max=${max(frameLatenciesMs)}ms  P5=${p5(frameLatenciesMs)}ms  P95=${p95(frameLatenciesMs)}ms
  Pose detection:  avg=${avg(poseLatenciesMs)}ms  min=${min(poseLatenciesMs)}ms  max=${max(poseLatenciesMs)}ms  P5=${p5(poseLatenciesMs)}ms  P95=${p95(poseLatenciesMs)}ms
  Inference:       avg=${avg(inferenceLatenciesMs)}ms  min=${min(inferenceLatenciesMs)}ms  max=${max(inferenceLatenciesMs)}ms  P5=${p5(inferenceLatenciesMs)}ms  P95=${p95(inferenceLatenciesMs)}ms

── Risk Distribution ──────────────────────
${riskDist.entries.sortedBy { it.key }.map { "  ${it.key}: ${it.value} (${it.value * 100 / maxOf(inferenceSuccessFrames, 1)}%)".padEnd(30) }.joinToString("\n")}

── Biomechanics Summary ───────────────────
  Left knee angle:   avg=${lKnees.average().toInt()}°  min=${lKnees.minOrNull()?.toInt()}°  max=${lKnees.maxOrNull()?.toInt()}°
  Right knee angle:  avg=${rKnees.average().toInt()}°  min=${rKnees.minOrNull()?.toInt()}°  max=${rKnees.maxOrNull()?.toInt()}°
  Motion quality:    avg=${totalMotionScore / maxOf(riskResults.size, 1).toFloat().toInt()} / 100
  Model confidence:  avg=${String.format("%.2f", totalConfidence / maxOf(riskResults.size, 1))}

── Risk Threshold Analysis ────────────────
${if (lKnees.isNotEmpty()) "  Low knee flex (L<150°): ${lKnees.count { it < 150f }} frames" else ""}
${if (rKnees.isNotEmpty()) "  Low knee flex (R<150°): ${rKnees.count { it < 150f }} frames" else ""}
${if (lKnees.isNotEmpty() && rKnees.isNotEmpty()) {
    val asymmetryCount = lKnees.mapIndexed { i, l -> kotlin.math.abs(l - rKnees[i]) }
                .count { it > 20f }
    "  High asymmetry (>20°): $asymmetryCount frames"
} else ""}

╔══════════════════════════════════════════════╗
║           Test Complete                     ║
╚══════════════════════════════════════════════╝
""".trimIndent()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            recreate()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        poseLandmarker?.close()
        lifecycleScope.launch {
            withContext(Dispatchers.IO) { engine.release() }
        }
    }
}
