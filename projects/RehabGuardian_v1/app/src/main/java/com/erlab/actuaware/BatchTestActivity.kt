package com.erlab.actuaware

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.erlab.actuaware.analysis.BiomechanicsAnalyzer
import com.erlab.actuaware.inference.MNNInferenceEngine
import com.erlab.actuaware.inference.SlidingWindowBuffer
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Batch test: reads PNG frames from storage, feeds through MediaPipe Pose + MNN pipeline,
 * logs latency and inference results to logcat.
 */
class BatchTestActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "BatchTest"
    }

    private var poseLandmarker: PoseLandmarker? = null
    private lateinit var inferenceEngine: MNNInferenceEngine
    private val poseBuffer = SlidingWindowBuffer<Array<FloatArray>>(5)
    private val riskBuffer = SlidingWindowBuffer<FloatArray>(20)

    private var frameCount = 0
    private var successCount = 0
    private var totalLatencyMs = 0L
    private var minLatencyMs = Long.MAX_VALUE
    private var maxLatencyMs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val tv = TextView(this).apply {
            text = "Running batch test..."
            setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 18f)
        }
        setContentView(tv)

        lifecycleScope.launch {
            // 1. Initialize MNN engine
            inferenceEngine = MNNInferenceEngine(this@BatchTestActivity)
            inferenceEngine.initialize()
            inferenceEngine.userWeightKg = 70f
            Log.i(TAG, "MNN engine initialized")

            // 2. Initialize MediaPipe Pose
            initPoseLandmarker()
            Log.i(TAG, "Pose landmarker initialized")

            // 3. Load frames
            val frames = loadFrames()
            Log.i(TAG, "Loaded ${frames.size} frames")

            tv.text = "Loaded ${frames.size} frames, starting test..."

            // 4. Process frames
            for (frame in frames) {
                val frameStart = System.currentTimeMillis()

                // Pose detection
                val keypoints = detectPose(frame)
                frame.recycle()

                if (keypoints != null) {
                    poseBuffer.addFrame(keypoints)

                    if (poseBuffer.isFull()) {
                        val poseSeq = poseBuffer.getSequence()
                        if (poseSeq != null) {
                            val result = withContext(Dispatchers.IO) {
                                inferenceEngine.infer(poseSeq.toTypedArray() as Array<FloatArray>)
                            }

                            if (result != null) {
                                successCount++
                                val latency = result.latencyMs.toLong()
                                totalLatencyMs += latency
                                minLatencyMs = minOf(minLatencyMs, latency)
                                maxLatencyMs = maxOf(maxLatencyMs, latency)

                                Log.i(TAG, "Frame $frameCount: risk=${"%.3f".format(result.riskScore)} " +
                                    "label=${result.riskLabel} latency=${latency}ms " +
                                    "kneeL=${"%.1f".format(result.jointAngles[6] * 180f / Math.PI.toFloat())}° " +
                                    "kneeR=${"%.1f".format(result.jointAngles[13] * 180f / Math.PI.toFloat())}°")

                                // Also log GRF
                                Log.i(TAG, "  GRF_L_Fz=${"%.2f".format(result.grfLeft[2])} " +
                                    "GRF_R_Fz=${"%.2f".format(result.grfRight[2])}")

                                // Build risk feature for risk buffer
                                val riskFeat = FloatArray(35)
                                result.jointAngles.copyInto(riskFeat, 0)
                                val grf = FloatArray(12)
                                result.grfLeft.copyInto(grf, 0)
                                result.grfRight.copyInto(grf, 6)
                                grf.copyInto(riskFeat, 23)
                                riskBuffer.addFrame(riskFeat)
                            }
                        }
                    }
                }

                frameCount++
                tv.text = "Processing: $frameCount/${frames.size}"
            }

            // 5. Summary
            val avgLatency = if (successCount > 0) totalLatencyMs / successCount else 0L
            val summary = """
                === Batch Test Summary ===
                Total frames: $frameCount
                Successful inferences: $successCount
                Avg latency: ${avgLatency}ms
                Min latency: ${if (minLatencyMs == Long.MAX_VALUE) 0 else minLatencyMs}ms
                Max latency: ${maxLatencyMs}ms
                Success rate: ${"%.1f".format(100f * successCount / maxOf(frameCount, 1))}%
            """.trimIndent()

            Log.i(TAG, summary)
            Log.i(TAG, "=== End Batch Test ===")

            withContext(Dispatchers.Main) {
                tv.text = summary
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
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, _ -> }
            .setErrorListener { e -> Log.e(TAG, "Pose error: $e") }
            .build()
        poseLandmarker = PoseLandmarker.createFromOptions(this, options)
    }

    private fun detectPose(bitmap: Bitmap): Array<FloatArray>? {
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = poseLandmarker?.detect(mpImage)
            mpImage.close()

            if (result != null && result.landmarks().isNotEmpty()) {
                val lms = result.landmarks()[0]
                val kps = Array(33) { i ->
                    val lm = lms.getOrNull(i)
                    val vis = lm?.visibility()?.orElse(0f) ?: 0f
                    if (vis < 0.3f) {
                        floatArrayOf(0f, 0f, 0f)
                    } else {
                        floatArrayOf(lm!!.x(), lm!!.y(), lm!!.z())
                    }
                }
                kps
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Pose detection error: ${e.message}")
            null
        }
    }

    private fun loadFrames(): List<Bitmap> {
        val frameDir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            "RehabGuardian/test_frames"
        )

        val frames = mutableListOf<Bitmap>()

        // Try frame files first
        if (frameDir.exists() && frameDir.isDirectory) {
            val files = frameDir.listFiles()?.filter { it.name.endsWith(".png") || it.name.endsWith(".jpg") }
                ?.sorted() ?: emptyList()
            Log.i(TAG, "Found ${files.size} frame files in $frameDir")

            for (file in files) {
                val bmp = BitmapFactory.decodeFile(file.absolutePath)
                if (bmp != null) {
                    frames.add(bmp)
                }
            }
        }

        // If no frames found, try video extraction
        if (frames.isEmpty()) {
            val videoFile = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                "RehabGuardian/test_human_video.mp4"
            )
            if (videoFile.exists()) {
                Log.i(TAG, "No frames found, extracting from video: ${videoFile.absolutePath}")
                frames.addAll(extractVideoFrames(videoFile, 30))
            }
        }

        // If still no frames, use synthetic ones from assets
        if (frames.isEmpty()) {
            Log.w(TAG, "No external frames found, generating synthetic test data")
            frames.addAll(generateSyntheticFrames(30))
        }

        return frames
    }

    private fun extractVideoFrames(videoFile: File, maxFrames: Int): List<Bitmap> {
        // Since we can't use ffmpeg from Android, generate synthetic frames instead
        // This is a placeholder — we'll use synthetic frames
        return emptyList()
    }

    private fun generateSyntheticFrames(count: Int): List<Bitmap> {
        val frames = mutableListOf<Bitmap>()

        for (i in 0 until count) {
            val bmp = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
            val draw = android.graphics.Canvas(bmp)

            // Black background
            draw.drawRGB(0, 0, 0)

            // Stick figure with motion
            val headX = 320f + 20f * (i.toFloat() / count) * (if (i < count / 2) 1f else -1f)
            val headY = 100f + 30f * kotlin.math.abs(5f - (i % 10)) / 10f

            // Head
            android.graphics.Paint().apply {
                color = 0xFFC89696.toInt()
                isAntiAlias = true
                draw.drawCircle(headX, headY, 15f, this)
            }

            // Body
            paint.apply {
                color = 0xFF6464C8.toInt()
                strokeWidth = 4f
            }
            draw.drawLine(headX, headY + 15f, headX, 250f, paint)

            // Arms
            val armY = 160f + 10f * (i.toFloat() / count)
            paint.color = 0xFFC86464.toInt()
            draw.drawLine(headX, armY, headX - 50f, armY + 40f, paint)
            draw.drawLine(headX, armY, headX + 50f, armY + 40f, paint)

            // Legs
            val legSwing = 20f * (i.toFloat() / count)
            paint.color = 0xFF323296.toInt()
            draw.drawLine(headX, 250f, headX - 30f + legSwing, 350f, paint)
            draw.drawLine(headX, 250f, headX + 30f - legSwing, 350f, paint)

            frames.add(bmp)
        }

        return frames
    }

    private val paint = android.graphics.Paint().apply {
        isAntiAlias = true
        style = android.graphics.Paint.Style.STROKE
    }

    override fun onDestroy() {
        super.onDestroy()
        poseLandmarker?.close()
        inferenceEngine.release()
    }
}
