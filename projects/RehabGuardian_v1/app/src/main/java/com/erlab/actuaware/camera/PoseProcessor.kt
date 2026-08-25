package com.erlab.actuaware.camera

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class PoseProcessor(
    private val context: Context,
    private val onResult: (Array<FloatArray>, PoseLandmarkerResult) -> Unit,
) {
    private var poseLandmarker: PoseLandmarker? = null

    fun initialize() {
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath("pose_landmarker_lite.task")
                    .build()
            )
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, _ ->
                val landmarks = result.landmarks()
                if (landmarks.isNotEmpty()) {
                    val lms = landmarks[0]
                    val criticalIndices = intArrayOf(23, 24, 25, 26, 27, 28) // hips, knees, ankles
                    var allCriticalValid = true
                    for (ci in criticalIndices) {
                        val lmCi = lms.getOrNull(ci)
                        val visCi = lmCi?.visibility()?.orElse(0f) ?: 0f
                        if (visCi < 0.3f) {
                            allCriticalValid = false
                            break
                        }
                    }
                    if (!allCriticalValid) return@setResultListener

                    val kps = Array(33) { i ->
                        val lm = lms.getOrNull(i)
                        // Filter low-confidence keypoints: if visibility < 0.3, zero-fill
                        val vis = lm?.visibility()?.orElse(0f) ?: 0f
                        if (vis < 0.3f) {
                            floatArrayOf(0f, 0f, 0f)
                        } else {
                            floatArrayOf(lm!!.x(), lm!!.y(), lm!!.z())
                        }
                    }
                    onResult(kps, result)
                }
            }
            .build()
        poseLandmarker = PoseLandmarker.createFromOptions(context, options)
    }

    fun processFrame(bitmap: Bitmap, timestampMs: Long) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        poseLandmarker?.detectAsync(mpImage, timestampMs)
    }

    fun release() {
        poseLandmarker?.close()
    }
}