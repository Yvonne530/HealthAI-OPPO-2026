package com.rehabguardian.camera

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerOptions
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class PoseProcessor(
    private val context: Context,
    private val onResult: (Array<FloatArray>, PoseLandmarkerResult) -> Unit,
) {
    private var poseLandmarker: PoseLandmarker? = null

    fun initialize() {
        val options = PoseLandmarkerOptions.builder()
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
                val landmarks = result.worldLandmarks()
                if (landmarks.isNotEmpty()) {
                    val kps = Array(33) { i ->
                        val lm = landmarks[0].getOrNull(i)
                        floatArrayOf(lm?.x() ?: 0f, lm?.y() ?: 0f, lm?.z() ?: 0f)
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