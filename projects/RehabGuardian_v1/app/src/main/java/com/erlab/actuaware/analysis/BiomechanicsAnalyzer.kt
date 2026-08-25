package com.erlab.actuaware.analysis

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.*

object BiomechanicsAnalyzer {
    private const val LEFT_HIP = 23
    private const val RIGHT_HIP = 24
    private const val LEFT_KNEE = 25
    private const val RIGHT_KNEE = 26
    private const val LEFT_ANKLE = 27
    private const val RIGHT_ANKLE = 28

    data class BiomechanicsMetrics(
        val kneeAngleL: Float,
        val kneeAngleR: Float,
        val kneeAsymmetry: Float,
        val isOverextL: Boolean,
        val isOverextR: Boolean,
        val grfMagnitude: Float,
        val valid: Boolean = true,
        val invalidReason: String? = null
    )

    fun calcAngle3D(a: FloatArray, b: FloatArray, c: FloatArray): Float {
        val ax = a[0] - b[0]; val ay = a[1] - b[1]; val az = a[2] - b[2]
        val cx = c[0] - b[0]; val cy = c[1] - b[1]; val cz = c[2] - b[2]
        val dot = ax * cx + ay * cy + az * cz
        val magA = sqrt(ax * ax + ay * ay + az * az)
        val magC = sqrt(cx * cx + cy * cy + cz * cz)
        if (magA < 1e-6f || magC < 1e-6f) return 180f
        val ratio = (dot / (magA * magC)).toDouble().coerceIn(-1.0, 1.0)
        return Math.toDegrees(ratio).toFloat()
    }

    fun angle3D(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark): Float {
        val ax = a.x() - b.x(); val ay = a.y() - b.y(); val az = a.z() - b.z()
        val cx = c.x() - b.x(); val cy = c.y() - b.y(); val cz = c.z() - b.z()
        val dot = ax * cx + ay * cy + az * cz
        val magA = sqrt(ax * ax + ay * ay + az * az)
        val magC = sqrt(cx * cx + cy * cy + cz * cz)
        if (magA < 1e-6f || magC < 1e-6f) return 180f
        val ratio = (dot / (magA * magC)).toDouble().coerceIn(-1.0, 1.0)
        return Math.toDegrees(ratio).toFloat()
    }

    fun analyze(landmarks: List<NormalizedLandmark>): BiomechanicsMetrics {
        if (landmarks.size < 29) return BiomechanicsMetrics(180f, 180f, 0f, false, false, 0f, valid = false, invalidReason = "landmarks太少")

        // Anatomical validation: check limb lengths
        val leftDx = landmarks[LEFT_KNEE].x() - landmarks[LEFT_HIP].x()
        val leftDy = landmarks[LEFT_KNEE].y() - landmarks[LEFT_HIP].y()
        val leftLimbLen = sqrt(leftDx * leftDx + leftDy * leftDy)

        val rightDx = landmarks[RIGHT_KNEE].x() - landmarks[RIGHT_HIP].x()
        val rightDy = landmarks[RIGHT_KNEE].y() - landmarks[RIGHT_HIP].y()
        val rightLimbLen = sqrt(rightDx * rightDx + rightDy * rightDy)

        // Check for zero-length limbs (landmarks not visible / collapsed)
        if (leftLimbLen < 0.05f || rightLimbLen < 0.05f) {
            return BiomechanicsMetrics(180f, 180f, 0f, false, false, 0f, valid = false, invalidReason = "肢体长度异常")
        }

        // Check limb length symmetry: extreme asymmetry indicates bad detection
        val limbRatio = if (rightLimbLen > 0f) leftLimbLen / rightLimbLen else 1f
        if (limbRatio < 0.5f || limbRatio > 2.0f) {
            return BiomechanicsMetrics(180f, 180f, 0f, false, false, 0f, valid = false, invalidReason = "双侧肢体长度严重不对称")
        }

        val kneeL = angle3D(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        val kneeR = angle3D(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])

        // Clamp knee angles to anatomically valid range: [10°, 190°]
        val clampedKneeL = kotlin.math.max(10f, kotlin.math.min(190f, kneeL))
        val clampedKneeR = kotlin.math.max(10f, kotlin.math.min(190f, kneeR))

        val overextL = clampedKneeL > 175f
        val overextR = clampedKneeR > 175f

        // Replace simplified GRF proxy with a placeholder — real GRF comes from FNO model via MNNInferenceEngine
        // This field is kept for UI display only; actual GRF analysis uses result.grfLeft[2] + result.grfRight[2]
        val grfEst = 0f

        return BiomechanicsMetrics(
            kneeAngleL = clampedKneeL,
            kneeAngleR = clampedKneeR,
            kneeAsymmetry = abs(clampedKneeL - clampedKneeR),
            isOverextL = overextL,
            isOverextR = overextR,
            grfMagnitude = grfEst,
            valid = true
        )
    }

    fun describeAbnormalities(metrics: BiomechanicsMetrics): List<String> {
        val issues = mutableListOf<String>()
        if (metrics.isOverextL) issues.add("左膝关节过伸")
        if (metrics.isOverextR) issues.add("右膝关节过伸")
        if (metrics.kneeAsymmetry > 20f) issues.add("双腿运动不对称（差${metrics.kneeAsymmetry.toInt()}°）")
        if (metrics.kneeAngleL < 90f) issues.add("左膝屈曲过大（${metrics.kneeAngleL.toInt()}°）")
        if (metrics.kneeAngleR < 90f) issues.add("右膝屈曲过大（${metrics.kneeAngleR.toInt()}°）")
        return issues
    }
}