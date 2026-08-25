package com.rehabguardian.analysis

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
        val grfMagnitude: Float
    )

    fun angle3D(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark): Float {
        val ax = a.x() - b.x(); val ay = a.y() - b.y(); val az = a.z() - b.z()
        val cx = c.x() - b.x(); val cy = c.y() - b.y(); val cz = c.z() - b.z()
        val dot = ax * cx + ay * cy + az * cz
        val magA = sqrt(ax * ax + ay * ay + az * az)
        val magC = sqrt(cx * cx + cy * cy + cz * cz)
        if (magA < 1e-6f || magC < 1e-6f) return 180f
        return Math.toDegrees(acos((dot / (magA * magC)).coerceIn(-1.0, 1.0))).toFloat()
    }

    fun analyze(landmarks: List<NormalizedLandmark>): BiomechanicsMetrics {
        if (landmarks.size < 29) return BiomechanicsMetrics(180f, 180f, 0f, false, false, 0f)

        val kneeL = angle3D(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        val kneeR = angle3D(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])

        val overextL = kneeL > 175f
        val overextR = kneeR > 175f

        val hipY = (landmarks[LEFT_HIP].y() + landmarks[RIGHT_HIP].y()) / 2f
        val ankleY = (landmarks[LEFT_ANKLE].y() + landmarks[RIGHT_ANKLE].y()) / 2f
        val grfEst = (1f - (hipY - ankleY)).coerceIn(0f, 1f)

        return BiomechanicsMetrics(
            kneeAngleL = kneeL,
            kneeAngleR = kneeR,
            kneeAsymmetry = abs(kneeL - kneeR),
            isOverextL = overextL,
            isOverextR = overextR,
            grfMagnitude = grfEst
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