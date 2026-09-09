package com.erlab.actuaware.inference

import org.json.JSONObject

data class NormStats(
    val jointAnglesMean: FloatArray,
    val jointAnglesStd: FloatArray,
    val grfLeftMean: FloatArray,
    val grfLeftStd: FloatArray,
    val grfRightMean: FloatArray,
    val grfRightStd: FloatArray,
    val comMean: FloatArray,
    val comStd: FloatArray,
)

object NormalizationParams {
    fun load(json: String): NormStats {
        val obj = JSONObject(json)
        fun arr(key: String) = FloatArray(obj.getJSONArray(key).length()) { i ->
            obj.getJSONArray(key).getDouble(i).toFloat()
        }
        return NormStats(
            arr("joint_angles_mean"),
            arr("joint_angles_std"),
            arr("grf_left_mean"),
            arr("grf_left_std"),
            arr("grf_right_mean"),
            arr("grf_right_std"),
            arr("com_mean"),
            arr("com_std"),
        )
    }
}