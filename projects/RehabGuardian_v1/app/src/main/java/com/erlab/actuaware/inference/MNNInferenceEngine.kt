package com.erlab.actuaware.inference

import android.content.Context
import android.util.Log
import com.taobao.android.mnn.MNNNetInstance
import kotlinx.coroutines.*

class MNNInferenceEngine(private val context: Context) {

    companion object {
        private const val TAG = "MNNEngine"
        private const val STGCN_INPUT_T = 5
        private const val FNO_INPUT_T = 20
        private const val RISK_INPUT_T = 20

        init {
            try {
                System.loadLibrary("MNN")
                Log.i(TAG, "libMNN.so loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load libMNN.so: ${e.message}")
            }
            try {
                System.loadLibrary("MNN_Express")
                Log.i(TAG, "libMNN_Express.so loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load libMNN_Express.so: ${e.message}")
            }
            try {
                System.loadLibrary("mnncore")
                Log.i(TAG, "libmnncore.so loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load libmnncore.so: ${e.message}")
            }
        }
    }

    private var stgcnNet: MNNNetInstance? = null
    private var fnoNet: MNNNetInstance? = null
    private var riskNet: MNNNetInstance? = null

    // 复用 Session
    private var stgcnSession: MNNNetInstance.Session? = null
    private var fnoSession: MNNNetInstance.Session? = null
    private var riskSession: MNNNetInstance.Session? = null

    private val poseBuffer = SlidingWindowBuffer<Array<FloatArray>>(STGCN_INPUT_T)
    private val bioBuffer = SlidingWindowBuffer<FloatArray>(FNO_INPUT_T)
    private val riskBuffer = SlidingWindowBuffer<FloatArray>(RISK_INPUT_T)

    private var prevJointAngles: FloatArray? = null
    private var prevVel: FloatArray? = null

    private var normStats: NormStats? = null
    private val inferenceDispatcher = Dispatchers.Default

    var userWeightKg = 70f

    data class InferenceResult(
        val jointAngles: FloatArray,
        val grfLeft: FloatArray,
        val grfRight: FloatArray,
        val riskScore: Float,
        val riskLabel: Int,
        val latencyMs: Float,
    )

    fun initialize() {
        try {
            stgcnNet = loadModel("stgcn.mnn")
            fnoNet = loadModel("fno.mnn")
            riskNet = loadModel("risk.mnn")

            stgcnSession = stgcnNet?.createSession(
                MNNNetInstance.Config().apply {
                    numThread = 4
                    forwardType = 0 // FORWARD_CPU
                }
            )
            fnoSession = fnoNet?.createSession(
                MNNNetInstance.Config().apply {
                    numThread = 4
                    forwardType = 0 // FORWARD_CPU
                }
            )
            riskSession = riskNet?.createSession(
                MNNNetInstance.Config().apply {
                    numThread = 2
                }
            )

            val normJson = context.assets.open("norm_stats_PhaseA.json").bufferedReader().readText()
            normStats = NormalizationParams.load(normJson)
            Log.i(TAG, "MNN 推理引擎初始化成功")
        } catch (e: Exception) {
            Log.e(TAG, "初始化失败: ${e.message}", e)
        }
    }

    private fun loadModel(filename: String): MNNNetInstance {
        val tmpFile = java.io.File(context.cacheDir, filename)
        if (!tmpFile.exists()) {
            context.assets.open(filename).use { inp ->
                tmpFile.outputStream().use { out ->
                    inp.copyTo(out)
                }
            }
            Log.i(TAG, "Extracted $filename to ${tmpFile.absolutePath} (size=${tmpFile.length()})")
        } else {
            Log.i(TAG, "Using cached $filename (size=${tmpFile.length()})")
        }
        val net = MNNNetInstance.createFromFile(tmpFile.absolutePath)
        if (net == null) {
            Log.e(TAG, "MNNNetInstance.createFromFile returned null for $filename, path=${tmpFile.absolutePath}, fileExists=${tmpFile.exists()}, fileSize=${tmpFile.length()}")
        } else {
            Log.i(TAG, "Loaded model $filename successfully")
        }
        return net
            ?: throw RuntimeException("无法加载模型 $filename")
    }

    suspend fun infer(keypoints: Array<FloatArray>): InferenceResult? =
        withContext(inferenceDispatcher) {
            val t0 = System.currentTimeMillis()
            poseBuffer.addFrame(keypoints)
            if (!poseBuffer.isFull()) return@withContext null

            val poseSeq = poseBuffer.getSequence() ?: return@withContext null
            val jointAngles = runSTGCN(poseSeq) ?: return@withContext null

            val bioFeat = buildBioFeatureWithDerivatives(jointAngles)
            bioBuffer.addFrame(bioFeat.first())

            val grf = if (bioBuffer.isFull()) {
                runFNO(bioBuffer.getSequence()!!) ?: FloatArray(12)
            } else {
                FloatArray(12)
            }

            val grfLeft = grf.slice(0..5).toFloatArray()
            val grfRight = grf.slice(6..11).toFloatArray()

            normStats?.let { ns ->
                for (i in 0..5) {
                    grfLeft[i] = grfLeft[i] * ns.grfLeftStd[i] + ns.grfLeftMean[i]
                    grfRight[i] = grfRight[i] * ns.grfRightStd[i] + ns.grfRightMean[i]
                }
            }

            val riskFeat = FloatArray(35).also {
                jointAngles.copyInto(it, 0)
                grf.copyInto(it, 23)
            }
            riskBuffer.addFrame(riskFeat)

            val riskOut = if (riskBuffer.isFull()) {
                runRisk(riskBuffer.getSequence()!!) ?: floatArrayOf(0.8f, 0.15f, 0.05f)
            } else {
                floatArrayOf(0.8f, 0.15f, 0.05f)
            }

            val riskLabel = riskOut.indices.maxByOrNull { riskOut[it] } ?: 0
            val riskScore = riskOut[2]

            val (finalLabel, finalScore) = safetyCheck(jointAngles, grfLeft, grfRight, riskLabel, riskScore)

            InferenceResult(
                jointAngles = jointAngles,
                grfLeft = grfLeft,
                grfRight = grfRight,
                riskScore = finalScore,
                riskLabel = finalLabel,
                latencyMs = (System.currentTimeMillis() - t0).toFloat(),
            )
        }

    private fun runSTGCN(poseSeq: List<Array<FloatArray>>): FloatArray? {
        val flat = poseSeq.flatMap { it.toList() }.flatMap { it.toList() }.toFloatArray()
        return try {
            val session = stgcnSession ?: return null
            val input = session.getInput("visual_seq")
            input?.setInputFloatData(flat)
            session.run()
            session.getOutput("joint_angles")?.floatData
        } catch (e: Exception) {
            Log.e(TAG, "ST-GCN 推理失败: ${e.message}")
            null
        }
    }

    private fun runFNO(bioSeq: List<FloatArray>): FloatArray? {
        val flat = bioSeq.flatMap { it.toList() }.toFloatArray()
        return try {
            val session = fnoSession ?: return null
            val input = session.getInput("bio_seq")
            input?.setInputFloatData(flat)
            session.run()
            session.getOutput("grf")?.floatData?.take(12)?.toFloatArray()
        } catch (e: Exception) {
            Log.e(TAG, "FNO 推理失败: ${e.message}")
            null
        }
    }

    private fun runRisk(riskSeq: List<FloatArray>): FloatArray? {
        val flat = riskSeq.flatMap { it.toList() }.toFloatArray()
        return try {
            val session = riskSession ?: return null
            val input = session.getInput("risk_feat")
            input?.setInputFloatData(flat)
            session.run()
            val logits = session.getOutput("logits")?.floatData ?: return null
            val exp = logits.map { kotlin.math.exp(it.toDouble()).toFloat() }
            val sum = exp.sum()
            exp.map { it / sum }.toFloatArray()
        } catch (e: Exception) {
            Log.e(TAG, "Risk 推理失败: ${e.message}")
            null
        }
    }

    private fun buildBioFeatureWithDerivatives(jointAngles: FloatArray): List<FloatArray> {
        val ns = normStats
        val out = mutableListOf<FloatArray>()

        val frame = FloatArray(72)

        if (ns != null) {
            for (i in 0..22) {
                frame[i] = (jointAngles[i] - ns.jointAnglesMean[i]) / ns.jointAnglesStd[i]
            }
        } else {
            jointAngles.copyInto(frame, 0)
        }

        if (prevJointAngles != null) {
            for (i in 0..22) {
                val vel = jointAngles[i] - prevJointAngles!![i]
                frame[23 + i] = if (ns != null) vel / ns.jointAnglesStd[i] else vel
            }
        }

        if (prevVel != null && prevJointAngles != null) {
            for (i in 0..22) {
                val currVel = jointAngles[i] - prevJointAngles!![i]
                val acc = currVel - prevVel!![i]
                frame[46 + i] = if (ns != null) acc / ns.jointAnglesStd[i] else acc
            }
        }

        prevVel = if (prevJointAngles != null) {
            FloatArray(23) { i -> jointAngles[i] - prevJointAngles!![i] }
        } else null
        prevJointAngles = jointAngles.copyOf()

        frame[69] = 0f; frame[70] = 0f; frame[71] = 0f
        return listOf(frame)
    }

    private fun safetyCheck(
        ja: FloatArray,
        grfL: FloatArray,
        grfR: FloatArray,
        modelLabel: Int,
        modelScore: Float
    ): Pair<Int, Float> {
        val kneeL = ja[6]
        val kneeR = ja[13]
        val grf_z = grfL[2] + grfR[2]
        val bw = userWeightKg * 9.81f

        var label = modelLabel
        var score = modelScore

        if (kneeL < Math.toRadians(-5.0).toFloat() || kneeR < Math.toRadians(-5.0).toFloat()) {
            label = 2
            score = maxOf(score, 0.92f)
        }

        if (grf_z > 2.5f * bw) {
            label = maxOf(label, 2)
            score = maxOf(score, 0.85f)
        } else if (grf_z > 1.5f * bw) {
            label = maxOf(label, 1)
            score = maxOf(score, 0.55f)
        }

        return Pair(label, score)
    }

    fun release() {
        try { stgcnSession?.run {} } catch (_: Exception) {}
        try { fnoSession?.run {} } catch (_: Exception) {}
        try { riskSession?.run {} } catch (_: Exception) {}
        stgcnNet?.release()
        fnoNet?.release()
        riskNet?.release()
    }
}