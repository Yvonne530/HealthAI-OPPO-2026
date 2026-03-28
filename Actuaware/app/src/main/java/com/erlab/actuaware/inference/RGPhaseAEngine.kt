package com.healthai.ankle.inference

import android.content.Context
import android.util.Log
import com.alibaba.android.mnn.MNNNetInstance
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

// ── Result type ─────────────────────────────────────────────────────────────

data class InferenceResult(
    val riskScore: Float,                    // prob[2] (high-risk probability)
    val riskLabel: Int,                      // 0=low 1=medium 2=high
    val confidence: Float,                   // softmax max probability
    val jointAngles: FloatArray,             // [23] raw STGCN output
    val grfLeft: FloatArray,                 // [6] left-foot GRF (denormalized if stats available)
    val grfRight: FloatArray,                // [6] right-foot GRF
    val kneeAnglesDeg: Pair<Float, Float>,   // (left, right) in degrees
    val explanations: List<String>,          // human-readable risk reasons
    val motionScore: Float                   // 0-100 composite motion quality
) {
    val riskLabelText: String get() = when (riskLabel) {
        0 -> "低风险"
        1 -> "中风险"
        else -> "高风险"
    }
}

// ── Engine ───────────────────────────────────────────────────────────────────

/**
 * Production on-device PhaseA inference engine.
 *
 * Threading model:
 *   - init() and release() must be called from the same owner thread (usually background).
 *   - infer() is safe to call from any thread; uses AtomicBoolean single-flight guard.
 *   - Last valid result is kept as fallback on exception or dropped frame.
 */
class RGPhaseAEngine {

    var userWeightKg: Float = 70f

    // MNN net instances (created once, reused per-frame)
    private var stgcnNet: MNNNetInstance? = null
    private var fnoNet:   MNNNetInstance? = null
    private var riskNet:  MNNNetInstance? = null

    // Sessions (created once, reused)
    private var stgcnSession: MNNNetInstance.Session? = null
    private var fnoSession:   MNNNetInstance.Session? = null
    private var riskSession:  MNNNetInstance.Session? = null

    // Pre-allocated I/O buffers (no per-frame heap in hot path)
    private val stgcnInputBuf  = FloatArray(1 * 5 * 33 * 3)       // [1,5,33,3]
    private val fnoInputBuf    = FloatArray(1 * 20 * 72)           // [1,20,72]
    private val riskInputBuf   = FloatArray(1 * 20 * 35)           // [1,20,35]
    private val stgcnOutBuf    = FloatArray(23)                    // [1,23]→flat
    private val fnoOutBuf      = FloatArray(1 * 10 * 12)           // [1,10,12]
    private val riskLogitsBuf  = FloatArray(3)                     // [1,3]→flat
    private val riskConfBuf    = FloatArray(1)                     // [1,1]→flat

    private val inferring = AtomicBoolean(false)
    private var lastResult: InferenceResult? = null
    private var normStats: RGNormStats? = null

    // ── init ─────────────────────────────────────────────────────────────

    fun init(context: Context) {
        normStats = RGNormStats(context)

        val config = MNNNetInstance.Config().apply {
            numThread = 4
            forwardType = MNNForwardType.FORWARD_CPU.type
        }

        try {
            val stgcnPath = copyModelPair(context, "stgcn_phaseA.mnn")
            val fnoPath   = copyModelPair(context, "fno_lstm_phaseA.mnn")
            val riskPath  = copyModelPair(context, "risk_phaseA.mnn")

            stgcnNet = MNNNetInstance.createFromFile(stgcnPath)
            fnoNet   = MNNNetInstance.createFromFile(fnoPath)
            riskNet  = MNNNetInstance.createFromFile(riskPath)

            stgcnSession = stgcnNet!!.createSession(config)
            fnoSession   = fnoNet!!.createSession(config)
            riskSession  = riskNet!!.createSession(config)

            Log.i(TAG, "RGPhaseAEngine init OK")
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed — AI disabled: ${e.message}")
            // Graceful degradation: engine stays in null state
            releaseNets()
        }
    }

    // ── infer ─────────────────────────────────────────────────────────────

    /**
     * Run the full STGCN → FNO → Risk pipeline.
     *
     * @param frames20  Rolling 20-frame buffer. Each frame: Array(33) of FloatArray(3).
     * @return InferenceResult, or last valid result if another inference is running,
     *         or null if engine is uninitialized.
     */
    fun infer(frames20: Array<Array<FloatArray>>): InferenceResult? {
        if (stgcnNet == null) return lastResult  // AI disabled path

        // Single-flight guard — drop new request if previous is still running
        if (!inferring.compareAndSet(false, true)) {
            return lastResult
        }

        return try {
            runPipeline(frames20)
        } catch (e: Exception) {
            Log.w(TAG, "Inference error — using last result: ${e.message}")
            lastResult
        } finally {
            inferring.set(false)
        }
    }

    // ── release ───────────────────────────────────────────────────────────

    fun release() {
        releaseNets()
    }

    // ── pipeline ──────────────────────────────────────────────────────────

    private fun runPipeline(frames20: Array<Array<FloatArray>>): InferenceResult {

        // §3.2 STGCN: 4 non-overlapping 5-frame chunks → jaSeq[20][23]
        val chunkResults = Array(4) { FloatArray(23) }
        for (chunk in 0 until 4) {
            val inputFlat = RGFeatureBuilder.sliceStgcnInput(frames20, chunk * 5)
            runStgcn(inputFlat, chunkResults[chunk])
        }
        val jaSeq = RGFeatureBuilder.buildJaSeq(chunkResults)

        // §3.3 Build bio_seq [1,20,72]
        val bioSeq = RGFeatureBuilder.buildBioSeq(jaSeq, frames20)
        System.arraycopy(bioSeq, 0, fnoInputBuf, 0, fnoInputBuf.size)

        // §3.4 FNO: grf_seq [1,10,12], extract grf0 = first timestep [12]
        runFno(fnoInputBuf, fnoOutBuf)
        val grf0 = FloatArray(12) { i -> fnoOutBuf[i] }  // t=0 slice

        // §3.5 Build risk_seq [1,20,35]
        val riskSeq = RGFeatureBuilder.buildRiskSeq(jaSeq, grf0, normStats!!)
        System.arraycopy(riskSeq, 0, riskInputBuf, 0, riskInputBuf.size)

        // §3.6 Risk: logits [1,3], confidence [1,1]
        runRisk(riskInputBuf, riskLogitsBuf, riskConfBuf)
        val probs = softmax(riskLogitsBuf)
        val riskScore  = probs[2]
        val riskLabel  = probs.indices.maxByOrNull { probs[it] } ?: 0
        val confidence = riskConfBuf[0].let { if (it.isFinite()) it else probs.max() }

        // Extract representative joint angles from last chunk
        val finalJa = jaSeq[19].copyOf()

        // GRF split: grf0[0..5]=left, grf0[6..11]=right
        val grfLeftRaw  = grf0.copyOfRange(0, 6)
        val grfRightRaw = grf0.copyOfRange(6, 12)
        val grfLeft  = normStats!!.denormGrfLeft(grfLeftRaw)
        val grfRight = normStats!!.denormGrfRight(grfRightRaw)

        // Knee angles in degrees (JA indices 6=left, 7=right)
        val lKneeDeg = finalJa[RGMarkerMapper.JA_L_KNEE] * RAD_TO_DEG
        val rKneeDeg = finalJa[RGMarkerMapper.JA_R_KNEE] * RAD_TO_DEG

        val explanations = buildExplanations(lKneeDeg, rKneeDeg, grfLeft, grfRight)
        val motionScore  = computeMotionScore(lKneeDeg, rKneeDeg, grfLeft, grfRight)

        InferenceResult(
            riskScore        = riskScore,
            riskLabel        = riskLabel,
            confidence       = confidence,
            jointAngles      = finalJa,
            grfLeft          = grfLeft,
            grfRight         = grfRight,
            kneeAnglesDeg    = Pair(lKneeDeg, rKneeDeg),
            explanations     = explanations,
            motionScore      = motionScore
        ).also { lastResult = it }
    }

    // ── model runners ─────────────────────────────────────────────────────

    private fun runStgcn(input: FloatArray, outBuf: FloatArray) {
        System.arraycopy(input, 0, stgcnInputBuf, 0, input.size)
        val t = stgcnNet!!.getSessionInput(stgcnSession, INPUT_VISUAL_SEQ)
        t.setInputFloatData(stgcnInputBuf)
        stgcnNet!!.runSession(stgcnSession)
        val out = stgcnNet!!.getSessionOutput(stgcnSession, OUTPUT_JOINT_ANGLES)
        val data = out.floatData
        val len = minOf(data.size, outBuf.size)
        System.arraycopy(data, 0, outBuf, 0, len)
    }

    private fun runFno(input: FloatArray, outBuf: FloatArray) {
        val t = fnoNet!!.getSessionInput(fnoSession, INPUT_BIO_SEQ)
        t.setInputFloatData(input)
        fnoNet!!.runSession(fnoSession)
        val out = fnoNet!!.getSessionOutput(fnoSession, OUTPUT_GRF_SEQ)
        val data = out.floatData
        val len = minOf(data.size, outBuf.size)
        System.arraycopy(data, 0, outBuf, 0, len)
    }

    private fun runRisk(input: FloatArray, logitsBuf: FloatArray, confBuf: FloatArray) {
        val t = riskNet!!.getSessionInput(riskSession, INPUT_RISK_SEQ)
        t.setInputFloatData(input)
        riskNet!!.runSession(riskSession)

        val logitsOut = riskNet!!.getSessionOutput(riskSession, OUTPUT_RISK_LOGITS)
        val confOut   = riskNet!!.getSessionOutput(riskSession, OUTPUT_RISK_CONFIDENCE)

        val ld = logitsOut.floatData
        val cd = confOut.floatData
        System.arraycopy(ld, 0, logitsBuf, 0, minOf(ld.size, 3))
        if (cd.isNotEmpty()) confBuf[0] = cd[0]
    }

    // ── feature post-processing ───────────────────────────────────────────

    /** Numerically-stable softmax */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxVal = logits.max()
        val exps = FloatArray(logits.size) { i -> exp((logits[i] - maxVal).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { i -> exps[i] / sum }
    }

    /** §6.1 Risk explanation text */
    private fun buildExplanations(
        lKneeDeg: Float, rKneeDeg: Float,
        grfLeft: FloatArray, grfRight: FloatArray
    ): List<String> {
        val reasons = mutableListOf<String>()
        if (lKneeDeg < 150f) reasons.add("左膝屈曲不足 (${lKneeDeg.toInt()}°)")
        if (rKneeDeg < 150f) reasons.add("右膝屈曲不足 (${rKneeDeg.toInt()}°)")
        val asymmetry = abs(lKneeDeg - rKneeDeg)
        if (asymmetry > 20f) reasons.add("双膝不对称 (${asymmetry.toInt()}°)")
        val lImpact = grfLeft.getOrElse(0) { 0f }
        val rImpact = grfRight.getOrElse(0) { 0f }
        val maxImpact = max(abs(lImpact), abs(rImpact))
        if (maxImpact > IMPACT_THRESHOLD) reasons.add("落地冲击力过大 (${String.format("%.1f", maxImpact)} BW)")
        return reasons
    }

    /** §6.2 Motion score 0–100 */
    private fun computeMotionScore(
        lKneeDeg: Float, rKneeDeg: Float,
        grfLeft: FloatArray, grfRight: FloatArray
    ): Float {
        val symScore    = (1f - minOf(abs(lKneeDeg - rKneeDeg) / 40f, 1f)) * 100f
        val flexScore   = (minOf(lKneeDeg, rKneeDeg) / 180f).coerceIn(0f, 1f) * 100f
        val impact      = max(abs(grfLeft.getOrElse(0) { 0f }), abs(grfRight.getOrElse(0) { 0f }))
        val impactScore = (1f - minOf(impact / 3f, 1f)) * 100f
        return (symScore * 0.3f + flexScore * 0.5f + impactScore * 0.2f).coerceIn(0f, 100f)
    }

    // ── utilities ─────────────────────────────────────────────────────────

    /**
     * Copies both .mnn and .mnn.weight from assets to app-private cache.
     * Returns the path to the .mnn file.
     */
    private fun copyModelPair(context: Context, fileName: String): String {
        val dir = File(context.cacheDir, "mnn_models").also { it.mkdirs() }
        copyAsset(context, "models/$fileName", File(dir, fileName))
        copyAsset(context, "models/$fileName.weight", File(dir, "$fileName.weight"))
        return File(dir, fileName).absolutePath
    }

    private fun copyAsset(context: Context, assetPath: String, dest: File) {
        if (dest.exists() && dest.length() > 0) return   // already cached
        context.assets.open(assetPath).use { input ->
            FileOutputStream(dest).use { output -> input.copyTo(output) }
        }
    }

    private fun releaseNets() {
        runCatching { stgcnSession?.let { stgcnNet?.releaseSession(it) } }
        runCatching { fnoSession?.let   { fnoNet?.releaseSession(it)   } }
        runCatching { riskSession?.let  { riskNet?.releaseSession(it)  } }
        runCatching { stgcnNet?.release() }
        runCatching { fnoNet?.release() }
        runCatching { riskNet?.release() }
        stgcnNet = null; fnoNet = null; riskNet = null
    }

    companion object {
        private const val TAG = "RGPhaseAEngine"

        // Tensor names (must match ONNX export)
        private const val INPUT_VISUAL_SEQ      = "visual_seq"
        private const val OUTPUT_JOINT_ANGLES   = "joint_angles"
        private const val INPUT_BIO_SEQ         = "bio_seq"
        private const val OUTPUT_GRF_SEQ        = "grf_seq"
        private const val INPUT_RISK_SEQ        = "risk_seq"
        private const val OUTPUT_RISK_LOGITS    = "risk_logits"
        private const val OUTPUT_RISK_CONFIDENCE = "risk_confidence"

        private const val RAD_TO_DEG = 180f / Math.PI.toFloat()
        private const val IMPACT_THRESHOLD = 2.5f  // body-weight units
    }
}