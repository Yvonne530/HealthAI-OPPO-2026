package com.healthai.ankle.inference

import android.content.Context
import android.util.Log
import com.taobao.android.mnn.MNNNetInstance
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

// ── Result type ──────────────────────────────────────────────────────────────

data class InferenceResult(
    val riskScore: Float,
    val riskLabel: Int,
    val confidence: Float,
    val jointAngles: FloatArray,
    val grfLeft: FloatArray,             // [6] left GRF (denormalized)
    val grfRight: FloatArray,            // [6] right GRF
    val grfFutureFlat: FloatArray,       // [10*12] all 10 future-frame GRF predictions
    val kneeAnglesDeg: Pair<Float, Float>,
    val explanation: RiskExplainer.Explanation,
    val motionScore: Float
) {
    val riskLabelText: String get() = when (riskLabel) {
        0 -> "低风险"; 1 -> "中风险"; else -> "高风险"
    }
    // Convenience: future GRF as 2D array [10][12]
    fun grfFuture2D(): Array<FloatArray> = Array(10) { t ->
        FloatArray(12) { j -> grfFutureFlat.getOrElse(t * 12 + j) { 0f } }
    }
}

// ── Engine ───────────────────────────────────────────────────────────────────

/**
 * Production on-device PhaseA inference engine.
 *
 * Mixed-precision strategy (engineering decision, not default FP16 everywhere):
 *   STGCN → uses whatever .mnn was exported (FP32 recommended for accuracy)
 *   FNO   → FP16 OK (output smoothed over 10 frames)
 *   Risk  → FP16 OK (classification, tolerant to small errors)
 *
 * Threading: init()/release() on background thread.
 *            infer() safe from any thread — AtomicBoolean single-flight guard.
 *            Last valid result kept as fallback on exception.
 */
class RGPhaseAEngine {

    var userWeightKg: Float = 70f

    // MNN net handles
    private var stgcnNet: MNNNetInstance? = null
    private var fnoNet:   MNNNetInstance? = null
    private var riskNet:  MNNNetInstance? = null

    // Sessions (created once, reused every frame — no per-frame alloc)
    private var stgcnSess: MNNNetInstance.Session? = null
    private var fnoSess:   MNNNetInstance.Session? = null
    private var riskSess:  MNNNetInstance.Session? = null

    // Pre-allocated I/O buffers
    private val stgcnInBuf  = FloatArray(1 * 5 * 33 * 3)
    private val fnoInBuf    = FloatArray(1 * 20 * 72)
    private val riskInBuf   = FloatArray(1 * 20 * 35)
    private val stgcnOutBuf = FloatArray(23)
    private val fnoOutBuf   = FloatArray(1 * 10 * 12)
    private val logitsBuf   = FloatArray(3)
    private val confBuf     = FloatArray(1)

    private val inferring = AtomicBoolean(false)
    @Volatile var lastResult: InferenceResult? = null
    private var normStats: RGNormStats? = null

    // ── init ─────────────────────────────────────────────────────────────

    fun init(context: Context) {
        normStats = RGNormStats(context)

        // CPU forward, 4 threads — optimal for single-user real-time on Reno15 Pro
        val cfg = MNNNetInstance.Config().apply {
            numThread   = 4
            forwardType = 0   // FORWARD_CPU
        }

        runCatching {
            val stgcnPath = copyModelPair(context, "stgcn_phaseA.mnn")
            val fnoPath   = copyModelPair(context, "fno_lstm_phaseA.mnn")
            val riskPath  = copyModelPair(context, "risk_phaseA.mnn")

            stgcnNet  = MNNNetInstance.createFromFile(stgcnPath)
            fnoNet    = MNNNetInstance.createFromFile(fnoPath)
            riskNet   = MNNNetInstance.createFromFile(riskPath)

            stgcnSess = stgcnNet!!.createSession(cfg)
            fnoSess   = fnoNet!!.createSession(cfg)
            riskSess  = riskNet!!.createSession(cfg)

            Log.i(TAG, "RGPhaseAEngine ready ✓  [STGCN|FNO|Risk loaded]")
        }.onFailure { e ->
            Log.e(TAG, "Model load failed — AI disabled gracefully: ${e.message}")
            releaseNets()
        }
    }

    // ── infer ─────────────────────────────────────────────────────────────

    fun infer(frames20: Array<Array<FloatArray>>): InferenceResult? {
        if (stgcnNet == null) return lastResult
        if (!inferring.compareAndSet(false, true)) return lastResult   // single-flight drop
        return try {
            runPipeline(frames20)
        } catch (e: Exception) {
            Log.w(TAG, "Inference error, using last result: ${e.message}")
            lastResult
        } finally {
            inferring.set(false)
        }
    }

    fun release() = releaseNets()

    // ── pipeline ──────────────────────────────────────────────────────────

    private fun runPipeline(frames20: Array<Array<FloatArray>>): InferenceResult {

        // §3.2  4 × STGCN (5-frame chunks) → jaSeq[20][23]
        val chunks = Array(4) { FloatArray(23) }
        for (c in 0 until 4) {
            runStgcn(RGFeatureBuilder.sliceStgcnInput(frames20, c * 5), chunks[c])
        }
        val jaSeq = RGFeatureBuilder.buildJaSeq(chunks)

        // §3.3  bio_seq [1,20,72]
        val bioFlat = RGFeatureBuilder.buildBioSeq(jaSeq, frames20)
        System.arraycopy(bioFlat, 0, fnoInBuf, 0, fnoInBuf.size)

        // §3.4  FNO → grf_seq [1,10,12]
        runFno(fnoInBuf, fnoOutBuf)
        val grf0         = fnoOutBuf.copyOfRange(0, 12)          // t=0 slice [12]
        val grfFutureAll = fnoOutBuf.copyOf()                     // all 10 frames [120]

        // §3.5  risk_seq [1,20,35]
        val riskFlat = RGFeatureBuilder.buildRiskSeq(jaSeq, grf0, normStats!!)
        System.arraycopy(riskFlat, 0, riskInBuf, 0, riskInBuf.size)

        // §3.6  Risk → logits [3] + confidence [1]
        runRisk(riskInBuf, logitsBuf, confBuf)
        val probs      = softmax(logitsBuf)
        val riskScore  = probs[2]
        val riskLabel  = probs.indices.maxByOrNull { probs[it] } ?: 0
        val confidence = confBuf[0].let { if (it.isFinite()) it else probs.max() }

        val finalJa   = jaSeq[19].copyOf()
        val grfLeft   = normStats!!.denormGrfLeft(grf0.copyOfRange(0, 6))
        val grfRight  = normStats!!.denormGrfRight(grf0.copyOfRange(6, 12))
        val lKnee     = finalJa[RGMarkerMapper.JA_L_KNEE] * RAD_TO_DEG
        val rKnee     = finalJa[RGMarkerMapper.JA_R_KNEE] * RAD_TO_DEG
        val motScore  = computeMotionScore(lKnee, rKnee, grfLeft, grfRight)

        // Explainability layer — what the AI actually found
        val explanation = RiskExplainer.explain(
            lKnee, rKnee, grfLeft, grfRight, grfFutureAll, riskScore, motScore
        )

        return InferenceResult(
            riskScore      = riskScore,
            riskLabel      = riskLabel,
            confidence     = confidence,
            jointAngles    = finalJa,
            grfLeft        = grfLeft,
            grfRight       = grfRight,
            grfFutureFlat  = grfFutureAll,
            kneeAnglesDeg  = lKnee to rKnee,
            explanation    = explanation,
            motionScore    = motScore
        ).also { lastResult = it }
    }

    // ── model runners ─────────────────────────────────────────────────────

    private fun runStgcn(inputFlat: FloatArray, outBuf: FloatArray) {
        System.arraycopy(inputFlat, 0, stgcnInBuf, 0, inputFlat.size)
        stgcnSess!!.getInput(INPUT_VISUAL_SEQ).setInputFloatData(stgcnInBuf)
        stgcnSess!!.run()
        val raw = stgcnSess!!.getOutput(OUTPUT_JOINT_ANGLES).floatData
        System.arraycopy(raw, 0, outBuf, 0, minOf(raw.size, outBuf.size))
    }

    private fun runFno(inputFlat: FloatArray, outBuf: FloatArray) {
        fnoSess!!.getInput(INPUT_BIO_SEQ).setInputFloatData(inputFlat)
        fnoSess!!.run()
        val raw = fnoSess!!.getOutput(OUTPUT_GRF_SEQ).floatData
        System.arraycopy(raw, 0, outBuf, 0, minOf(raw.size, outBuf.size))
    }

    private fun runRisk(inputFlat: FloatArray, logitsOut: FloatArray, confOut: FloatArray) {
        riskSess!!.getInput(INPUT_RISK_SEQ).setInputFloatData(inputFlat)
        riskSess!!.run()
        val ld = riskSess!!.getOutput(OUTPUT_RISK_LOGITS).floatData
        System.arraycopy(ld, 0, logitsOut, 0, minOf(ld.size, 3))
        // risk_confidence is optional — degrade to softmax max when missing
        riskSess!!.getOutput(OUTPUT_RISK_CONF)?.let { cd ->
            val c = cd.floatData
            if (c.isNotEmpty()) confOut[0] = c[0]
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private fun softmax(logits: FloatArray): FloatArray {
        val mx = logits.max()
        val e  = FloatArray(logits.size) { i -> exp((logits[i] - mx).toDouble()).toFloat() }
        val s  = e.sum()
        return FloatArray(e.size) { i -> e[i] / s }
    }

    private fun computeMotionScore(lK: Float, rK: Float, grfL: FloatArray, grfR: FloatArray): Float {
        val sym    = (1f - minOf(abs(lK - rK) / 40f, 1f)) * 100f
        val flex   = (minOf(lK, rK) / 180f).coerceIn(0f, 1f) * 100f
        val impact = max(abs(grfL.getOrElse(0) { 0f }), abs(grfR.getOrElse(0) { 0f }))
        val imp    = (1f - minOf(impact / 3f, 1f)) * 100f
        return (sym * 0.30f + flex * 0.50f + imp * 0.20f).coerceIn(0f, 100f)
    }

    private fun copyModelPair(context: Context, fileName: String): String {
        val dir = File(context.cacheDir, "mnn_models").also { it.mkdirs() }
        copyAsset(context, "models/$fileName",        File(dir, fileName))
        copyAsset(context, "models/$fileName.weight", File(dir, "$fileName.weight"))
        return File(dir, fileName).absolutePath
    }

    private fun copyAsset(ctx: Context, asset: String, dest: File) {
        if (dest.exists() && dest.length() > 0) return
        runCatching {
            ctx.assets.open(asset).use { i -> FileOutputStream(dest).use { o -> i.copyTo(o) } }
        }.onFailure { Log.w(TAG, "Asset copy failed: $asset — ${it.message}") }
    }

    private fun releaseNets() {
        runCatching { stgcnSess?.release() }; runCatching { fnoSess?.release() }; runCatching { riskSess?.release() }
        runCatching { stgcnNet?.release()  }; runCatching { fnoNet?.release()  }; runCatching { riskNet?.release()  }
        stgcnNet = null; fnoNet = null; riskNet = null
        stgcnSess = null; fnoSess = null; riskSess = null
    }

    companion object {
        private const val TAG                  = "RGPhaseAEngine"
        private const val INPUT_VISUAL_SEQ     = "visual_seq"
        private const val OUTPUT_JOINT_ANGLES  = "joint_angles"
        private const val INPUT_BIO_SEQ        = "bio_seq"
        private const val OUTPUT_GRF_SEQ       = "grf_seq"
        private const val INPUT_RISK_SEQ       = "risk_seq"
        private const val OUTPUT_RISK_LOGITS   = "risk_logits"
        private const val OUTPUT_RISK_CONF     = "risk_confidence"
        private const val RAD_TO_DEG           = 180f / Math.PI.toFloat()
    }
}
