// MNNModelInference.kt
// Example Android integration for STGCN, FNO, Risk MNN models

import android.content.Context
import com.alibaba.android.mnn.Interpreter
import kotlin.math.exp
import kotlin.math.ln

/**
 * MNN Inference manager for RehabGuardian health monitoring
 * Three-model pipeline: STGCN → FNO → Risk
 */
class MNNModelInference(context: Context) {
    
    private val assetManager = context.resources.assets
    private var stgcnInterpreter: Interpreter? = null
    private var fnoInterpreter: Interpreter? = null
    private var riskInterpreter: Interpreter? = null
    
    // Calibration statistics (from training)
    private val stgcnStats = NormalizationStats(
        meanJointAngles = floatArrayOf(-0.05f, 0.12f, -0.08f, /* ... 23 total */),
        stdJointAngles = floatArrayOf(0.45f, 0.38f, 0.52f, /* ... 23 total */)
    )
    
    private val riskStats = NormalizationStats(
        meanBioSeq = FloatArray(35) { 0f },  // Load from config
        stdBioSeq = FloatArray(35) { 1f }
    )
    
    init {
        loadModels()
    }
    
    private fun loadModels() {
        try {
            // Load model files from assets or app-private storage
            val stgcnPath = copyAssetToCache("stgcn_phaseA.mnn")
            val fnoPath = copyAssetToCache("fno_lstm_phaseA.mnn")
            val riskPath = copyAssetToCache("risk_phaseA.mnn")
            
            stgcnInterpreter = Interpreter(stgcnPath)
            fnoInterpreter = Interpreter(fnoPath)
            riskInterpreter = Interpreter(riskPath)
            
            // Verify model dimensions (optional, but recommended)
            val stgcnSession = stgcnInterpreter!!.createSession()
            val inTensor = stgcnInterpreter!!.getSessionInput(stgcnSession, "visual_seq")
            logModelInfo("STGCN", stgcnInterpreter!!)
            logModelInfo("FNO", fnoInterpreter!!)
            logModelInfo("Risk", riskInterpreter!!)
        } catch (e: Exception) {
            throw RuntimeException("Failed to load MNN models: ${e.message}")
        }
    }
    
    /**
     * Single inference cycle through all three models
     */
    fun inferRiskLevel(
        mediapipeLandmarks: Array<FloatArray>,  // [5 frames][33 landmarks][3 coords]
        heartRate: Float,
        sleepQuality: Float
    ): RiskPrediction {
        val startTime = System.currentTimeMillis()
        
        // Step 1: STGCN - Extract joint angles from landmarks
        val jointAngles = inferSTGCN(mediapipeLandmarks)
        val timeScan1 = System.currentTimeMillis()
        
        // Step 2: FNO - Predict GRF from biomechanics
        val grfPrediction = inferFNO(jointAngles)
        val timeScan2 = System.currentTimeMillis()
        
        // Step 3: Risk - Classify injury risk
        val riskLogits = inferRisk(jointAngles, heartRate, sleepQuality)
        val totalTime = System.currentTimeMillis() - startTime
        
        // Post-process and return
        return RiskPrediction(
            riskLogits = riskLogits,
            riskClass = riskLogits.indices.maxByOrNull { riskLogits[it] } ?: 0,
            riskLabel = mapRiskClass(riskLogits.indices.maxByOrNull { riskLogits[it] } ?: 0),
            confidence = softmax(riskLogits).maxOrNull() ?: 0f,
            latencyMs = totalTime.toFloat(),
            subLatencies = mapOf(
                "stgcn" to (timeScan1 - startTime).toFloat(),
                "fno" to (timeScan2 - timeScan1).toFloat(),
                "risk" to (totalTime - (timeScan2 - startTime)).toFloat()
            )
        )
    }
    
    /**
     * Stage 1: STGCN (0.77 ms average)
     * Input: [1, 5, 33, 3] MediaPipe pose
     * Output: [1, 23] Joint angles
     */
    private fun inferSTGCN(
        mediapipeLandmarks: Array<FloatArray>  // [5, 33, 3]
    ): FloatArray {
        val session = stgcnInterpreter!!.createSession()
        val inputTensor = stgcnInterpreter!!.getSessionInput(session, "visual_seq")
        
        // Marshal to [1, 5, 33, 3]
        val inputData = FloatArray(1 * 5 * 33 * 3)
        var idx = 0
        for (frame in mediapipeLandmarks) {  // 5 frames
            for (landmark in frame) {  // 33 landmarks
                for (coord in landmark) {  // 3 coords
                    inputData[idx++] = coord
                }
            }
        }
        
        // Infer
        val inputTensorWrapper = Interpreter.Tensor(
            floatArrayOf(1f, 5f, 33f, 3f),  // shape
            inputData
        )
        inputTensor.loadFromHostTensor(inputTensorWrapper)
        stgcnInterpreter!!.runSession(session)
        
        // Extract output
        val outputTensor = stgcnInterpreter!!.getSessionOutput(session, "joint_angles")
        val outputData = FloatArray(23)  // [1, 23] → flatten to [23]
        val outputWrapper = Interpreter.Tensor(
            floatArrayOf(1f, 23f),
            outputData
        )
        outputTensor.loadToHostTensor(outputWrapper)
        
        stgcnInterpreter!!.releaseSession(session)
        return outputData
    }
    
    /**
     * Stage 2: FNO (0.48 ms average)
     * Input: [1, 20, 72] Biomechanics sequence
     * Output: [1, 10, 12] GRF prediction
     */
    private fun inferFNO(jointAngles: FloatArray): Array<FloatArray> {
        val session = fnoInterpreter!!.createSession()
        
        // Construct bio_seq from joint angles + velocity + acceleration history
        // This would pull from a rolling buffer in real implementation
        val bioSeq = constructBioSequence(jointAngles)  // [20, 72]
        
        val inputTensor = fnoInterpreter!!.getSessionInput(session, "bio_seq")
        val inputData = FloatArray(1 * 20 * 72)
        var idx = 0
        for (frame in bioSeq) {
            for (feature in frame) {
                inputData[idx++] = feature
            }
        }
        
        val inputTensorWrapper = Interpreter.Tensor(
            floatArrayOf(1f, 20f, 72f),
            inputData
        )
        inputTensor.loadFromHostTensor(inputTensorWrapper)
        fnoInterpreter!!.runSession(session)
        
        // Extract output [1, 10, 12]
        val outputTensor = fnoInterpreter!!.getSessionOutput(session, "grf_seq")
        val outputData = FloatArray(1 * 10 * 12)
        val outputWrapper = Interpreter.Tensor(
            floatArrayOf(1f, 10f, 12f),
            outputData
        )
        outputTensor.loadToHostTensor(outputWrapper)
        
        // Reshape to [10, 12] for convenience
        val grfPred = Array(10) { i ->
            FloatArray(12) { j ->
                outputData[i * 12 + j]
            }
        }
        
        fnoInterpreter!!.releaseSession(session)
        return grfPred
    }
    
    /**
     * Stage 3: Risk (0.09 ms average)
     * Input: [1, 20, 35] Risk features + HR + sleep
     * Output: [1, 3] Logits → softmax → probabilities
     */
    private fun inferRisk(
        jointAngles: FloatArray,
        heartRate: Float,
        sleepQuality: Float
    ): FloatArray {
        val session = riskInterpreter!!.createSession()
        
        // Construct risk_seq [20, 35]
        val riskSeq = constructRiskSequence(jointAngles, heartRate, sleepQuality)
        
        val inputTensor = riskInterpreter!!.getSessionInput(session, "risk_seq")
        val inputData = FloatArray(1 * 20 * 35)
        var idx = 0
        for (frame in riskSeq) {
            for (feature in frame) {
                inputData[idx++] = feature
            }
        }
        
        val inputTensorWrapper = Interpreter.Tensor(
            floatArrayOf(1f, 20f, 35f),
            inputData
        )
        inputTensor.loadFromHostTensor(inputTensorWrapper)
        riskInterpreter!!.runSession(session)
        
        // Extract both outputs
        val logitsTensor = riskInterpreter!!.getSessionOutput(session, "risk_logits")
        val logitsData = FloatArray(3)
        val logitsTensorWrapper = Interpreter.Tensor(
            floatArrayOf(1f, 3f),
            logitsData
        )
        logitsTensor.loadToHostTensor(logitsTensorWrapper)
        
        riskInterpreter!!.releaseSession(session)
        return logitsData
    }
    
    /**
     * Construct biomechanics sequence [20, 72]
     * In production, these would be maintained in rolling buffers
     */
    private fun constructBioSequence(jointAngles: FloatArray): Array<FloatArray> {
        // Features: ja [23] + vel [23] + acc [23] + com [3] = 72
        // Placeholder - in real implementation, pull from motion history buffers
        val bioSeq = Array(20) { FloatArray(72) }
        for (t in 0 until 20) {
            // Last frame (most recent)
            if (t == 19) {
                System.arraycopy(jointAngles, 0, bioSeq[t], 0, 23)
                // Velocity and acceleration would come from history
            }
        }
        return bioSeq
    }
    
    /**
     * Construct risk input [20, 35]
     * Selected joint features + HR + sleep
     */
    private fun constructRiskSequence(
        jointAngles: FloatArray,
        heartRate: Float,
        sleepQuality: Float
    ): Array<FloatArray> {
        val riskSeq = Array(20) { FloatArray(35) }
        for (t in 0 until 20) {
            if (t == 19) {
                // Take subset of joint angles (35 dims)
                System.arraycopy(jointAngles, 0, riskSeq[t], 0, minOf(23, 35))
                // HR and sleep in later dims
                if (riskSeq[t].size > 33) {
                    riskSeq[t][33] = heartRate
                    riskSeq[t][34] = sleepQuality
                }
            }
        }
        return riskSeq
    }
    
    // ============ Utilities ============
    
    private fun softmax(logits: FloatArray): FloatArray {
        // Numerically stable softmax
        val maxLogit = logits.maxOrNull() ?: 0f
        val shifted = logits.map { exp(it - maxLogit) }
        val sum = shifted.sum()
        return shifted.map { it / sum }.toFloatArray()
    }
    
    private fun mapRiskClass(classIdx: Int): String {
        return when (classIdx) {
            0 -> "LOW"
            1 -> "MEDIUM"
            2 -> "HIGH"
            else -> "UNKNOWN"
        }
    }
    
    private fun copyAssetToCache(assetName: String): String {
        val cacheFile = java.io.File(
            assetManager.openFd(assetName).fileDescriptor.toString(),
            assetName
        )
        return cacheFile.absolutePath
    }
    
    private fun logModelInfo(name: String, interpreter: Interpreter) {
        val session = interpreter.createSession()
        println("$name model loaded successfully")
        interpreter.releaseSession(session)
    }
    
    fun close() {
        stgcnInterpreter?.close()
        fnoInterpreter?.close()
        riskInterpreter?.close()
    }
    
    // ============ Data Classes ============
    
    data class RiskPrediction(
        val riskLogits: FloatArray,      // [3] raw scores
        val riskClass: Int,               // 0: LOW, 1: MEDIUM, 2: HIGH
        val riskLabel: String,            // "LOW", "MEDIUM", "HIGH"
        val confidence: Float,            // max probability
        val latencyMs: Float,             // total inference time
        val subLatencies: Map<String, Float>  // per-model breakdown
    )
    
    data class NormalizationStats(
        val meanJointAngles: FloatArray? = null,
        val stdJointAngles: FloatArray? = null,
        val meanBioSeq: FloatArray? = null,
        val stdBioSeq: FloatArray? = null
    )
}

// ============ Usage Example ============

class HealthMonitorActivity : AppCompatActivity() {
    
    private lateinit var mnnInference: MNNModelInference
    private val mediapipePoseBuffer = mutableListOf<Array<FloatArray>>()  // 5-frame buffer
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize MNN models
        mnnInference = MNNModelInference(this)
    }
    
    fun onPoseDetected(landmarks: Array<FloatArray>, heartRate: Float) {
        // Maintain 5-frame sliding window
        mediapipePoseBuffer.add(landmarks)
        if (mediapipePoseBuffer.size > 5) {
            mediapipePoseBuffer.removeAt(0)
        }
        
        // Once we have 5 frames, run inference
        if (mediapipePoseBuffer.size == 5) {
            val prediction = mnnInference.inferRiskLevel(
                mediapipePoseBuffer.toTypedArray(),
                heartRate,
                sleepQuality = 0.8f  // from device sensors
            )
            
            // Handle prediction
            when (prediction.riskClass) {
                0 -> showStatus("✅ LOW RISK", android.graphics.Color.GREEN)
                1 -> showStatus("⚠️ MEDIUM RISK", android.graphics.Color.YELLOW)
                2 -> showStatus("🚨 HIGH RISK", android.graphics.Color.RED)
            }
            
            // Log performance
            Log.d("MNN", "Risk inference: ${prediction.latencyMs}ms (STGCN: ${prediction.subLatencies["stgcn"]}ms)")
        }
    }
    
    private fun showStatus(message: String, color: Int) {
        findViewById<TextView>(R.id.statusText).apply {
            text = message
            setTextColor(color)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        mnnInference.close()
    }
}
