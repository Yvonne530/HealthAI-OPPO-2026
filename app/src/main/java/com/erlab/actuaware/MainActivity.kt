package com.rehabguardian

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.rehabguardian.analysis.BiomechanicsAnalyzer
import com.rehabguardian.analysis.RiskStateMachine
import com.rehabguardian.camera.CameraManager
import com.rehabguardian.camera.PoseProcessor
import com.rehabguardian.data.FrameData
import com.rehabguardian.inference.MNNInferenceEngine
import com.rehabguardian.ui.RiskRenderer
import com.rehabguardian.ui.SessionManager
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var riskRenderer: RiskRenderer
    private lateinit var btnRecord: Button
    private lateinit var btnFlip: Button
    private lateinit var tvStatus: TextView

    private lateinit var cameraManager: CameraManager
    private lateinit var poseProcessor: PoseProcessor
    private lateinit var inferenceEngine: MNNInferenceEngine
    private lateinit var sessionManager: SessionManager
    private lateinit var riskStateMachine: RiskStateMachine

    private var frameCount = 0
    private var lastFpsUpdate = 0L
    private var currentFps = 30f

    private val processingScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val isProcessing = AtomicBoolean(false)

    private var lastLandmarks = listOf<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>()
    private var lastKneeAngleL = 180f
    private var lastKneeAngleR = 180f
    private var lastRiskScore = 0f
    private var lastRiskState = RiskStateMachine.State.SAFE
    private var lastLatency = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        requestPermissions()
        initModules()
        setupListeners()
        startProcessing()
    }

    private fun initViews() {
        previewView = findViewById(R.id.previewView)
        riskRenderer = findViewById(R.id.riskRenderer)
        btnRecord = findViewById(R.id.btnRecord)
        btnFlip = findViewById(R.id.btnFlip)
        tvStatus = findViewById(R.id.tvStatus)
        riskRenderer.setBackgroundColor(android.graphics.Color.TRANSPARENT)
    }

    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
        val needRequest = permissions.any {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needRequest) {
            ActivityCompat.requestPermissions(this, permissions, 100)
        }
    }

    private fun initModules() {
        sessionManager = SessionManager(this)
        riskStateMachine = RiskStateMachine()

        inferenceEngine = MNNInferenceEngine(this)
        inferenceEngine.userWeightKg = 70f

        poseProcessor = PoseProcessor(this) { keypoints, result ->
            val landmarks = result.landmarks()
            if (landmarks.isNotEmpty()) {
                lastLandmarks = landmarks[0]
                val metrics = BiomechanicsAnalyzer.analyze(landmarks[0])
                lastKneeAngleL = metrics.kneeAngleL
                lastKneeAngleR = metrics.kneeAngleR

                processingScope.launch {
                    if (isProcessing.compareAndSet(false, true)) {
                        try {
                            val inferenceResult = inferenceEngine.infer(keypoints)
                            inferenceResult?.let { res ->
                                withContext(Dispatchers.Main) {
                                    val state = riskStateMachine.update(
                                        res.riskScore,
                                        confidence = 0.85f,
                                        heartRate = 75f,
                                        sleepScore = 80f
                                    )
                                    lastRiskScore = res.riskScore
                                    lastRiskState = state
                                    lastLatency = res.latencyMs

                                    if (sessionManager.isRecording) {
                                        val frameData = FrameData(
                                            timestampMs = System.currentTimeMillis(),
                                            keypoints = keypoints,
                                            riskScore = res.riskScore,
                                            riskLabel = res.riskLabel,
                                            grfLeft = res.grfLeft,
                                            grfRight = res.grfRight,
                                            kneeAngleL = metrics.kneeAngleL,
                                            kneeAngleR = metrics.kneeAngleR
                                        )
                                        sessionManager.addFrame(frameData)
                                    }
                                    updateUI()
                                }
                            }
                        } catch (e: Exception) {
                            e.printStackTrace()
                        } finally {
                            isProcessing.set(false)
                        }
                    }
                }
            }
        }

        cameraManager = CameraManager(
            context = this,
            lifecycleOwner = this,
            previewView = previewView
        ) { bitmap, timestampMs ->
            frameCount++
            val now = System.currentTimeMillis()
            if (now - lastFpsUpdate > 1000) {
                currentFps = frameCount / ((now - lastFpsUpdate) / 1000f)
                frameCount = 0
                lastFpsUpdate = now
                sessionManager.setFps(currentFps)
                runOnUiThread {
                    tvStatus.text = "FPS: ${"%.1f".format(currentFps)} | Latency: ${"%.1f".format(lastLatency)}ms"
                }
            }
            poseProcessor.processFrame(bitmap, timestampMs)
        }

        lifecycleScope.launch(Dispatchers.IO) {
            inferenceEngine.initialize()
            poseProcessor.initialize()
            withContext(Dispatchers.Main) {
                Toast.makeText(this@MainActivity, "AI 引擎初始化完成", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setupListeners() {
        btnRecord.setOnClickListener {
            if (sessionManager.isRecording) {
                val session = sessionManager.stopSession()
                btnRecord.text = "开始录制"
                Toast.makeText(this, "会话已保存", Toast.LENGTH_SHORT).show()
                session?.let {
                    val report = sessionManager.generateReport(it)
                    showReportDialog(report)
                }
            } else {
                sessionManager.startSession()
                btnRecord.text = "停止录制"
                Toast.makeText(this, "开始录制运动分析", Toast.LENGTH_SHORT).show()
            }
        }

        btnFlip.setOnClickListener {
            cameraManager.flipCamera()
        }
    }

    private fun updateUI() {
        // 注意：这里需要将 lastLandmarks 转换为 PoseLandmarkerResult 对象，实际使用中需要根据 MediaPipe API 构造
        // 简化处理：我们使用一个临时的空结果，或者直接传递 landmarks 给渲染器。这里仅示意。
        // 在真实实现中，RiskRenderer 应该接收 landmarks 而不是 result，可以修改 RiskRenderer 接口。
        // 此处省略具体转换，你可以根据实际需求调整。
        // 由于 RiskRenderer 需要 PoseLandmarkerResult，你可以修改 RiskRenderer 直接接收 landmarks。
    }

    private fun showReportDialog(report: String) {
        AlertDialog.Builder(this)
            .setTitle("运动分析报告")
            .setMessage(report)
            .setPositiveButton("确定", null)
            .setNeutralButton("分享") { _, _ ->
                val shareIntent = android.content.Intent().apply {
                    action = android.content.Intent.ACTION_SEND
                    putExtra(android.content.Intent.EXTRA_TEXT, report)
                    type = "text/plain"
                }
                startActivity(android.content.Intent.createChooser(shareIntent, "分享报告"))
            }
            .show()
    }

    private fun startProcessing() {
        cameraManager.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        processingScope.cancel()
        inferenceEngine.release()
        poseProcessor.release()
        cameraManager.stop()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100) {
            val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (!allGranted) {
                Toast.makeText(this, "需要相机权限才能使用应用", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }
}