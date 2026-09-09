package com.erlab.actuaware

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.erlab.actuaware.analysis.BiomechanicsAnalyzer
import com.erlab.actuaware.analysis.RiskStateMachine
import com.erlab.actuaware.camera.CameraManager
import com.erlab.actuaware.camera.PoseProcessor
import com.erlab.actuaware.data.FrameData
import com.erlab.actuaware.databinding.ActivityMainBinding
import com.erlab.actuaware.inference.MNNInferenceEngine
import com.erlab.actuaware.ui.SessionManager
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    // ViewBinding
    private lateinit var binding: ActivityMainBinding

    // 核心模块
    private lateinit var inferenceEngine: MNNInferenceEngine
    private lateinit var sessionManager: SessionManager
    private lateinit var riskStateMachine: RiskStateMachine
    private var cameraManager: CameraManager? = null
    private var poseProcessor: PoseProcessor? = null

    // 状态控制
    private var isDetecting = false
    private val isProcessing = AtomicBoolean(false)   // 并发控制
    private var inferJob: Job? = null                  // 用于取消旧推理任务

    // 性能统计
    private var frameCount = 0
    private var lastFpsUpdate = 0L
    private var currentFps = 30f

    // 最近数据（用于 UI 更新）
    private var lastRiskState = RiskStateMachine.State.SAFE
    private var lastRiskScore = 0f
    private var lastLatency = 0f
    private var lastKneeAngleL = 180f
    private var lastKneeAngleR = 180f
    private var lastMpResult: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult? = null

    companion object {
        private const val CAMERA_PERMISSION = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 初始化模块
        inferenceEngine = MNNInferenceEngine(this)
        sessionManager = SessionManager(this)
        riskStateMachine = RiskStateMachine

        // 设置 UI 监听
        setupUI()

        // 检查相机权限
        checkCameraPermission()
    }

    // ======================== UI 初始化 ========================

    private fun setupUI() {
        // 开始/停止检测
        binding.btnStart.setOnClickListener {
            if (!isDetecting) startDetection() else stopDetection()
        }

        // 历史记录
        binding.btnHistory.setOnClickListener {
            Toast.makeText(this, "历史记录功能开发中", Toast.LENGTH_SHORT).show()
        }

        // 翻转摄像头
        binding.btnFlip.setOnClickListener {
            cameraManager?.flipCamera()
        }

        // 报告按钮（检测结束后显示）
        binding.btnReport.setOnClickListener {
            showReportDialog()
        }
        binding.btnReport.visibility = View.GONE
    }

    // ======================== 权限处理 ========================

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION
            )
        } else {
            initModules()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            initModules()
        } else {
            Toast.makeText(this, "需要相机权限才能使用应用", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    // ======================== 模块初始化 ========================

    private fun initModules() {
        // 1. 初始化推理引擎（异步）
        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                inferenceEngine.initialize()
                inferenceEngine.userWeightKg = 70f
            }
            withContext(Dispatchers.Main) {
                binding.tvStatus.text = "模型加载完成，就绪"
            }
        }

        // 2. 初始化姿态处理器
        poseProcessor = PoseProcessor(this) { keypoints, mpResult ->
            if (!isDetecting) return@PoseProcessor
            lastMpResult = mpResult

            val timestampMs = System.currentTimeMillis()

            // 计算膝关节角度（复用 BiomechanicsAnalyzer）
            val kneeAngleL = BiomechanicsAnalyzer.calcAngle3D(
                keypoints[23], keypoints[25], keypoints[27]
            )
            val kneeAngleR = BiomechanicsAnalyzer.calcAngle3D(
                keypoints[24], keypoints[26], keypoints[28]
            )
            lastKneeAngleL = kneeAngleL
            lastKneeAngleR = kneeAngleR

            // 异步推理（使用并发控制，避免任务堆积）
            if (isProcessing.compareAndSet(false, true)) {
                inferJob?.cancel()
                inferJob = lifecycleScope.launch(Dispatchers.Default) {
                    try {
                        val result = inferenceEngine.infer(keypoints) ?: return@launch

                        // 更新状态机（可传入心率和睡眠分数，这里暂时用默认值）
                        val state = riskStateMachine.update(
                            rawScore = result.riskScore,
                            confidence = 0.85f,
                            heartRate = 75f,
                            sleepScore = 80f
                        )
                        lastRiskState = state
                        lastRiskScore = riskStateMachine.currentScore
                        lastLatency = result.latencyMs

                        // 如果是录制状态，保存帧数据
                        if (sessionManager.isRecording) {
                            val frameData = FrameData(
                                timestampMs = timestampMs,
                                keypoints = keypoints,
                                riskScore = lastRiskScore,
                                riskLabel = when (state) {
                                    RiskStateMachine.State.SAFE -> 0
                                    RiskStateMachine.State.WARNING -> 1
                                    RiskStateMachine.State.DANGER -> 2
                                },
                                grfLeft = result.grfLeft,
                                grfRight = result.grfRight,
                                kneeAngleL = kneeAngleL,
                                kneeAngleR = kneeAngleR
                            )
                            sessionManager.addFrame(frameData)
                        }

                        // 更新 UI
                        withContext(Dispatchers.Main) {
                            updateUI()
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                    } finally {
                        isProcessing.set(false)
                    }
                }
            }
        }
        poseProcessor?.initialize()

        // 3. 初始化摄像头
        cameraManager = CameraManager(this, this, binding.previewView) { bitmap, timestampMs ->
            poseProcessor?.processFrame(bitmap, timestampMs)
        }
        cameraManager?.start()

        // 4. 启动 FPS 统计
        startFpsMonitor()
    }

    // ======================== 检测控制 ========================

    private fun startDetection() {
        isDetecting = true
        sessionManager.startSession()
        riskStateMachine.reset()
        binding.btnStart.text = "停止检测"
        binding.btnStart.setBackgroundColor(getColor(android.R.color.holo_red_light))
        binding.btnReport.visibility = View.GONE
        binding.tvStatus.text = "检测中..."
        // 清除旧数据
        lastRiskState = RiskStateMachine.State.SAFE
        lastRiskScore = 0f
    }

    private fun stopDetection() {
        isDetecting = false
        val session = sessionManager.stopSession()
        binding.btnStart.text = "开始检测"
        binding.btnStart.setBackgroundColor(getColor(android.R.color.holo_green_dark))

        if (session != null) {
            val avgRisk = session.avgRisk
            val riskText = when {
                avgRisk < 0.35f -> "低风险"
                avgRisk < 0.70f -> "中风险"
                else -> "高风险"
            }
            binding.tvStatus.text = "检测完成 | 平均风险: ${"%.2f".format(avgRisk)} ($riskText)"
            binding.btnReport.visibility = View.VISIBLE
        } else {
            binding.tvStatus.text = "检测结束"
        }
    }

    // ======================== UI 更新 ========================

    private fun updateUI() {
        // 更新渲染器（RiskRenderer）
        // 注意：这里需要将状态转换为标签和分数，并传入延迟
        // 假设 binding.riskRenderer 是 RiskRenderer 实例
        binding.riskRenderer.update(
            result = lastMpResult ?: return,
            risk = lastRiskScore,
            label = when (lastRiskState) {
                RiskStateMachine.State.SAFE -> 0
                RiskStateMachine.State.WARNING -> 1
                RiskStateMachine.State.DANGER -> 2
            },
            kneL = lastKneeAngleL,
            kneR = lastKneeAngleR,
            imgW = 640,
            imgH = 480,
            latency = lastLatency
        )

        // 更新风险指示器（进度条 + 文字）
        val riskPercent = (lastRiskScore * 100).toInt()
        binding.tvRisk.text = when (lastRiskState) {
            RiskStateMachine.State.SAFE -> "低风险 ${riskPercent}%"
            RiskStateMachine.State.WARNING -> "中风险 ${riskPercent}%"
            RiskStateMachine.State.DANGER -> "高风险 ${riskPercent}%"
        }
        val riskColor = when (lastRiskState) {
            RiskStateMachine.State.SAFE -> android.R.color.holo_green_dark
            RiskStateMachine.State.WARNING -> android.R.color.holo_orange_light
            RiskStateMachine.State.DANGER -> android.R.color.holo_red_light
        }
        binding.tvRisk.setTextColor(getColor(riskColor))
        binding.progressRisk.progress = riskPercent
        binding.progressRisk.progressTintList = android.content.res.ColorStateList.valueOf(
            getColor(riskColor)
        )

        // 延迟显示
        binding.tvLatency.text = "${lastLatency.toInt()} ms"

        // 高风险时显示警告条
        binding.layoutWarning.visibility = if (lastRiskState == RiskStateMachine.State.DANGER) {
            View.VISIBLE
        } else {
            View.GONE
        }

        // FPS 显示
        binding.tvFps.text = "${"%.1f".format(currentFps)} FPS"
    }

    private fun startFpsMonitor() {
        lifecycleScope.launch(Dispatchers.Default) {
            while (true) {
                delay(1000)
                frameCount = 0
                lastFpsUpdate = System.currentTimeMillis()
            }
        }
    }

    // 用于在相机回调中统计帧数
    private fun onFrameReceived() {
        frameCount++
        val now = System.currentTimeMillis()
        if (now - lastFpsUpdate >= 1000) {
            currentFps = frameCount / ((now - lastFpsUpdate) / 1000f)
            frameCount = 0
            lastFpsUpdate = now
            sessionManager.setFps(currentFps)
        }
    }

    // ======================== 报告弹窗 ========================

    private fun showReportDialog() {
        val session = sessionManager.stopSession() // 获取当前会话（如果还在录制则结束）
        if (session == null) {
            Toast.makeText(this, "没有可用的会话记录", Toast.LENGTH_SHORT).show()
            return
        }
        val reportText = sessionManager.generateReport(session)
        AlertDialog.Builder(this)
            .setTitle("运动分析报告")
            .setMessage(reportText)
            .setPositiveButton("关闭", null)
            .setNeutralButton("查看历史") { _, _ ->
                Toast.makeText(this, "历史记录功能开发中", Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    // ======================== 生命周期 ========================

    override fun onDestroy() {
        super.onDestroy()
        inferenceEngine.release()
        cameraManager?.stop()
        poseProcessor?.release()
        inferJob?.cancel()
    }
}