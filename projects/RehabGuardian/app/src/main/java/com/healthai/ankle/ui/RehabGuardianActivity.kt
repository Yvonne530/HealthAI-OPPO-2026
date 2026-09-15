package com.healthai.ankle.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.healthai.ankle.R
import com.healthai.ankle.db.FrameEntity
import com.healthai.ankle.db.RehabDatabase
import com.healthai.ankle.db.SessionEntity
import com.healthai.ankle.inference.InferenceResult
import com.healthai.ankle.inference.RGPhaseAEngine
import com.healthai.ankle.inference.RiskExplainer
import com.healthai.ankle.report.PdfReportBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.ArrayDeque
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class RehabGuardianActivity : AppCompatActivity() {

    // ── Views ─────────────────────────────────────────────────────────────
    private lateinit var previewView:    PreviewView
    private lateinit var overlayView:    PoseOverlayView
    private lateinit var riskCardView:   RiskCardView
    private lateinit var grfForecast:    GrfForecastView
    private lateinit var compareView:    BestFrameCompareView
    private lateinit var tvKneeLeft:     TextView
    private lateinit var tvKneeRight:    TextView
    private lateinit var tvLatency:      TextView
    private lateinit var tvFrameCount:   TextView
    private lateinit var ivCatMascot:    ImageView
    private lateinit var btnStartStop:   Button
    private lateinit var btnReport:      ImageButton
    private lateinit var cameraSpacerView: View

    // Legacy hidden views (kept for backward-compat with DB/report code)
    private lateinit var tvRiskLabel:   TextView
    private lateinit var tvRiskScore:   TextView
    private lateinit var tvMotionScore: TextView
    private lateinit var tvHint:        TextView
    private lateinit var tvConfidence:  TextView
    private lateinit var viewRiskBar:   ProgressBar

    // ── Engine + Camera ───────────────────────────────────────────────────
    private val engine           = RGPhaseAEngine()
    private var poseLandmarker:  PoseLandmarker? = null
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // ── 20-frame ring buffer ──────────────────────────────────────────────
    private val frameBuffer = ArrayDeque<Array<FloatArray>>(20)
    private val bufLock     = Any()
    private val running     = AtomicBoolean(false)

    // ── Best-frame tracking ───────────────────────────────────────────────
    // §6.4: best = lowest riskScore where both knees ≥ 150°; fallback = global min
    private var bestResult:     InferenceResult? = null
    private var bestRiskScore:  Float = Float.MAX_VALUE
    private var bestGoodKnees:  InferenceResult? = null  // knees ≥ 150°
    private var bestGoodScore:  Float = Float.MAX_VALUE
    private var bestLandmarks:  Array<FloatArray>? = null
    private var bestGoodLandmarks: Array<FloatArray>? = null

    // ── Session ───────────────────────────────────────────────────────────
    private var currentSessionId = -1L
    private var sessionStartMs   = 0L
    private var totalFrames      = 0
    private var maxRisk          = 0
    private var sumMotion        = 0f
    private val db by lazy { RehabDatabase.getInstance(this) }

    // ── Hint throttle ─────────────────────────────────────────────────────
    private val mainHandler = Handler(Looper.getMainLooper())
    private var lastHintMs  = 0L

    // ── Permission ────────────────────────────────────────────────────────
    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        if (grants[Manifest.permission.CAMERA] == true) startCamera()
        else Toast.makeText(this, "需要摄像头权限才能运行", Toast.LENGTH_LONG).show()
    }

    // ─────────────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.statusBarColor = Color.TRANSPARENT
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        setContentView(R.layout.activity_rehab_guardian)

        bindViews()
        setSpacerHeight()
        animateCat()

        // Engine initializes on background thread — UI never blocked
        analysisExecutor.execute { engine.init(this) }

        btnStartStop.setOnClickListener { toggleSession() }
        btnReport.setOnClickListener   { exportReport()  }

        if (hasCameraPermission()) startCamera() else requestCameraPermission()
    }

    // ── View binding ──────────────────────────────────────────────────────

    private fun bindViews() {
        previewView     = findViewById(R.id.previewView)
        overlayView     = findViewById(R.id.overlayView)
        riskCardView    = findViewById(R.id.riskCardView)
        grfForecast     = findViewById(R.id.grfForecastView)
        compareView     = findViewById(R.id.compareView)
        tvKneeLeft      = findViewById(R.id.tvKneeLeft)
        tvKneeRight     = findViewById(R.id.tvKneeRight)
        tvLatency       = findViewById(R.id.tvLatency)
        tvFrameCount    = findViewById(R.id.tvFrameCount)
        ivCatMascot     = findViewById(R.id.ivCatMascot)
        btnStartStop    = findViewById(R.id.btnStartStop)
        btnReport       = findViewById(R.id.btnReport)
        cameraSpacerView = findViewById(R.id.cameraSpacerView)

        // Legacy hidden
        tvRiskLabel   = findViewById(R.id.tvRiskLabel)
        tvRiskScore   = findViewById(R.id.tvRiskScore)
        tvMotionScore = findViewById(R.id.tvMotionScore)
        tvHint        = findViewById(R.id.tvHint)
        tvConfidence  = findViewById(R.id.tvConfidence)
        viewRiskBar   = findViewById(R.id.viewRiskBar)

        riskCardView.setWaiting()
    }

    /** Set the camera spacer to ~42% of screen height so the sheet starts mid-screen */
    private fun setSpacerHeight() {
        val screenH = resources.displayMetrics.heightPixels
        val lp = cameraSpacerView.layoutParams
        lp.height = (screenH * 0.42f).toInt()
        cameraSpacerView.layoutParams = lp
    }

    // ── Cat idle animation ────────────────────────────────────────────────

    private fun animateCat() {
        ivCatMascot.animate()
            .translationY(-8f).setDuration(950)
            .setInterpolator(android.view.animation.DecelerateInterpolator())
            .withEndAction {
                ivCatMascot.animate()
                    .translationY(0f).setDuration(950)
                    .setInterpolator(android.view.animation.AccelerateDecelerateInterpolator())
                    .withEndAction { animateCat() }.start()
            }.start()
    }

    // ── Session ───────────────────────────────────────────────────────────

    private fun toggleSession() {
        if (!running.get()) startSession() else stopSession()
    }

    private fun startSession() {
        running.set(true)
        btnStartStop.text = "⏹ 停止"
        btnReport.visibility = View.GONE
        totalFrames = 0; maxRisk = 0; sumMotion = 0f
        sessionStartMs = System.currentTimeMillis()
        resetBestFrame()
        grfForecast.clear()

        lifecycleScope.launch(Dispatchers.IO) {
            currentSessionId = db.dao().insertSession(
                SessionEntity(userWeightKg = engine.userWeightKg)
            )
        }
    }

    private fun stopSession() {
        running.set(false)
        btnStartStop.text = "▶ 开始"
        btnReport.visibility = View.VISIBLE

        if (currentSessionId > 0 && totalFrames > 0) {
            val sid = currentSessionId
            val avg = sumMotion / totalFrames
            lifecycleScope.launch(Dispatchers.IO) {
                db.dao().updateSession(SessionEntity(
                    id             = sid,
                    startTimeMs    = sessionStartMs,
                    endTimeMs      = System.currentTimeMillis(),
                    userWeightKg   = engine.userWeightKg,
                    avgMotionScore = avg,
                    maxRiskLabel   = maxRisk,
                    totalFrames    = totalFrames
                ))
            }
        }
    }

    private fun resetBestFrame() {
        bestResult = null;       bestRiskScore  = Float.MAX_VALUE
        bestGoodKnees = null;   bestGoodScore  = Float.MAX_VALUE
        bestLandmarks = null;   bestGoodLandmarks = null
        compareView.clear()
    }

    // ── Camera ────────────────────────────────────────────────────────────

    private fun startCamera() {
        initPoseLandmarker()
        val cameraFuture = ProcessCameraProvider.getInstance(this)
        cameraFuture.addListener({
            val provider = cameraFuture.get()
            val preview  = Preview.Builder().build()
                .also { p -> p.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build().also { a -> a.setAnalyzer(analysisExecutor, ::processFrame) }
            runCatching {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }.onFailure { e -> Log.e(TAG, "Camera bind: ${e.message}") }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun initPoseLandmarker() {
        runCatching {
            val opts = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder()
                    .setModelAssetPath("models/pose_landmarker_lite.task").build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setNumPoses(1)
                .setResultListener { result, _ -> onPoseResult(result) }
                .setErrorListener { e -> Log.w(TAG, "Pose: ${e.message}") }
                .build()
            poseLandmarker = PoseLandmarker.createFromOptions(this, opts)
        }.onFailure { Log.e(TAG, "PoseLandmarker init: ${it.message}") }
    }

    private fun processFrame(proxy: ImageProxy) {
        val bmp = imageProxyToBitmap(proxy)
        proxy.close()
        if (bmp != null) poseLandmarker?.detectAsync(BitmapImageBuilder(bmp).build(), System.currentTimeMillis())
    }

    // ── Pose result ───────────────────────────────────────────────────────

    private fun onPoseResult(result: PoseLandmarkerResult) {
        val lms = result.landmarks().firstOrNull()?.takeIf { it.size >= 33 } ?: return
        val frame = Array(33) { i -> val lm = lms[i]; floatArrayOf(lm.x(), lm.y(), lm.z()) }

        runOnUiThread { overlayView.updateLandmarks(frame) }

        synchronized(bufLock) {
            if (frameBuffer.size >= 20) frameBuffer.pollFirst()
            frameBuffer.addLast(frame)
        }

        if (!running.get()) return

        val frames20: Array<Array<FloatArray>>
        synchronized(bufLock) {
            if (frameBuffer.size < 20) return
            frames20 = frameBuffer.toTypedArray()
        }

        val t0  = System.currentTimeMillis()
        val res = engine.infer(frames20) ?: return
        val ms  = System.currentTimeMillis() - t0

        onResult(res, frame, ms)
    }

    // ── Result handling ───────────────────────────────────────────────────

    private fun onResult(res: InferenceResult, currentFrame: Array<FloatArray>, latencyMs: Long) {
        totalFrames++
        sumMotion += res.motionScore
        if (res.riskLabel > maxRisk) maxRisk = res.riskLabel

        // §6.4 Best-frame tracking
        trackBestFrame(res, currentFrame)

        // Persist to DB async
        if (currentSessionId > 0) {
            val sid = currentSessionId
            lifecycleScope.launch(Dispatchers.IO) {
                db.dao().insertFrame(FrameEntity(
                    sessionId    = sid,
                    riskScore    = res.riskScore,
                    riskLabel    = res.riskLabel,
                    confidence   = res.confidence,
                    lKneeDeg     = res.kneeAnglesDeg.first,
                    rKneeDeg     = res.kneeAnglesDeg.second,
                    motionScore  = res.motionScore,
                    grfLeftFz    = res.grfLeft.getOrElse(0) { 0f },
                    grfRightFz   = res.grfRight.getOrElse(0) { 0f },
                    explanations = res.explanation.reasons.joinToString(";") { it.text }
                ))
            }
        }

        runOnUiThread { updateUi(res, currentFrame, latencyMs) }
    }

    /**
     * §6.4: Best frame = lowest risk among frames with both knees ≥ 150°.
     * Fallback: global lowest-risk frame.
     */
    private fun trackBestFrame(res: InferenceResult, frame: Array<FloatArray>) {
        val lk = res.kneeAnglesDeg.first
        val rk = res.kneeAnglesDeg.second

        // Global best
        if (res.riskScore < bestRiskScore) {
            bestRiskScore = res.riskScore
            bestResult    = res
            bestLandmarks = frame.map { it.copyOf() }.toTypedArray()
        }

        // Good-knees best (primary)
        if (lk >= 150f && rk >= 150f && res.riskScore < bestGoodScore) {
            bestGoodScore     = res.riskScore
            bestGoodKnees     = res
            bestGoodLandmarks = frame.map { it.copyOf() }.toTypedArray()
        }
    }

    // ── UI update ─────────────────────────────────────────────────────────

    private fun updateUi(res: InferenceResult, currentFrame: Array<FloatArray>, latencyMs: Long) {

        // ── RiskCardView (hero panel) ──────────────────────────────────────
        riskCardView.update(
            label   = res.riskLabel,
            score   = res.riskScore,
            conf    = res.confidence,
            motion  = res.motionScore,
            expl    = res.explanation,
            future  = res.explanation.futureThreat
        )

        // ── GRF Forecast chart ────────────────────────────────────────────
        grfForecast.push(
            leftFz  = res.grfLeft.getOrElse(0)  { 0f },
            rightFz = res.grfRight.getOrElse(0) { 0f },
            future  = res.grfFutureFlat
        )

        // ── Skeleton comparison ────────────────────────────────────────────
        val lk = res.kneeAnglesDeg.first
        val rk = res.kneeAnglesDeg.second
        compareView.updateCurrent(BestFrameCompareView.SkeletonData(
            landmarks   = currentFrame,
            lKneeDeg    = lk,
            rKneeDeg    = rk,
            motionScore = res.motionScore,
            label       = "当前动作"
        ))
        // Update best panel if we have a good-knees best
        val bestLm = bestGoodLandmarks ?: bestLandmarks
        val bestRes = bestGoodKnees   ?: bestResult
        if (bestLm != null && bestRes != null) {
            compareView.updateBest(BestFrameCompareView.SkeletonData(
                landmarks   = bestLm,
                lKneeDeg    = bestRes.kneeAnglesDeg.first,
                rKneeDeg    = bestRes.kneeAnglesDeg.second,
                motionScore = bestRes.motionScore,
                label       = if (bestGoodKnees != null) "最佳动作 ✓" else "最低风险帧"
            ))
        }

        // ── Metric chips ──────────────────────────────────────────────────
        val kneeColor = { deg: Float ->
            when { deg >= 150f -> Color.parseColor("#22C55E")
                   deg >= 130f -> Color.parseColor("#F59E0B")
                   else        -> Color.parseColor("#EF4444") }
        }
        tvKneeLeft.text  = "${lk.toInt()} °";  tvKneeLeft.setTextColor(kneeColor(lk))
        tvKneeRight.text = "${rk.toInt()} °";  tvKneeRight.setTextColor(kneeColor(rk))
        tvLatency.text   = "${latencyMs} ms"
        tvFrameCount.text = "帧 $totalFrames  ·  端侧推理 🐾"

        // ── Cat reacts to risk ────────────────────────────────────────────
        val scale = when (res.riskLabel) { 2 -> 1.2f; 1 -> 1.06f; else -> 1f }
        ivCatMascot.animate().scaleX(scale).scaleY(scale).setDuration(260).start()

        // ── §6.3 Throttled hint (medium/high, once per 2s) ─────────────
        if (res.riskLabel >= 1) {
            val now = System.currentTimeMillis()
            if (now - lastHintMs >= HINT_THROTTLE_MS) {
                lastHintMs = now
                val tip = res.explanation.topCorrection
                if (tip.isNotBlank()) {
                    Toast.makeText(this, "🐾 $tip", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    // ── Report export ─────────────────────────────────────────────────────

    private fun exportReport() {
        val sid = currentSessionId
        if (sid <= 0) { Toast.makeText(this, "没有会话数据", Toast.LENGTH_SHORT).show(); return }
        lifecycleScope.launch {
            val session = withContext(Dispatchers.IO) { db.dao().sessionById(sid) } ?: return@launch
            val frames  = withContext(Dispatchers.IO) { db.dao().framesForSession(sid) }
            val out     = File(getExternalFilesDir("reports"), "rehab_${sid}.pdf")
            withContext(Dispatchers.IO) { PdfReportBuilder.build(this@RehabGuardianActivity, session, frames, out) }
            Toast.makeText(this@RehabGuardianActivity, "📄 报告已保存: ${out.name}", Toast.LENGTH_LONG).show()
        }
    }

    // ── Util ──────────────────────────────────────────────────────────────

    private fun imageProxyToBitmap(proxy: ImageProxy): Bitmap? = runCatching {
        val buf  = proxy.planes[0].buffer
        val data = ByteArray(buf.remaining()).also { buf.get(it) }
        val bmp  = Bitmap.createBitmap(proxy.width, proxy.height, Bitmap.Config.ARGB_8888)
        bmp.copyPixelsFromBuffer(java.nio.ByteBuffer.wrap(data))
        val mat  = Matrix().apply { postRotate(proxy.imageInfo.rotationDegrees.toFloat()) }
        Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, mat, true)
    }.getOrNull()

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        permLauncher.launch(arrayOf(Manifest.permission.CAMERA))
    }

    override fun onDestroy() {
        super.onDestroy()
        running.set(false)
        poseLandmarker?.close()
        analysisExecutor.execute { engine.release() }
        analysisExecutor.shutdown()
    }

    companion object {
        private const val TAG              = "RehabGuardianActivity"
        private const val HINT_THROTTLE_MS = 2000L
    }
}
