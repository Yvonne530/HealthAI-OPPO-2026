package com.erlab.actuaware

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.GravityCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.erlab.actuaware.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import java.io.File
import io.noties.markwon.Markwon
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.ext.tasklist.TaskListPlugin
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.Executors

interface StreamingCallback {
    fun onToken(token: String)
}

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var drawerToggle: ActionBarDrawerToggle
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var historyAdapter: HistoryAdapter
    private lateinit var networkHelper: NetworkHelper
    private lateinit var historyHelper: HistoryHelper
    private lateinit var toolManager: ToolManager
    private lateinit var toolChatManager: ToolChatManager
    private lateinit var performanceManager: PerformanceManager
    private lateinit var recognitionMarkwon: Markwon  // 用于识别界面的 Markdown 渲染
    private val imageCacheManager = ImageCacheManager.getInstance()
    private var progressDialog: ModelLoadProgressDialog? = null
    private var currentImageUri: Uri? = null
    private var recognitionImageUri: Uri? = null
    private var cachedModelPath: String? = null
    private var cachedMmprojPath: String? = null
    private var recognitionPrompt: String = "请分析这张康复训练动作是否规范，给出建议。"
    private var isModelLoaded = false
    private var systemPrompt: String = "你是一个有用的助手。"
    private val conversationHistory = mutableListOf<String>()
    private var maxTokens: Int = 512
    private var contextSize: Int = 8192
    private var temperature: Float = 0.7f
    private var topP: Float = 0.9f
    private var topK: Int = 40
    private var enableNetwork: Boolean = false
    private var enableTools: Boolean = true
    private var isDarkMode: Boolean = false
    private var currentHistoryId: String = ""
    private var isHistoryLoadingToJNI: Boolean = false
    private var isInitializing: Boolean = false

    // CameraX 实时摄像头捕获相关
    private var cameraExecutor: ExecutorService? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var isCameraCapturing: Boolean = false
    private var cameraCaptureInterval: Long = 100 // 每100ms捕获一次帧（10 FPS）
    private var currentCameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    // MediaPipe Pose 检测器
    private var poseDetector: PoseDetector? = null

    // 智能滚动：用户向上滚动时暂停自动滚动，滚动到底部时恢复
    private var autoScrollEnabled: Boolean = true

    private var userNickname: String = "用户"
    private var userAvatarPath: String? = null
    private var aiNickname: String = "AI助手"
    private var aiAvatarPath: String? = null

    private external fun nativeSetPerformanceParams(threads: Int, batchSize: Int, gpuLayers: Int, enableFlashAttention: Boolean)
    private external fun nativeLoadModel(modelPath: String, mmprojPath: String, systemPrompt: String, maxTokens: Int, contextSize: Int, temperature: Float, topP: Float, topK: Int): String
    // 使用 DirectByteBuffer 减少 JNI 复制开销，支持 ARGB 格式
    private external fun nativeAnalyzeImageDirect(imageData: java.nio.ByteBuffer, width: Int, height: Int, userInput: String): String
    private external fun nativeChat(userInput: String, resetHistory: Boolean): String
    private external fun nativeResetChatHistory(): Unit
    private external fun nativeRestoreContext(historyJson: String): String
    private external fun nativeFreeModel()
    private external fun nativeInitStreamingCallback(callback: StreamingCallback)
    private external fun nativeCleanupStreamingCallback()

    companion object {
        init {
            System.loadLibrary("llama_jni")
        }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            currentImageUri = it
            displayImage(it)
            binding.imageSection.visibility = View.VISIBLE
        }
    }

    private val captureImageLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success && currentImageUri != null) {
            displayImage(currentImageUri!!)
            binding.imageSection.visibility = View.VISIBLE
        } else {
            currentImageUri = null
        }
    }

    private val pickUserAvatarLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            saveUserAvatar(it)
        }
    }

    private val pickModelLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            progressDialog = ModelLoadProgressDialog(this)
            progressDialog?.setTitle("复制模型")
            progressDialog?.setMessage("正在复制主模型文件...")
            progressDialog?.show()

            Thread {
                try {
                    val copiedPath = copyFileToPrivateDirWithProgress(it, true, progressDialog)
                    Handler(Looper.getMainLooper()).post {
                        progressDialog?.dismiss()
                        if (copiedPath != null) {
                            // 复制成功后，删除旧的模型文件
                            val modelsDir = File(filesDir, "models")
                            if (modelsDir.exists()) {
                                modelsDir.listFiles { file ->
                                    file.isFile && file.name.endsWith(".gguf") && file.absolutePath != copiedPath
                                }?.forEach { file ->
                                    file.delete()
                                }
                            }
                            
                            cachedModelPath = copiedPath
                            saveUserSettings()  // 保存模型路径
                            Toast.makeText(this, "模型已复制到私有目录", Toast.LENGTH_SHORT).show()
                            updateSettingsModelDisplay()
                        } else {
                            Toast.makeText(this, "复制失败或已取消", Toast.LENGTH_SHORT).show()
                        }
                    }
                } catch (e: Exception) {
                    Handler(Looper.getMainLooper()).post {
                        progressDialog?.dismiss()
                        Toast.makeText(this, "复制失败: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }.start()
        }
    }

    private val pickMmprojLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            progressDialog = ModelLoadProgressDialog(this)
            progressDialog?.setTitle("复制多模态")
            progressDialog?.setMessage("正在复制多模态文件...")
            progressDialog?.show()

            Thread {
                try {
                    val copiedPath = copyFileToPrivateDirWithProgress(it, false, progressDialog)
                    Handler(Looper.getMainLooper()).post {
                        progressDialog?.dismiss()
                        if (copiedPath != null) {
                            // 复制成功后，删除旧的多模态文件
                            val mmprojDir = File(filesDir, "mmproj")
                            if (mmprojDir.exists()) {
                                mmprojDir.listFiles { file ->
                                    file.isFile && file.name.endsWith(".gguf") && file.absolutePath != copiedPath
                                }?.forEach { file ->
                                    file.delete()
                                }
                            }
                            
                            cachedMmprojPath = copiedPath
                            saveUserSettings()  // 保存 mmproj 路径
                            Toast.makeText(this, "多模态文件已复制到私有目录", Toast.LENGTH_SHORT).show()
                            updateSettingsModelDisplay()
                        } else {
                            Toast.makeText(this, "复制失败或已取消", Toast.LENGTH_SHORT).show()
                        }
                    }
                } catch (e: Exception) {
                    Handler(Looper.getMainLooper()).post {
                        progressDialog?.dismiss()
                        Toast.makeText(this, "复制失败: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }.start()
        }
    }

    private val pickImportMmprojLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            progressDialog = ModelLoadProgressDialog(this)
            progressDialog?.setTitle("导入多模态")
            progressDialog?.setMessage("正在导入多模态文件...")
            progressDialog?.show()

            Thread {
                try {
                    val copiedPath = copyFileToPrivateDirWithProgress(it, false, progressDialog, "mmproj")
                    Handler(Looper.getMainLooper()).post {
                        progressDialog?.dismiss()
                        if (copiedPath != null) {
                            // 复制成功后，删除旧的多模态文件
                            val mmprojDir = File(filesDir, "mmproj")
                            if (mmprojDir.exists()) {
                                mmprojDir.listFiles { file ->
                                    file.isFile && file.name.endsWith(".gguf") && file.absolutePath != copiedPath
                                }?.forEach { file ->
                                    file.delete()
                                }
                            }
                            
                            cachedMmprojPath = copiedPath
                            saveUserSettings()  // 保存 mmproj 路径
                            Toast.makeText(this, "多模态文件已导入", Toast.LENGTH_SHORT).show()
                            updateSettingsModelDisplay()
                        } else {
                            Toast.makeText(this, "导入失败或已取消", Toast.LENGTH_SHORT).show()
                        }
                    }
                } catch (e: Exception) {
                    Handler(Looper.getMainLooper()).post {
                        progressDialog?.dismiss()
                        Toast.makeText(this, "导入失败: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }.start()
        }
    }

    private val pickRecognitionImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            recognitionImageUri = it
            displayRecognitionImage(it)
        }
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            try {
                // 权限已授予，启动相机
                val photoFile = File(cacheDir, "recognition_${System.currentTimeMillis()}.jpg")
                recognitionImageUri = androidx.core.content.FileProvider.getUriForFile(
                    this,
                    "com.erlab.actuaware.fileprovider",
                    photoFile
                )
                captureRecognitionImageLauncher.launch(recognitionImageUri!!)
            } catch (e: Exception) {
                Toast.makeText(this, "启动相机失败: ${e.message}", Toast.LENGTH_SHORT).show()
                Log.e("MainActivity", "启动相机失败", e)
            }
        } else {
            Toast.makeText(this, "需要相机权限才能拍摄图片", Toast.LENGTH_SHORT).show()
        }
    }

    private val captureRecognitionImageLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success && recognitionImageUri != null) {
            displayRecognitionImage(recognitionImageUri!!)
        } else {
            recognitionImageUri = null
        }
    }

    private val chatImagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            try {
                val photoFile = File(cacheDir, "chat_${System.currentTimeMillis()}.jpg")
                currentImageUri = androidx.core.content.FileProvider.getUriForFile(
                    this,
                    "com.erlab.actuaware.fileprovider",
                    photoFile
                )
                captureImageLauncher.launch(currentImageUri!!)
            } catch (e: Exception) {
                Toast.makeText(this, "启动相机失败: ${e.message}", Toast.LENGTH_SHORT).show()
                Log.e("MainActivity", "启动相机失败", e)
            }
        } else {
            Toast.makeText(this, "需要相机权限才能拍摄图片", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.d("MainActivity", "onCreate 开始")
        super.onCreate(savedInstanceState)

        try {
            networkHelper = NetworkHelper()
            historyHelper = HistoryHelper(this)

            toolManager = ToolManager(this)
            toolChatManager = ToolChatManager(toolManager) { message, reset ->
                nativeChat(message, reset)
            }

            // 初始化性能管理器
            performanceManager = PerformanceManager(this)
            Log.d("MainActivity", performanceManager.getDeviceInfo())

            // 初始化识别界面的 Markwon 实例
            recognitionMarkwon = Markwon.builder(this)
                .usePlugin(StrikethroughPlugin.create())
                .usePlugin(TablePlugin.create(this))
                .usePlugin(TaskListPlugin.create(this))
                .build()

            // 根据设备性能自适应配置
            cameraCaptureInterval = performanceManager.getRecommendedCameraInterval()
            Log.d("MainActivity", "摄像头捕获间隔已设置为: ${cameraCaptureInterval}ms")

            binding = ActivityMainBinding.inflate(layoutInflater)
            setContentView(binding.root)

            chatAdapter = ChatAdapter()
            binding.recyclerViewChat.apply {
                layoutManager = LinearLayoutManager(this@MainActivity)
                adapter = chatAdapter
            }
            chatAdapter.addSystemMessage("请先加载模型，然后开始对话")

            historyAdapter = HistoryAdapter(
                onLoad = { historyItem ->
                    loadHistory(historyItem)
                },
                onDelete = { historyItem ->
                    deleteHistory(historyItem)
                }
            )
            binding.recyclerViewHistory.apply {
                layoutManager = LinearLayoutManager(this@MainActivity)
                adapter = historyAdapter
            }

            // 初始化 MediaPipe Pose 检测器
            try {
                poseDetector = PoseDetection.getClient(
                    PoseDetectorOptions.Builder()
                        .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                        .build()
                )
            } catch (e: Exception) {
                e.printStackTrace()
                Log.e("MainActivity", "初始化 Pose 检测器失败: ${e.message}")
            }

            binding.bottomNav.setOnItemSelectedListener { item ->
                when (item.itemId) {
                    R.id.navigation_chat -> {
                        showChatView()
                        true
                    }
                    R.id.navigation_history -> {
                        showHistoryView()
                        true
                    }
                    R.id.navigation_recognition -> {
                        showRecognitionView()
                        true
                    }
                    R.id.navigation_models -> {
                        showModelsView()
                        true
                    }
                    else -> false
                }
            }

            binding.buttonNewChat.setOnClickListener {
                startNewChat()
            }

            setSupportActionBar(binding.toolbar)
            supportActionBar?.setDisplayHomeAsUpEnabled(true)

            drawerToggle = ActionBarDrawerToggle(
                this,
                binding.drawerLayout,
                R.string.drawer_open,
                R.string.drawer_close
            )
            binding.drawerLayout.addDrawerListener(drawerToggle)
            drawerToggle.syncState()

            binding.toolbar.setNavigationOnClickListener {
                if (binding.drawerLayout.isDrawerOpen(GravityCompat.START)) {
                    binding.drawerLayout.closeDrawer(GravityCompat.START)
                } else {
                    binding.drawerLayout.openDrawer(GravityCompat.START)
                }
            }

            binding.buttonSendMessage.setOnClickListener {
                val message = binding.editTextInput.text.toString().trim()
                if (message.isEmpty() && currentImageUri == null) {
                    Toast.makeText(this, "请输入消息或选择图片", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                if (!isModelLoaded) {
                    chatAdapter.addSystemMessage("请先加载模型")
                    Toast.makeText(this, "请先加载模型", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                val imm = getSystemService(android.content.Context.INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager
                imm.hideSoftInputFromWindow(binding.editTextInput.windowToken, 0)

                sendMessage(message)
                binding.recyclerViewChat.post {
                    binding.recyclerViewChat.scrollToPosition(chatAdapter.itemCount - 1)
                }
            }

            binding.buttonAddImage.setOnClickListener {
                val options = arrayOf("拍照", "从图库选择")
                android.app.AlertDialog.Builder(this)
                    .setTitle("选择图片来源")
                    .setItems(options) { _, which ->
                        when (which) {
                            0 -> chatImagePermissionLauncher.launch(android.Manifest.permission.CAMERA)
                            1 -> pickImageLauncher.launch("image/*")
                        }
                    }
                    .show()
            }

            binding.buttonRemoveImage.setOnClickListener {
                currentImageUri = null
                binding.imageSection.visibility = View.GONE
                binding.imageView.setImageURI(null)
            }

            binding.imageViewUserAvatar.setOnClickListener {
                showChangeAvatarDialog()
            }

            binding.textViewUserNickname.setOnClickListener {
                showChangeNicknameDialog()
            }

            binding.buttonImportMmproj.setOnClickListener {
                pickImportMmprojLauncher.launch("*/*")
            }

            binding.buttonImportModelFromView.setOnClickListener {
                pickModelLauncher.launch("*/*")
            }

            binding.buttonImportMmprojFromView.setOnClickListener {
                pickImportMmprojLauncher.launch("*/*")
            }

            binding.buttonSelectRecognitionImage.setOnClickListener {
                pickRecognitionImageLauncher.launch("image/*")
            }

            binding.buttonCaptureRecognitionImage.setOnClickListener {
                cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
            }

            binding.buttonRemoveRecognitionImage.setOnClickListener {
                recognitionImageUri = null
                binding.recognitionImageSection.visibility = View.GONE
                binding.textViewRecognitionPlaceholder.visibility = View.VISIBLE
                binding.imageViewRecognition.setImageURI(null)
                binding.textViewRecognitionResult.text = "等待识别..."
                binding.buttonStartRecognitionAnalysis.isEnabled = false
            }

            binding.buttonStartRecognitionAnalysis.setOnClickListener {
                performRecognitionAnalysis()
            }

            // 设置识别界面 ScrollView 的滚动监听器，实现智能滚动
            binding.scrollViewRecognition.viewTreeObserver.addOnScrollChangedListener {
                val scrollView = binding.scrollViewRecognition
                val child = scrollView.getChildAt(0)
                if (child != null) {
                    // 判断是否滚动到底部（允许 10px 误差）
                    val isAtBottom = scrollView.height + scrollView.scrollY >= child.height - 10
                    autoScrollEnabled = isAtBottom
                }
            }

            // 实时摄像头相关按钮
            binding.buttonStartCameraCapture.setOnClickListener {
                startCameraCapture()
            }

            binding.buttonSwitchCamera.setOnClickListener {
                switchCamera()
            }

            binding.buttonStopCameraCapture.setOnClickListener {
                stopCameraCapture()
            }

            binding.buttonSetSystemPrompt.setOnClickListener {
                showSystemPromptDialog()
            }
            binding.editTextMaxTokens.setText("$maxTokens")
            binding.seekBarMaxTokens.progress = maxTokens
            binding.seekBarMaxTokens.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                    maxTokens = if (progress < 1) 1 else progress
                    if (fromUser) {
                        binding.editTextMaxTokens.setText("$maxTokens")
                    }
                }
                override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                    Toast.makeText(this@MainActivity, "最大生成token数: $maxTokens", Toast.LENGTH_SHORT).show()
                }
            })
            binding.editTextMaxTokens.setOnEditorActionListener { _, _, _ ->
                val value = binding.editTextMaxTokens.text.toString().toIntOrNull()
                if (value != null && value > 0 && value <= 4096) {
                    maxTokens = value
                    binding.seekBarMaxTokens.progress = maxTokens
                    true
                } else {
                    binding.editTextMaxTokens.setText("$maxTokens")
                    Toast.makeText(this@MainActivity, "请输入 1-4096 之间的整数", Toast.LENGTH_SHORT).show()
                    true
                }
            }

            binding.editTextContextSize.setText("$contextSize")
            binding.seekBarContextSize.progress = contextSize
            binding.seekBarContextSize.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                    contextSize = if (progress < 1024) 1024 else progress
                    if (fromUser) {
                        binding.editTextContextSize.setText("$contextSize")
                    }
                }
                override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                    Toast.makeText(this@MainActivity, "上下文大小: $contextSize", Toast.LENGTH_SHORT).show()
                }
            })
            binding.editTextContextSize.setOnEditorActionListener { _, _, _ ->
                val value = binding.editTextContextSize.text.toString().toIntOrNull()
                if (value != null && value >= 1024 && value <= 32768) {
                    contextSize = value
                    binding.seekBarContextSize.progress = contextSize
                    true
                } else {
                    binding.editTextContextSize.setText("$contextSize")
                    Toast.makeText(this@MainActivity, "请输入 1024-32768 之间的整数", Toast.LENGTH_SHORT).show()
                    true
                }
            }

            // 温度设置
            binding.editTextTemperature.setText(String.format("%.1f", temperature))
            binding.seekBarTemperature.progress = (temperature * 100).toInt()
            binding.seekBarTemperature.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                    if (fromUser) {
                        temperature = progress / 100f
                        binding.editTextTemperature.setText(String.format("%.1f", temperature))
                    }
                }
                override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                    Toast.makeText(this@MainActivity, "温度: $temperature", Toast.LENGTH_SHORT).show()
                    saveUserSettings()
                }
            })
            binding.editTextTemperature.setOnEditorActionListener { _, _, _ ->
                val value = binding.editTextTemperature.text.toString().toFloatOrNull()
                if (value != null && value >= 0f && value <= 2f) {
                    temperature = value
                    binding.seekBarTemperature.progress = (temperature * 100).toInt()
                    saveUserSettings()
                    true
                } else {
                    binding.editTextTemperature.setText(String.format("%.1f", temperature))
                    Toast.makeText(this@MainActivity, "请输入 0.0-2.0 之间的数值", Toast.LENGTH_SHORT).show()
                    true
                }
            }

            // Top-P设置
            binding.editTextTopP.setText(String.format("%.1f", topP))
            binding.seekBarTopP.progress = (topP * 100).toInt()
            binding.seekBarTopP.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                    if (fromUser) {
                        topP = progress / 100f
                        binding.editTextTopP.setText(String.format("%.1f", topP))
                    }
                }
                override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                    Toast.makeText(this@MainActivity, "Top-P: $topP", Toast.LENGTH_SHORT).show()
                    saveUserSettings()
                }
            })
            binding.editTextTopP.setOnEditorActionListener { _, _, _ ->
                val value = binding.editTextTopP.text.toString().toFloatOrNull()
                if (value != null && value >= 0f && value <= 1f) {
                    topP = value
                    binding.seekBarTopP.progress = (topP * 100).toInt()
                    saveUserSettings()
                    true
                } else {
                    binding.editTextTopP.setText(String.format("%.1f", topP))
                    Toast.makeText(this@MainActivity, "请输入 0.0-1.0 之间的数值", Toast.LENGTH_SHORT).show()
                    true
                }
            }

            // Top-K设置
            binding.editTextTopK.setText("$topK")
            binding.seekBarTopK.progress = topK
            binding.seekBarTopK.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                    if (fromUser) {
                        topK = progress
                        binding.editTextTopK.setText("$topK")
                    }
                }
                override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                    Toast.makeText(this@MainActivity, "Top-K: $topK", Toast.LENGTH_SHORT).show()
                    saveUserSettings()
                }
            })
            binding.editTextTopK.setOnEditorActionListener { _, _, _ ->
                val value = binding.editTextTopK.text.toString().toIntOrNull()
                if (value != null && value >= 1 && value <= 100) {
                    topK = value
                    binding.seekBarTopK.progress = topK
                    saveUserSettings()
                    true
                } else {
                    binding.editTextTopK.setText("$topK")
                    Toast.makeText(this@MainActivity, "请输入 1-100 之间的整数", Toast.LENGTH_SHORT).show()
                    true
                }
            }

            binding.buttonClearHistory.setOnClickListener {
                conversationHistory.clear()
                chatAdapter.clearMessages()
                chatAdapter.addSystemMessage("对话历史已清空")
                Toast.makeText(this, "对话历史已清空", Toast.LENGTH_SHORT).show()
                if (isModelLoaded) {
                    try {
                        nativeResetChatHistory()
                    } catch (e: Exception) {
                    }
                }
            }

            binding.switchNetwork.isChecked = enableNetwork
            binding.switchNetwork.setOnCheckedChangeListener { _, isChecked ->
                enableNetwork = isChecked
                saveCurrentHistory()
                Toast.makeText(this, if (isChecked) "联网功能已启用" else "联网功能已禁用", Toast.LENGTH_SHORT).show()
            }

            loadDarkModePreference()
            binding.switchDarkMode.isChecked = isDarkMode
            binding.switchDarkMode.setOnCheckedChangeListener { _, isChecked ->
                if (isDarkMode != isChecked) {
                    isDarkMode = isChecked
                    saveDarkModePreference()
                    recreate()
                }
            }

            loadUserSettings()

            updateSystemPromptDisplay()

            // 启动时自动加载模型
            Log.d("MainActivity", "检查是否需要自动加载模型: isModelLoaded=$isModelLoaded, cachedModelPath=$cachedModelPath")
            if (!isModelLoaded && cachedModelPath != null) {
                val modelFile = File(cachedModelPath!!)
                Log.d("MainActivity", "模型文件存在: ${modelFile.exists()}")
                if (modelFile.exists()) {
                    Log.d("MainActivity", "启动时自动加载模型: $cachedModelPath")
                    Toast.makeText(this, "正在自动加载模型...", Toast.LENGTH_SHORT).show()
                    loadModelInBackground(clearChat = false)
                } else {
                    Log.d("MainActivity", "模型文件不存在: $cachedModelPath")
                }
            } else {
                Log.d("MainActivity", "跳过自动加载模型: isModelLoaded=$isModelLoaded, cachedModelPath=$cachedModelPath")
            }

            Toast.makeText(this, "初始化成功", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("MainActivity", "初始化失败", e)
            e.printStackTrace()
            Toast.makeText(this, "初始化失败: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (::drawerToggle.isInitialized && drawerToggle.onOptionsItemSelected(item)) {
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    override fun onBackPressed() {
        onBackPressedDispatcher.onBackPressed()
    }

    private fun showChatView() {
        binding.chatView.visibility = View.VISIBLE
        binding.historyView.visibility = View.GONE
        binding.modelsView.visibility = View.GONE
        binding.recognitionView.visibility = View.GONE

        // 如果模型未加载且有缓存的模型路径，自动加载模型
        if (!isModelLoaded && cachedModelPath != null) {
            val modelFile = File(cachedModelPath!!)
            if (modelFile.exists()) {
                loadModelInBackground(clearChat = false)
            }
        }
    }

    private fun showHistoryView() {
        binding.chatView.visibility = View.GONE
        binding.historyView.visibility = View.VISIBLE
        binding.modelsView.visibility = View.GONE
        binding.recognitionView.visibility = View.GONE
        loadHistoryList()
    }

    private fun showModelsView() {
        binding.chatView.visibility = View.GONE
        binding.historyView.visibility = View.GONE
        binding.modelsView.visibility = View.VISIBLE
        binding.recognitionView.visibility = View.GONE
        updateSettingsModelDisplay()
    }

    private fun showRecognitionView() {
        binding.chatView.visibility = View.GONE
        binding.historyView.visibility = View.GONE
        binding.modelsView.visibility = View.GONE
        binding.recognitionView.visibility = View.VISIBLE
    }

    private fun loadUserSettings() {
        val prefs = getSharedPreferences("user_settings", MODE_PRIVATE)
        userNickname = prefs.getString("user_nickname", "用户") ?: "用户"
        userAvatarPath = prefs.getString("user_avatar_path", null)
        aiNickname = prefs.getString("ai_nickname", "AI助手") ?: "AI助手"
        aiAvatarPath = prefs.getString("ai_avatar_path", null)
        maxTokens = prefs.getInt("max_tokens", 512)
        contextSize = prefs.getInt("context_size", 8192)
        temperature = prefs.getFloat("temperature", 0.7f)
        topP = prefs.getFloat("top_p", 0.9f)
        topK = prefs.getInt("top_k", 40)
        
        // 加载缓存的模型路径
        cachedModelPath = prefs.getString("cached_model_path", null)
        cachedMmprojPath = prefs.getString("cached_mmproj_path", null)
        Log.d("MainActivity", "加载缓存的模型路径: $cachedModelPath")
        Log.d("MainActivity", "加载缓存的 mmproj 路径: $cachedMmprojPath")

        // 更新聊天适配器
        chatAdapter.userNickname = userNickname
        chatAdapter.userAvatarPath = userAvatarPath
        chatAdapter.aiNickname = aiNickname
        chatAdapter.aiAvatarPath = aiAvatarPath

        // 更新侧边栏显示
        updateUserDisplay()
    }

    private fun saveUserSettings() {
        val prefs = getSharedPreferences("user_settings", MODE_PRIVATE)
        prefs.edit()
            .putString("user_nickname", userNickname)
            .putString("user_avatar_path", userAvatarPath)
            .putString("ai_nickname", aiNickname)
            .putString("ai_avatar_path", aiAvatarPath)
            .putInt("max_tokens", maxTokens)
            .putInt("context_size", contextSize)
            .putFloat("temperature", temperature)
            .putFloat("top_p", topP)
            .putInt("top_k", topK)
            .putString("cached_model_path", cachedModelPath)
            .putString("cached_mmproj_path", cachedMmprojPath)
            .commit()
    }

    private fun startNewChat() {
        currentHistoryId = ""
        conversationHistory.clear()
        chatAdapter.clearMessages()
        chatAdapter.addSystemMessage("请先加载模型，然后开始对话")
        showChatView()
        binding.bottomNav.selectedItemId = R.id.navigation_chat
    }

    private fun displayImage(uri: Uri) {
        try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            binding.imageView.setImageBitmap(bitmap)
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "加载图片失败", Toast.LENGTH_SHORT).show()
        }
    }

    private fun sendMessage(message: String) {
        binding.editTextInput.text?.clear()
        val hasImage = currentImageUri != null
        val imageUriToProcess = currentImageUri

        chatAdapter.addUserMessage(message, imageUriToProcess)

        if (message.isNotEmpty() || hasImage) {
            val formattedMessage = "[用户]|||$message"
            conversationHistory.add(formattedMessage)
            saveCurrentHistory()
        }

        currentImageUri = null
        binding.imageSection.visibility = View.GONE
        binding.imageView.setImageURI(null)

        Thread {
            try {
                var finalMessage = message

                if (enableNetwork && !hasImage && message.isNotEmpty()) {
                    Handler(Looper.getMainLooper()).post {
                        addMessageToChat("系统", "正在联网搜索相关信息...")
                    }

                    try {
                        val searchResults = performSearch(message)
                        if (searchResults != null && searchResults.isNotEmpty()) {
                            finalMessage = "用户问题: $message\n\n网络搜索结果:\n$searchResults\n\n请基于以上信息回答用户的问题。"
                            Handler(Looper.getMainLooper()).post {
                                addMessageToChat("系统", "搜索完成，已获取相关信息")
                            }
                        } else {
                            Handler(Looper.getMainLooper()).post {
                                addMessageToChat("系统", "搜索未返回有效结果，使用本地知识回答")
                            }
                        }
                    } catch (e: Exception) {
                        Handler(Looper.getMainLooper()).post {
                            addMessageToChat("系统", "搜索失败: ${e.message}，使用本地知识回答")
                        }
                    }
                }

                if (hasImage && imageUriToProcess != null) {
                    // 初始化流式输出回调
                    nativeInitStreamingCallback(object : StreamingCallback {
                        override fun onToken(token: String) {
                            Handler(Looper.getMainLooper()).post {
                                // 更新最后一条 AI 消息
                                chatAdapter.updateLastAIMessage(token)
                            }
                        }
                    })

                    // 添加一个空的 AI 消息用于流式输出
                    Handler(Looper.getMainLooper()).post {
                        chatAdapter.addAIMessage("")
                    }

                    Thread {
                        try {
                            val inputStream = contentResolver.openInputStream(imageUriToProcess)
                            val bitmap = BitmapFactory.decodeStream(inputStream)
                            inputStream?.close()

                            if (bitmap != null) {
                                // 先缩放图片，限制最大尺寸为 800px，减少数据传输和处理时间
                                val maxDimension = performanceManager.getRecommendedImageMaxDimension()
                                val ratio = maxDimension.toFloat() / maxOf(bitmap.width, bitmap.height)
                                val scaledBitmap = if (ratio < 1f) {
                                    Bitmap.createScaledBitmap(bitmap,
                                        (bitmap.width * ratio).toInt(),
                                        (bitmap.height * ratio).toInt(), true)
                                } else {
                                    bitmap
                                }

                                // 优化：直接从 Bitmap 写入 ByteBuffer，无 IntArray 中间变量
                                val argbBufferSize = scaledBitmap.width * scaledBitmap.height * 4
                                val argbBuffer = java.nio.ByteBuffer.allocateDirect(argbBufferSize)
                                scaledBitmap.copyPixelsToBuffer(argbBuffer)
                                argbBuffer.flip()

                                val result = nativeAnalyzeImageDirect(argbBuffer, scaledBitmap.width, scaledBitmap.height, message)
                                Handler(Looper.getMainLooper()).post {
                                    conversationHistory.add("[助手]|||$result")
                                    saveCurrentHistory()
                                }
                            }
                        } catch (e: Exception) {
                            Handler(Looper.getMainLooper()).post {
                                addMessageToChat("系统", "图片分析失败: ${e.message}")
                            }
                        } finally {
                            // 清理流式输出回调
                            nativeCleanupStreamingCallback()
                        }
                    }.start()
                } else {
                    // 所有文本聊天都使用流式输出
                    Log.d("MainActivity", "走流式输出分支")
                    initStreamingChat(finalMessage)
                }
            } catch (e: Exception) {
                Handler(Looper.getMainLooper()).post {
                    addMessageToChat("系统", "消息发送失败: ${e.message}")
                }
            }
        }.start()
    }

    private fun addMessageToChat(sender: String, message: String, updateUI: Boolean = true) {
        if (updateUI) {
            when (sender) {
                "用户" -> chatAdapter.addUserMessage(message)
                "助手" -> chatAdapter.addAIMessage(message)
                "系统" -> chatAdapter.addSystemMessage(message)
            }
        }
    }

    private fun performSearch(query: String): String? {
        var searchResults: String? = null

        return try {
            networkHelper.search(query) { results, error ->
                if (error != null) {
                    throw error
                }
                searchResults = results
                Handler(Looper.getMainLooper()).post {
                    if (results?.isNotEmpty() == true) {
                        chatAdapter.addSystemMessage("搜索结果:\n$results")
                    }
                }
            }
            searchResults
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun initStreamingChat(message: String) {
        Log.d("MainActivity", "initStreamingChat 被调用")
        
        // 添加一个空的 AI 消息用于流式输出
        chatAdapter.addAIMessage("")

        // 获取主线程的 Handler
        val mainHandler = Handler(Looper.getMainLooper())

        // 在主线程中初始化流式回调
        try {
            nativeInitStreamingCallback(object : StreamingCallback {
                override fun onToken(token: String) {
                    Log.d("MainActivity", "收到回调: $token")
                    // 确保所有 UI 更新都在主线程中执行
                    mainHandler.post {
                        try {
                            chatAdapter.updateLastAIMessage(token)
                        } catch (e: Exception) {
                            Log.e("MainActivity", "更新 UI 失败", e)
                        }
                    }
                }
            })
            Log.d("MainActivity", "流式回调初始化完成")
        } catch (e: Exception) {
            Log.e("MainActivity", "初始化流式回调失败", e)
            addMessageToChat("系统", "初始化流式回调失败: ${e.message}")
            return
        }

        // 在后台线程中调用 nativeChat
        Thread {
            Log.d("MainActivity", "开始调用 nativeChat")
            try {
                val response = nativeChat(message, false)
                Log.d("MainActivity", "nativeChat 返回，长度: ${response.length}")
                
                runOnUiThread {
                    conversationHistory.add("[助手]|||$response")
                    saveCurrentHistory()
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "聊天失败", e)
                runOnUiThread {
                    addMessageToChat("系统", "聊天失败: ${e.message}")
                }
            }
            
            // 延迟清理流式回调，确保所有回调都已完成
            Thread.sleep(500)
            
            Log.d("MainActivity", "清理流式回调")
            // 清理流式输出回调
            try {
                nativeCleanupStreamingCallback()
            } catch (e: Exception) {
                Log.e("MainActivity", "清理流式回调失败", e)
            }
        }.start()
    }

    private fun simulateStreamingOutput(message: String): String {
        Handler(Looper.getMainLooper()).post {
            chatAdapter.addAIMessage("")
        }

        val response = nativeChat(message, false)

        var displayedText = ""
        val chars = response.toList()

        for ((index, char) in chars.withIndex()) {
            displayedText += char

            if (index % 3 == 0 || index == chars.size - 1) {
                val finalText = displayedText
                Handler(Looper.getMainLooper()).post {
                    chatAdapter.updateLastAIMessage(finalText)
                }

                Thread.sleep(10)
            }
        }

        conversationHistory.add("[助手]|||$response")
        saveCurrentHistory()

        Handler(Looper.getMainLooper()).post {
            chatAdapter.updateLastAIMessageMeta(tokenCount = response.length, duration = 0)
        }

        return response
    }

    private fun copyFileToPrivateDirWithProgress(uri: Uri, isModel: Boolean = true, progressDialog: ModelLoadProgressDialog?, targetDirName: String = "models"): String? {
        return try {
            val inputStream = contentResolver.openInputStream(uri) ?: return null

            val filesDir = File(filesDir, targetDirName)
            if (!filesDir.exists()) {
                filesDir.mkdirs()
            }

            val fileName = try {
                val projection = arrayOf(android.provider.OpenableColumns.DISPLAY_NAME)
                val cursor = contentResolver.query(uri, projection, null, null, null)
                cursor?.use {
                    if (it.moveToFirst()) {
                        it.getString(0)
                    } else null
                } ?: if (isModel) "model_${System.currentTimeMillis()}.gguf" else "mmproj_${System.currentTimeMillis()}.gguf"
            } catch (e: Exception) {
                if (isModel) "model_${System.currentTimeMillis()}.gguf" else "mmproj_${System.currentTimeMillis()}.gguf"
            }

            val targetFile = File(filesDir, fileName)
            val outputStream = FileOutputStream(targetFile)

            val buffer = ByteArray(8192 * 8)
            var bytesRead: Int
            var totalBytes = 0L
            val fileTypeName = if (isModel) "模型" else "多模态"

            val fileSize = try {
                contentResolver.openFileDescriptor(uri, "r")?.use { it.statSize } ?: -1L
            } catch (e: Exception) {
                -1L
            }

            inputStream.use { input ->
                outputStream.use { output ->
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        if (progressDialog?.isCancelled == true) {
                            return null
                        }

                        output.write(buffer, 0, bytesRead)
                        totalBytes += bytesRead

                        if (fileSize > 0 && progressDialog != null && !progressDialog.isCancelled) {
                            val progress = ((totalBytes * 100) / fileSize).toInt()
                            val sizeMB = totalBytes / (1024 * 1024)
                            val totalMB = fileSize / (1024 * 1024)
                            val pd = progressDialog
                            Handler(Looper.getMainLooper()).post {
                                try {
                                    if (!pd.isCancelled) {
                                        pd.setMessage("正在复制${fileTypeName}文件...\n${sizeMB}MB / ${totalMB}MB")
                                        pd.setProgress(progress)
                                    }
                                } catch (e: Exception) {
                                }
                            }
                        }
                    }
                }
            }

            targetFile.absolutePath
        } catch (e: Exception) {
            null
        }
    }

    private fun loadHistoryList() {
        val historyItems = historyHelper.getHistoryList()
        historyAdapter.updateItems(historyItems)
    }

    private fun loadHistory(historyItem: HistoryItem) {
        try {
            Log.d("MainActivity", "开始加载历史记录: ${historyItem.title}")

            val historyText = String(android.util.Base64.decode(historyItem.data, android.util.Base64.DEFAULT))
            conversationHistory.clear()
            conversationHistory.addAll(historyText.split("\n").filter { it.isNotEmpty() })

            Log.d("MainActivity", "解析到 ${conversationHistory.size} 条历史记录")

            chatAdapter.clearMessages()
            conversationHistory.forEach { entry ->
                try {
                    if (entry.startsWith("[用户]|||")) {
                        val message = entry.substringAfter("[用户]|||")
                        chatAdapter.addUserMessage(message)
                    } else if (entry.startsWith("[助手]|||")) {
                        val message = entry.substringAfter("[助手]|||")
                        chatAdapter.addAIMessage(message)
                    } else if (entry.startsWith("[系统]|||")) {
                        val message = entry.substringAfter("[系统]|||")
                        chatAdapter.addSystemMessage(message)
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "解析历史记录条目失败: $entry", e)
                }
            }

            systemPrompt = historyItem.systemPrompt
            maxTokens = historyItem.maxTokens
            contextSize = historyItem.contextSize
            enableNetwork = historyItem.enableNetwork

            currentHistoryId = historyItem.id

            // 立即切换到聊天界面
            showChatView()
            binding.bottomNav.selectedItemId = R.id.navigation_chat

            if (historyItem.modelPath.isNotEmpty()) {
                val modelFile = File(historyItem.modelPath)
                if (modelFile.exists()) {
                    cachedModelPath = historyItem.modelPath
                    cachedMmprojPath = if (historyItem.mmprojPath.isNotEmpty()) historyItem.mmprojPath else null

                    Log.d("MainActivity", "开始加载模型: $cachedModelPath")
                    // 在后台加载模型
                    loadModelInBackground(
                        clearChat = false,
                        onLoadSuccess = {
                            // 模型加载成功后，恢复对话历史到 JNI 层
                            loadConversationHistoryToJNI { success ->
                                if (success) {
                                    Log.d("MainActivity", "历史记录上下文恢复成功")
                                } else {
                                    Log.e("MainActivity", "历史记录上下文恢复失败")
                                }
                            }
                        },
                        onLoadFailed = { error ->
                            // 模型加载失败
                            runOnUiThread {
                                Toast.makeText(this, "模型加载失败: $error", Toast.LENGTH_LONG).show()
                            }
                        }
                    )
                } else {
                    Log.w("MainActivity", "模型文件不存在: ${historyItem.modelPath}")
                    Toast.makeText(this, "模型文件不存在", Toast.LENGTH_SHORT).show()
                }
            } else {
                // 没有模型路径
                if (isModelLoaded) {
                    // 模型已加载，恢复对话历史到 JNI 层
                    loadConversationHistoryToJNI { success ->
                        if (success) {
                            Log.d("MainActivity", "历史记录上下文恢复成功")
                            runOnUiThread {
                                Toast.makeText(this, "已加载历史记录", Toast.LENGTH_SHORT).show()
                            }
                        } else {
                            Log.e("MainActivity", "历史记录上下文恢复失败")
                            runOnUiThread {
                                Toast.makeText(this, "加载历史记录失败", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                } else {
                    Toast.makeText(this, "已加载历史记录（无模型）", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "加载历史记录失败", e)
            Toast.makeText(this, "加载失败: ${e.message}", Toast.LENGTH_SHORT).show()
            // 恢复到安全状态
            conversationHistory.clear()
            chatAdapter.clearMessages()
            chatAdapter.addSystemMessage("加载历史记录失败，请重试")
        }
    }

    private fun deleteHistory(historyItem: HistoryItem) {
        try {
            historyHelper.deleteHistory(historyItem.id)
            historyAdapter.removeItem(historyItem)
            Toast.makeText(this, "已删除", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "删除失败", Toast.LENGTH_SHORT).show()
        }
    }

    private fun saveCurrentHistory() {
        if (conversationHistory.isEmpty()) return

        val historyId = if (currentHistoryId.isEmpty()) {
            System.currentTimeMillis().toString()
        } else {
            currentHistoryId
        }

        val historyText = conversationHistory.joinToString("\n")
        val historyData = android.util.Base64.encodeToString(historyText.toByteArray(), android.util.Base64.DEFAULT)

        try {
            val title = if (conversationHistory.size >= 2) {
                conversationHistory[1].substringAfter("|||").take(20)
            } else {
                "新对话"
            }

            val historyItem = HistoryItem(
                id = historyId,
                title = title,
                date = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date()),
                preview = historyText.take(100),
                data = historyData,
                modelPath = cachedModelPath ?: "",
                mmprojPath = cachedMmprojPath ?: "",
                maxTokens = maxTokens,
                contextSize = contextSize,
                enableNetwork = enableNetwork,
                systemPrompt = systemPrompt
            )

            historyHelper.saveHistory(historyItem)
            currentHistoryId = historyId
        } catch (e: Exception) {
        }
    }

    private fun loadModelInBackground(
        clearChat: Boolean = true,
        onLoadSuccess: (() -> Unit)? = null,
        onLoadFailed: ((String) -> Unit)? = null
    ) {
        if (cachedModelPath == null) {
            Toast.makeText(this, "请先选择模型", Toast.LENGTH_LONG).show()
            onLoadFailed?.invoke("未选择模型")
            return
        }

        if (binding.drawerLayout.isDrawerOpen(GravityCompat.START)) {
            binding.drawerLayout.closeDrawer(GravityCompat.START)
        }

        binding.textViewModelStatus.text = "模型状态: 正在加载..."

        if (clearChat) {
            chatAdapter.clearMessages()
            chatAdapter.addSystemMessage("正在加载模型到内存...\n这可能需要几分钟时间，请稍候...")
        }

        Thread {
            try {
                // 设置性能参数到 C++ 层
                try {
                    nativeSetPerformanceParams(
                        performanceManager.getRecommendedThreads(),
                        performanceManager.getRecommendedBatchSize(),
                        performanceManager.getRecommendedGpuLayers(),
                        performanceManager.shouldEnableFlashAttention()
                    )
                    Log.d("MainActivity", "性能参数已设置到 C++ 层")
                } catch (e: Exception) {
                    Log.e("MainActivity", "设置性能参数失败: ${e.message}")
                }

                val modelPath = cachedModelPath!!
                val mmprojPath = cachedMmprojPath

                val modelFile = File(modelPath)
                if (!modelFile.exists()) {
                    Handler(Looper.getMainLooper()).post {
                        binding.textViewModelStatus.text = "模型状态: 加载失败"
                        if (clearChat) {
                            chatAdapter.clearMessages()
                            chatAdapter.addSystemMessage("模型文件不存在: $modelPath")
                        }
                        Toast.makeText(this, "模型文件不存在", Toast.LENGTH_LONG).show()
                        onLoadFailed?.invoke("模型文件不存在")
                    }
                    return@Thread
                }

                val result = nativeLoadModel(modelPath, mmprojPath ?: "", systemPrompt, maxTokens, contextSize, temperature, topP, topK)

                Handler(Looper.getMainLooper()).post {
                    if (clearChat) {
                        chatAdapter.clearMessages()
                    }
                    Toast.makeText(this, "加载完成", Toast.LENGTH_SHORT).show()

                    if (result.contains("成功", ignoreCase = true)) {
                        isModelLoaded = true
                        binding.textViewModelStatus.text = "模型状态: 已加载"
                        updateSettingsModelDisplay()

                        if (clearChat) {
                            conversationHistory.clear()
                        }

                        // 显示模型加载成功信息
                        val modelName = File(modelPath).name
                        val mmprojName = if (!mmprojPath.isNullOrEmpty()) {
                            File(mmprojPath).name
                        } else {
                            "未使用"
                        }
                        val networkStatus = if (enableNetwork) "已启用" else "未启用"

                        // 显示成功消息（无论是否清空聊天）
                        val successMessage = """模型加载成功！

当前模型: $modelName
多模态: $mmprojName
系统提示词: ${systemPrompt.take(50)}${if (systemPrompt.length > 50) "..." else ""}
上下文长度: $contextSize
最大 Token 数: $maxTokens
联网搜索功能: $networkStatus"""

                        if (clearChat) {
                            // 在聊天页面加载时显示详细信息
                            chatAdapter.addSystemMessage(successMessage)
                            // 调用成功回调
                            onLoadSuccess?.invoke()
                        } else {
                            // 在识别页面或加载历史记录时，先加载历史记录再显示消息
                            if (conversationHistory.isNotEmpty()) {
                                try {
                                    loadConversationHistoryToJNI { _ ->
                                        // 历史记录加载完成后才显示消息并调用回调
                                        chatAdapter.addSystemMessage(successMessage)
                                        onLoadSuccess?.invoke()
                                    }
                                } catch (e: Exception) {
                                    Log.e("MainActivity", "加载历史记录到JNI时发生异常", e)
                                    // 即使失败也显示消息并调用回调
                                    chatAdapter.addSystemMessage(successMessage)
                                    onLoadSuccess?.invoke()
                                }
                            } else {
                                // 没有历史记录，直接显示消息并调用回调
                                Log.i("MainActivity", "模型加载成功: $modelName")
                                chatAdapter.addSystemMessage(successMessage)
                                onLoadSuccess?.invoke()
                            }
                        }
                    } else {
                        binding.textViewModelStatus.text = "模型状态: 加载失败"
                        if (clearChat) {
                            chatAdapter.clearMessages()
                        }
                        chatAdapter.addSystemMessage("加载结果:\n$result\n\n模型路径: $modelPath\n多模态路径: ${mmprojPath?.ifEmpty { "未选择（单文件模式）" } ?: "未选择（单文件模式）"}")
                        onLoadFailed?.invoke(result)
                    }
                }
            } catch (e: Exception) {
                Handler(Looper.getMainLooper()).post {
                    binding.textViewModelStatus.text = "模型状态: 加载失败"
                    if (clearChat) {
                        chatAdapter.clearMessages()
                        chatAdapter.addSystemMessage("加载模型异常: ${e.message}\n\n${e.stackTraceToString()}")
                    }
                    Toast.makeText(this, "加载模型失败: ${e.message}", Toast.LENGTH_LONG).show()
                    onLoadFailed?.invoke(e.message ?: "未知错误")
                }
            }
        }.start()
    }

    private fun loadConversationHistoryToJNI(onComplete: ((Boolean) -> Unit)? = null) {
        if (!isModelLoaded) {
            Log.w("MainActivity", "模型未加载，跳过恢复上下文")
            onComplete?.invoke(false)
            return
        }

        if (conversationHistory.isEmpty()) {
            Log.w("MainActivity", "对话历史为空，跳过恢复上下文")
            onComplete?.invoke(true)
            return
        }

        isHistoryLoadingToJNI = true
        Log.d("MainActivity", "开始加载对话历史到JNI")

        Thread {
            try {
                Thread.sleep(500)

                val historyString = conversationHistory.joinToString("")
                Log.d("MainActivity", "历史记录字符串长度: ${historyString.length}")

                val result = nativeRestoreContext(historyString)
                Log.d("MainActivity", "nativeRestoreContext 结果: $result")

                if (result.contains("失败") || result.contains("错误")) {
                    Handler(Looper.getMainLooper()).post {
                        Log.e("MainActivity", "加载历史记录上下文失败: $result")
                        isHistoryLoadingToJNI = false
                        onComplete?.invoke(false)
                    }
                } else {
                    Handler(Looper.getMainLooper()).post {
                        Log.d("MainActivity", "历史记录已加载到JNI")
                        isHistoryLoadingToJNI = false
                        onComplete?.invoke(true)
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "加载历史记录上下文异常", e)
                Handler(Looper.getMainLooper()).post {
                    isHistoryLoadingToJNI = false
                    onComplete?.invoke(false)
                }
            }
        }.start()
    }

    private fun updateSystemPromptDisplay() {
        binding.textViewSystemPrompt.text = "系统提示词: $systemPrompt"
    }

    private fun saveDarkModePreference() {
        val prefs = getSharedPreferences("app_settings", MODE_PRIVATE)
        prefs.edit().putBoolean("dark_mode", isDarkMode).apply()
    }

    private fun loadDarkModePreference() {
        val prefs = getSharedPreferences("app_settings", MODE_PRIVATE)
        isDarkMode = prefs.getBoolean("dark_mode", false)
        AppCompatDelegate.setDefaultNightMode(if (isDarkMode) AppCompatDelegate.MODE_NIGHT_YES else AppCompatDelegate.MODE_NIGHT_NO)
    }

    private fun showSystemPromptDialog() {
        val editText = EditText(this)
        editText.setText(systemPrompt)
        editText.minLines = 3
        editText.maxLines = 10
        editText.gravity = android.view.Gravity.TOP or android.view.Gravity.START

        android.app.AlertDialog.Builder(this)
            .setTitle("设置系统提示词")
            .setView(editText)
            .setPositiveButton("确定") { _, _ ->
                val newPrompt = editText.text.toString().trim()
                if (newPrompt.isNotEmpty()) {
                    systemPrompt = newPrompt
                    updateSystemPromptDisplay()
                    Toast.makeText(this, "系统提示词已更新", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()

        // 停止摄像头
        try {
            stopCameraCapture()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // 关闭线程池
        try {
            cameraExecutor?.shutdown()
            cameraExecutor = null
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // 关闭姿态检测器
        try {
            poseDetector?.close()
            poseDetector = null
        } catch (e: Exception) {
            e.printStackTrace()
        }

        if (isModelLoaded) {
            try {
                nativeFreeModel()
            } catch (e: Exception) {
            }
        }
    }

    private fun showChangeNicknameDialog() {
        val editText = android.widget.EditText(this)
        editText.hint = "请输入昵称"
        editText.setText(userNickname)
        editText.setSelection(userNickname.length)

        android.app.AlertDialog.Builder(this)
            .setTitle("修改昵称")
            .setView(editText)
            .setPositiveButton("确定") { _, _ ->
                val newName = editText.text.toString().trim()
                if (newName.isNotEmpty()) {
                    userNickname = newName
                    chatAdapter.userNickname = userNickname
                    updateUserDisplay()
                    saveUserSettings()
                    Toast.makeText(this, "昵称已更新", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun showChangeAvatarDialog() {
        android.app.AlertDialog.Builder(this)
            .setTitle("更换头像")
            .setMessage("选择一张新图片作为头像")
            .setPositiveButton("选择图片") { _, _ ->
                pickUserAvatarLauncher.launch("image/*")
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun saveUserAvatar(uri: Uri) {
        Thread {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                if (inputStream == null) {
                    Handler(Looper.getMainLooper()).post {
                        Toast.makeText(this, "无法打开图片", Toast.LENGTH_SHORT).show()
                    }
                    return@Thread
                }

                val avatarDir = File(filesDir, "avatars")
                if (!avatarDir.exists()) {
                    avatarDir.mkdirs()
                }
                val avatarFile = File(avatarDir, "user_avatar_${System.currentTimeMillis()}.jpg")

                inputStream.use { input ->
                    FileOutputStream(avatarFile).use { output ->
                        input.copyTo(output)
                    }
                }

                if (avatarFile.exists() && avatarFile.length() > 0) {
                    userAvatarPath = avatarFile.absolutePath
                    
                    Handler(Looper.getMainLooper()).post {
                        chatAdapter.userAvatarPath = userAvatarPath
                        updateUserDisplay()
                        saveUserSettings()
                        Toast.makeText(this, "头像已更新", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    Handler(Looper.getMainLooper()).post {
                        Toast.makeText(this, "保存头像失败", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "保存用户头像失败", e)
                Handler(Looper.getMainLooper()).post {
                    Toast.makeText(this, "保存头像失败: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    private fun updateUserDisplay() {
        binding.textViewUserNickname.text = userNickname
        
        val avatarPath = userAvatarPath
        if (avatarPath != null) {
            try {
                val file = File(avatarPath)
                if (file.exists()) {
                    val bitmap = BitmapFactory.decodeFile(avatarPath)
                    if (bitmap != null) {
                        binding.imageViewUserAvatar.setImageBitmap(bitmap)
                        binding.imageViewUserAvatar.setBackgroundColor(0x00000000)
                        return
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "加载头像失败", e)
            }
        }
        Log.d("MainActivity", "使用默认头像")
        binding.imageViewUserAvatar.setImageDrawable(null)
        binding.imageViewUserAvatar.setBackgroundColor(0xFF333333.toInt())
    }

    

    private fun updateSettingsModelDisplay() {
        // 更新模型显示
        if (cachedModelPath != null) {
            val modelFile = File(cachedModelPath!!)
            if (modelFile.exists()) {
                binding.textViewCurrentModel.text = modelFile.name
                val sizeKB = modelFile.length() / 1024
                val sizeMB = sizeKB / 1024
                val sizeGB = sizeMB / 1024
                val sizeText = when {
                    sizeGB > 0 -> String.format("%.2f GB", sizeGB + (sizeMB % 1024) / 1024.0)
                    sizeMB > 0 -> String.format("%.2f MB", sizeMB + (sizeKB % 1024) / 1024.0)
                    else -> String.format("%.2f KB", sizeKB.toDouble())
                }
                binding.textViewModelSize.text = "大小: $sizeText"
            } else {
                binding.textViewCurrentModel.text = "模型文件不存在"
                binding.textViewModelSize.text = ""
            }
        } else {
            binding.textViewCurrentModel.text = "未加载模型"
            binding.textViewModelSize.text = ""
        }

        // 更新多模态显示
        if (cachedMmprojPath != null) {
            val mmprojFile = File(cachedMmprojPath!!)
            if (mmprojFile.exists()) {
                binding.textViewCurrentMmproj.text = mmprojFile.name
                val sizeKB = mmprojFile.length() / 1024
                val sizeMB = sizeKB / 1024
                val sizeGB = sizeMB / 1024
                val sizeText = when {
                    sizeGB > 0 -> String.format("%.2f GB", sizeGB + (sizeMB % 1024) / 1024.0)
                    sizeMB > 0 -> String.format("%.2f MB", sizeMB + (sizeKB % 1024) / 1024.0)
                    else -> String.format("%.2f KB", sizeKB.toDouble())
                }
                binding.textViewMmprojSize.text = "大小: $sizeText"
            } else {
                binding.textViewCurrentMmproj.text = "多模态文件不存在"
                binding.textViewMmprojSize.text = ""
            }
        } else {
            binding.textViewCurrentMmproj.text = "未加载多模态文件"
            binding.textViewMmprojSize.text = ""
        }
    }

    

    // 缓存缩放后的识别图片，避免重复缩放
    private var recognitionBitmap: Bitmap? = null

    private fun displayRecognitionImage(uri: Uri) {
        try {
            val inputStream = contentResolver.openInputStream(uri)
            val originalBitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            if (originalBitmap != null) {
                // 压缩图片到合适的大小 (最大宽度800px) - 只缩放一次
                val maxDimension = 800
                val ratio = maxDimension.toFloat() / maxOf(originalBitmap.width, originalBitmap.height)
                val newWidth = (originalBitmap.width * ratio).toInt()
                val newHeight = (originalBitmap.height * ratio).toInt()

                // 直接创建为可变的 ARGB_8888 格式，后续骨骼绘制无需再复制
                val compressedBitmap = if (ratio < 1f) {
                    Bitmap.createScaledBitmap(originalBitmap, newWidth, newHeight, true)
                } else {
                    originalBitmap
                }
                
                // 缓存缩放后的 bitmap
                recognitionBitmap = compressedBitmap
                binding.imageViewRecognition.setImageBitmap(compressedBitmap)

                binding.recognitionImageSection.visibility = View.VISIBLE
                binding.textViewRecognitionPlaceholder.visibility = View.GONE
                binding.textViewRecognitionResult.text = "等待识别..."

                // 启用开始分析按钮
                binding.buttonStartRecognitionAnalysis.isEnabled = true
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "加载图片失败", Toast.LENGTH_SHORT).show()
        }
    }

    private fun performRecognitionAnalysis() {
        // 优先使用缓存的 bitmap，避免从 ImageView 获取
        val bitmap = recognitionBitmap
        if (bitmap == null) {
            // 降级：从 ImageView 获取
            val drawable = binding.imageViewRecognition.drawable
            if (drawable == null) {
                Toast.makeText(this, "请先选择或拍摄图片", Toast.LENGTH_SHORT).show()
                return
            }
            val ivBitmap = (drawable as android.graphics.drawable.BitmapDrawable).bitmap
            performActionRecognition(ivBitmap)
        } else {
            performActionRecognition(bitmap)
        }
    }

    private fun performActionRecognition(bitmap: Bitmap) {
        if (!isModelLoaded) {
            // 检查是否有缓存的模型路径
            if (cachedModelPath != null && File(cachedModelPath!!).exists()) {
                binding.textViewRecognitionResult.text = "正在加载模型..."
                Toast.makeText(this, "正在加载模型，请稍候...", Toast.LENGTH_SHORT).show()
                
                // 加载模型
                loadModelInBackground(clearChat = false, onLoadSuccess = {
                    // 模型加载成功后，执行识别
                    runOnUiThread {
                        performImageAnalysis(bitmap)
                    }
                }, onLoadFailed = { error ->
                    // 模型加载失败
                    runOnUiThread {
                        binding.textViewRecognitionResult.text = "模型加载失败: $error"
                        Toast.makeText(this, "模型加载失败: $error", Toast.LENGTH_LONG).show()
                    }
                })
            } else {
                binding.textViewRecognitionResult.text = "请先加载模型"
                Toast.makeText(this, "请先在模型页面加载模型", Toast.LENGTH_LONG).show()
                return
            }
        } else {
            // 模型已加载，直接执行识别
            performImageAnalysis(bitmap)
        }
    }

    private fun performImageAnalysis(bitmap: Bitmap) {
        binding.textViewRecognitionResult.text = "正在识别..."

        // 在后台线程执行骨骼检测和图片处理
        Thread {
            val startTime = System.currentTimeMillis()
            Log.d("MainActivity", "=== 开始图片识别流程 (优化版) ===")
            Log.d("MainActivity", "输入图片尺寸: ${bitmap.width}x${bitmap.height}, 配置: ${bitmap.config}")
            
            try {
                // 第一步：检测骨骼姿态并直接绘制到 bitmap 上 (in-place)
                val poseStartTime = System.currentTimeMillis()
                var poseInfo = ""
                var processedBitmap: Bitmap = bitmap  // 默认使用原始 bitmap

                try {
                    val pose = detectPoseSync(bitmap)
                    if (pose != null) {
                        // in-place 绘制骨骼，无需复制 bitmap
                        processedBitmap = drawPoseLandmarksInPlace(bitmap, pose)
                        poseInfo = buildPoseInfoText(pose)
                        Log.d("MainActivity", "骨骼绘制完成，尺寸: ${processedBitmap.width}x${processedBitmap.height}")
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "骨骼检测失败", e)
                }
                val poseEndTime = System.currentTimeMillis()
                Log.d("MainActivity", "姿态检测+绘制耗时: ${poseEndTime - poseStartTime}ms")

                // 优化：已删除重复缩放步骤，直接使用已缩放的图片

                // 更新识别界面的图片显示（必须在主线程）
                Handler(Looper.getMainLooper()).post {
                    binding.imageViewRecognition.setImageBitmap(processedBitmap)
                }

                // 第二步：获取像素数据发送给AI（优化：直接转为 RGB ByteBuffer）
                val pixelStartTime = System.currentTimeMillis()
                val imageWidth = processedBitmap.width
                val imageHeight = processedBitmap.height
                
                // 优化：直接从 Bitmap 写入 ByteBuffer，无 IntArray 中间变量
                // 方案：使用 copyPixelsToBuffer 直接写入 ARGB 数据，JNI 端处理转换
                val argbBufferSize = imageWidth * imageHeight * 4  // ARGB 每像素 4 字节
                val argbBuffer = java.nio.ByteBuffer.allocateDirect(argbBufferSize)
                processedBitmap.copyPixelsToBuffer(argbBuffer)
                argbBuffer.flip()
                
                val pixelEndTime = System.currentTimeMillis()
                Log.d("MainActivity", "ARGB数据准备耗时: ${pixelEndTime - pixelStartTime}ms, 数据大小: $argbBufferSize 字节")

                // 第三步：调用 AI 分析
                val aiStartTime = System.currentTimeMillis()
                try {
                    val fullPrompt = if (poseInfo.isNotEmpty()) {
                        "$recognitionPrompt\n\n骨骼姿态信息:\n$poseInfo"
                    } else {
                        recognitionPrompt
                    }

                    // 设置流式回调
                    nativeInitStreamingCallback(object : StreamingCallback {
                        override fun onToken(token: String) {
                            Handler(Looper.getMainLooper()).post {
                                // 使用 Markwon 渲染 Markdown
                                recognitionMarkwon.setMarkdown(binding.textViewRecognitionResult, token)
                                // 智能滚动：只有在用户没有手动向上滚动时才自动滚动
                                if (autoScrollEnabled) {
                                    binding.scrollViewRecognition.post {
                                        // 使用 scrollTo 替代 fullScroll，避免焦点问题
                                        val child = binding.scrollViewRecognition.getChildAt(0)
                                        if (child != null) {
                                            val scrollY = child.height - binding.scrollViewRecognition.height
                                            if (scrollY > 0) {
                                                binding.scrollViewRecognition.scrollTo(0, scrollY)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    })

                    // 流式输出开始时禁用文本选择，避免抢占焦点导致滚动问题
                    binding.textViewRecognitionResult.setTextIsSelectable(false)
                    binding.textViewRecognitionResult.text = "" // 清空之前的文本
                    autoScrollEnabled = true // 新的识别开始时，重置自动滚动状态
                    
                    // 使用优化后的 DirectByteBuffer 方法（ARGB 格式）
                    nativeAnalyzeImageDirect(argbBuffer, imageWidth, imageHeight, fullPrompt)
                    
                    val aiEndTime = System.currentTimeMillis()
                    Log.d("MainActivity", "AI分析耗时: ${aiEndTime - aiStartTime}ms")

                    // 清理流式回调
                    nativeCleanupStreamingCallback()
                    
                    // 流式输出结束后恢复文本选择功能
                    Handler(Looper.getMainLooper()).post {
                        binding.textViewRecognitionResult.setTextIsSelectable(true)
                    }
                } catch (e: Exception) {
                    Handler(Looper.getMainLooper()).post {
                        binding.textViewRecognitionResult.text = "识别失败: ${e.message}"
                    }
                }
                
                val totalTime = System.currentTimeMillis() - startTime
                Log.d("MainActivity", "=== 总耗时: ${totalTime}ms ===")
            } catch (e: Exception) {
                Log.e("MainActivity", "图片分析失败", e)
                Handler(Looper.getMainLooper()).post {
                    binding.textViewRecognitionResult.text = "处理失败: ${e.message}"
                }
            }
        }.start()
    }

    // ========== MediaPipe Pose 相关函数 ==========

    // 同步检测姿态
    private fun detectPoseSync(bitmap: Bitmap): Pose? {
        if (poseDetector == null) {
            Log.w("MainActivity", "PoseDetector 未初始化")
            return null
        }

        var result: Pose? = null
        val latch = CountDownLatch(1)

        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            Log.d("MainActivity", "InputImage 创建完成，尺寸: ${image.width}x${image.height}")

            poseDetector?.process(image)
                ?.addOnSuccessListener { detectedPose: Pose ->
                    // 统计检测到的关键点数量
                    val allLandmarks = detectedPose.allPoseLandmarks
                    Log.d("MainActivity", "骨骼检测成功，检测到 ${allLandmarks.size} 个关键点")

                    // 打印每个关键点的信息
                    for (landmark in allLandmarks) {
                        Log.v("MainActivity", "关键点: ${landmark.landmarkType}, " +
                                "位置: (${landmark.position.x}, ${landmark.position.y}), " +
                                "置信度: ${landmark.inFrameLikelihood}")
                    }

                    result = detectedPose
                    latch.countDown()
                }
                ?.addOnFailureListener { e: Exception ->
                    Log.e("MainActivity", "Pose detection failed", e)
                    latch.countDown()
                }

            // 等待结果，最多等待 2 秒（优化：从 5 秒缩短）
            Log.d("MainActivity", "等待骨骼检测结果...")
            val completed = latch.await(2, TimeUnit.SECONDS)

            if (!completed) {
                Log.w("MainActivity", "骨骼检测超时")
            }
            if (result == null) {
                Log.w("MainActivity", "骨骼检测返回 null")
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Pose detection error", e)
        }

        return result
    }

    // 在图片上绘制骨骼关键点和连线 (优化版：in-place 绘制)
    private fun drawPoseLandmarksInPlace(bitmap: Bitmap, pose: Pose): Bitmap {
        Log.d("MainActivity", "开始绘制骨骼 (in-place)，图片尺寸: ${bitmap.width}x${bitmap.height}")

        // 只有在 bitmap 不可变时才复制
        val canvasBitmap = if (bitmap.isMutable) {
            bitmap
        } else {
            Log.d("MainActivity", "Bitmap 不可变，需要复制")
            bitmap.copy(Bitmap.Config.ARGB_8888, true)
        }
        
        val canvas = android.graphics.Canvas(canvasBitmap)

        val paint = android.graphics.Paint().apply {
            color = android.graphics.Color.GREEN
            strokeWidth = 8f
            style = android.graphics.Paint.Style.FILL
        }

        val linePaint = android.graphics.Paint().apply {
            color = android.graphics.Color.GREEN
            strokeWidth = 6f
            style = android.graphics.Paint.Style.STROKE
        }

        // 绘制关键点之间的连线
        val connections = listOf(
            Pair(PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
            Pair(PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
            Pair(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
            Pair(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
            Pair(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
            Pair(PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
            Pair(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
            Pair(PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
            Pair(PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
            Pair(PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
            Pair(PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
            Pair(PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE)
        )

        for ((startType, endType) in connections) {
            val start = pose.getPoseLandmark(startType)
            val end = pose.getPoseLandmark(endType)
            if (start != null && end != null) {
                canvas.drawLine(
                    start.position.x, start.position.y,
                    end.position.x, end.position.y,
                    linePaint
                )
            }
        }

        // 绘制所有关键点
        val landmarkTypes = listOf(
            PoseLandmark.NOSE,
            PoseLandmark.LEFT_EYE, PoseLandmark.RIGHT_EYE,
            PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR,
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE,
            PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE
        )

        for (landmarkType in landmarkTypes) {
            val landmark = pose.getPoseLandmark(landmarkType)
            if (landmark != null) {
                canvas.drawCircle(landmark.position.x, landmark.position.y, 10f, paint)
            }
        }

        return canvasBitmap
    }

    // 构建姿态信息文本
    private fun buildPoseInfoText(pose: Pose): String {
        val sb = StringBuilder()
        sb.appendLine("=== 骨骼关键点坐标 ===")

        val landmarkNames = mapOf(
            PoseLandmark.NOSE to "鼻子",
            PoseLandmark.LEFT_EYE to "左眼", PoseLandmark.RIGHT_EYE to "右眼",
            PoseLandmark.LEFT_EAR to "左耳", PoseLandmark.RIGHT_EAR to "右耳",
            PoseLandmark.LEFT_SHOULDER to "左肩", PoseLandmark.RIGHT_SHOULDER to "右肩",
            PoseLandmark.LEFT_ELBOW to "左肘", PoseLandmark.RIGHT_ELBOW to "右肘",
            PoseLandmark.LEFT_WRIST to "左手腕", PoseLandmark.RIGHT_WRIST to "右手腕",
            PoseLandmark.LEFT_HIP to "左髋", PoseLandmark.RIGHT_HIP to "右髋",
            PoseLandmark.LEFT_KNEE to "左膝", PoseLandmark.RIGHT_KNEE to "右膝",
            PoseLandmark.LEFT_ANKLE to "左脚踝", PoseLandmark.RIGHT_ANKLE to "右脚踝"
        )

        for ((landmarkType, name) in landmarkNames) {
            val landmark = pose.getPoseLandmark(landmarkType)
            if (landmark != null && landmark.inFrameLikelihood > 0.3f) {
                val x = String.format("%.3f", landmark.position.x)
                val y = String.format("%.3f", landmark.position.y)
                val conf = String.format("%.2f", landmark.inFrameLikelihood)
                sb.appendLine("$name: 坐标($x, $y), 置信度: $conf")
            }
        }

        return sb.toString()
    }

    // ========== 实时摄像头功能 ==========

    private fun startCameraCapture() {
        // 检查相机权限
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "需要相机权限", Toast.LENGTH_SHORT).show()
            cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
            return
        }

        if (isCameraCapturing) {
            Toast.makeText(this, "摄像头已启动", Toast.LENGTH_SHORT).show()
            return
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()

                // 显示摄像头预览
                binding.cameraPreviewSection.visibility = View.VISIBLE
                binding.textViewRecognitionPlaceholder.visibility = View.GONE

                // 解绑之前的使用
                cameraProvider?.unbindAll()

                // 创建预览用例
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(binding.previewView.surfaceProvider)
                    }

                // 创建图像分析用例
                val imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor ?: Executors.newSingleThreadExecutor(), CameraFrameAnalyzer())
                    }

                // 绑定用例
                cameraProvider?.bindToLifecycle(
                    this,
                    currentCameraSelector,
                    preview,
                    imageAnalyzer
                )

                isCameraCapturing = true
                Toast.makeText(this, "摄像头已启动", Toast.LENGTH_SHORT).show()

            } catch (e: Exception) {
                e.printStackTrace()
                Toast.makeText(this, "启动摄像头失败: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCameraCapture() {
        try {
            cameraProvider?.unbindAll()
            isCameraCapturing = false
            binding.cameraPreviewSection.visibility = View.GONE
            binding.textViewRecognitionPlaceholder.visibility = View.VISIBLE
            Toast.makeText(this, "摄像头已停止", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun switchCamera() {
        if (!isCameraCapturing) {
            Toast.makeText(this, "请先启动摄像头", Toast.LENGTH_SHORT).show()
            return
        }

        currentCameraSelector = if (currentCameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }

        // 重新绑定摄像头
        try {
            cameraProvider?.unbindAll()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor ?: Executors.newSingleThreadExecutor(), CameraFrameAnalyzer())
                }

            cameraProvider?.bindToLifecycle(
                this,
                currentCameraSelector,
                preview,
                imageAnalyzer
            )

            Toast.makeText(this, "已切换${if (currentCameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) "后置" else "前置"}摄像头", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "切换摄像头失败: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private inner class CameraFrameAnalyzer : ImageAnalysis.Analyzer {
        private var lastCaptureTime = 0L

        @androidx.camera.core.ExperimentalGetImage
        override fun analyze(imageProxy: ImageProxy) {
            // 检查 Activity 是否还在运行
            if (isDestroyed || isFinishing) {
                imageProxy.close()
                return
            }

            try {
                val currentTime = System.currentTimeMillis()

                // 每隔 cameraCaptureInterval 毫秒分析一次
                if (currentTime - lastCaptureTime >= cameraCaptureInterval) {
                    lastCaptureTime = currentTime

                    // 将 ImageProxy 转换为 Bitmap
                    val bitmap = imageProxyToBitmap(imageProxy)
                    if (bitmap != null && !isDestroyed && !isFinishing) {
                        // 执行姿态检测
                        if (poseDetector != null) {
                            val inputImage = InputImage.fromBitmap(bitmap, 0)
                            poseDetector?.process(inputImage)
                                ?.addOnSuccessListener { pose ->
                                    // 使用绿色骨骼标记绘制姿态 (in-place)
                                    val poseBitmap = drawPoseLandmarksInPlace(bitmap, pose)
                                    // 更新姿态覆盖层
                                    runOnUiThread {
                                        if (!isDestroyed && !isFinishing) {
                                            try {
                                                binding.cameraPoseOverlay.setImageBitmap(poseBitmap)
                                            } catch (e: Exception) {
                                                e.printStackTrace()
                                            }
                                        }
                                    }
                                }
                                ?.addOnFailureListener { e ->
                                    e.printStackTrace()
                                }
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                Log.e("CameraFrameAnalyzer", "分析失败: ${e.message}")
            } finally {
                imageProxy.close()
            }
        }

        /**
         * 优化版：直接 YUV 转 RGB/Bitmap，跳过 JPEG 编解码中间步骤
         * 原流程：YUV → NV21 → JPEG 压缩 → JPEG 解码 → Bitmap (耗时约 30-50ms)
         * 优化后：YUV → RGB int[] → Bitmap (耗时约 5-10ms)
         */
        private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
            try {
                // 检查平面数量
                if (imageProxy.planes.size < 3) {
                    Log.e("CameraFrameAnalyzer", "图像平面数量不足: ${imageProxy.planes.size}")
                    return null
                }

                val width = imageProxy.width
                val height = imageProxy.height
                val yBuffer = imageProxy.planes[0].buffer
                val uBuffer = imageProxy.planes[1].buffer
                val vBuffer = imageProxy.planes[2].buffer

                // 直接从 buffer 读取 YUV 数据
                val ySize = yBuffer.remaining()
                yBuffer.position(0)
                val yData = ByteArray(ySize)
                yBuffer.get(yData)

                // 获取 UV 平面的参数
                val uvPixelStride = imageProxy.planes[1].pixelStride
                val uvRowStride = imageProxy.planes[1].rowStride

                // 创建 RGB 像素数组
                val argb = IntArray(width * height)

                // 直接 YUV 转 ARGB（避免 NV21 中间格式和 JPEG 编解码）
                for (j in 0 until height) {
                    for (i in 0 until width) {
                        val yIndex = j * imageProxy.planes[0].rowStride + i
                        val y = yData[yIndex].toInt() and 0xFF

                        // UV 采样（每 2x2 像素共享一个 UV）
                        val uvIndex = (j / 2) * uvRowStride + (i / 2) * uvPixelStride
                        val u = (uBuffer.get(uvIndex).toInt() and 0xFF) - 128
                        val v = (vBuffer.get(uvIndex).toInt() and 0xFF) - 128

                        // YUV to RGB 转换（使用整数运算优化）
                        var r = y + ((1436 * v) shr 10)
                        var g = y - ((352 * u + 731 * v) shr 10)
                        var b = y + ((1814 * u) shr 10)

                        // 钳制到 0-255 范围
                        r = r.coerceIn(0, 255)
                        g = g.coerceIn(0, 255)
                        b = b.coerceIn(0, 255)

                        // 组合为 ARGB 像素（不透明）
                        argb[j * width + i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }

                // 直接创建 Bitmap
                var bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                bitmap.setPixels(argb, 0, width, 0, 0, width, height)

                // 根据旋转角度旋转图像
                val rotation = imageProxy.imageInfo.rotationDegrees
                if (rotation != 0 && bitmap != null) {
                    val matrix = android.graphics.Matrix()
                    matrix.postRotate(rotation.toFloat())
                    val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                    bitmap.recycle()
                    bitmap = rotatedBitmap
                }

                // 使用性能管理器推荐的图像最大尺寸进行缩放
                if (bitmap != null) {
                    val maxDimension = performanceManager.getRecommendedImageMaxDimension()
                    if (bitmap.width > maxDimension || bitmap.height > maxDimension) {
                        val scaledBitmap = imageCacheManager.scaleBitmap(bitmap, maxDimension)
                        bitmap.recycle()
                        bitmap = scaledBitmap
                    }
                }

                return bitmap
            } catch (e: Exception) {
                e.printStackTrace()
                Log.e("CameraFrameAnalyzer", "转换 Bitmap 失败: ${e.message}")
                return null
            }
        }
    }
}