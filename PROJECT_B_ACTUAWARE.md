# Project B · ActuAware (动悟) — Offline Multimodal AI Assistant for Android

> A llama.cpp-powered, fully offline multimodal AI assistant running entirely on-device.
> ← [Back to repository home](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/main/README.md)

> **Source code: [`Android-Debug` branch](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/Android-Debug)**
>
> This project was developed collaboratively; its complete original development
> history (8 code commits, 2026-03) is preserved unchanged on that branch.
> See the project documentation and Git history for implementation details.

## 1. Overview

ActuAware is an Android AI assistant that runs without any cloud service:
a local GGUF large language model drives streaming chat, image analysis,
skeleton-based pose analysis, web-search-augmented answers (RAG), and
conversation history management — all on a single phone.

## 2. Features

- **Local LLM chat** — llama.cpp inference with GGUF models; replies stream
  token-by-token with per-message metadata (token count, generation time)
- **Image analysis** — ML Kit pose detection overlays green skeletons/keypoints
  and feeds joint coordinates to the multimodal model for assisted analysis
- **Real-time camera capture** — CameraX preview with live skeleton detection,
  front/back camera switching
- **Web-search RAG** — Bing search results are fetched and injected into the
  local model's context to supplement time-sensitive answers
- **Conversation history** — save/load/delete sessions as JSON (history,
  model path, parameters, system prompt)
- **Markdown rendering** — tables, task lists, one-tap copy of AI messages
- **Device-adaptive performance** — automatic device-tier detection adjusts
  threads, batch size, context size and camera frame rate

## 3. Architecture

```
Kotlin UI (ViewBinding + Material Design)
   ├─ ChatAdapter        RecyclerView chat bubbles · streaming output · Markwon
   ├─ ToolManager /      file tools (create/read/edit/delete) with path,
   │  ToolChatManager    type & size validation
   ├─ NetworkHelper      Bing search → page fetch → RAG context injection
   ├─ HistoryHelper      JSON session persistence (files/history)
   ├─ PerformanceManager device tiering → adaptive threads/batch/context/fps
   └─ JNI (llama_jni.cpp) ──► llama.cpp (+ mtmd, Flash Attention, OpenCL)
            nativeLoadModel / nativeChat / nativeAnalyzeImage /
            streaming callback → incremental UI rendering
CameraX + ML Kit Pose Detection ──► skeleton coordinates → JNI → AI analysis
```

## 4. llama.cpp / JNI Integration

`llama_jni.cpp` exposes the native layer to Kotlin:

- `nativeLoadModel(modelPath, mmprojPath, systemPrompt, maxTokens, contextSize, temperature, topP, topK)`
- `nativeChat(userInput, resetHistory)` / `nativeResetChatHistory()` / `nativeRestoreContext(historyJson)`
- `nativeAnalyzeImage(imageData, width, height, userInput)` for multimodal input
- `nativeSetPerformanceParams(threads, batchSize, gpuLayers, enableFlashAttention)`
- Streaming callbacks for incremental token rendering

Notable engineering fix recorded in commit `4cf0097`: the original decode loop
re-decoded the whole history into the KV cache each turn (O(n²), causing token
overlap and repeated generation). It was rewritten to incremental decoding —
tracking `prev_token_count`, decoding only new tokens, and clearing the KV
cache on truncation/reset.

## 5. GPU Acceleration Exploration

A real debugging journey preserved in commit history:

| Commit | Outcome |
|---|---|
| `ca489ae` | Knowledge-base feature added; Vulkan acceleration attempt hit issues |
| `f1406b5` | Vulkan unsupported on Snapdragon → switched to OpenCL (~20% faster than pure CPU, informal comparison); OpenCL headers vendored in |
| `afd9f0e` | Vulkan code removed; OpenCL kept; QNN noted as possible next step |

Note: the "~20%" figure comes from the developer's informal on-device
comparison recorded in the commit message; no formal benchmark methodology
was captured, so treat it as indicative rather than a measured result.

## 6. Multimodal Pipeline

User photo → background-thread processing (ANR-safe) → full-resolution image
for skeleton detection → drawn keypoints/skeleton → joint coordinates sent via
JNI to the multimodal model → Markdown-formatted analysis rendered in chat.
Resolution is adapted per device tier (e.g. high-tier 1280→512 px, mid 1024→448 px,
low 800→384 px).

## 7. Android Structure

```
app/src/main/
├── java/com/erlab/actuaware/     # MainActivity · ChatAdapter · HistoryHelper ·
│                                 # ToolManager · PerformanceManager · TtsManager …
├── cpp/
│   ├── llama_jni.cpp             # JNI bridge
│   ├── llama.cpp/                # upstream llama.cpp source
│   └── opencl-headers/           # OpenCL headers
└── res/                          # Material Design UI (light/dark themes)
```

Build: minSdk 24 · compileSdk 34 · NDK 26.1.10909125 · CMake 3.22.1 · arm64-v8a.
Detailed interface documentation:
[`PROJECT_GUIDE.md`](https://github.com/Yvonne530/HealthAI-OPPO-2026/blob/Android-Debug/PROJECT_GUIDE.md).

## 8. Git History

Full history lives on the [`Android-Debug`](https://github.com/Yvonne530/HealthAI-OPPO-2026/tree/Android-Debug) branch:

```
d90418a 03-04  Initial APK source upload with detailed code documentation
ab3ccd0 03-06  ML Kit pose detection integration into the recognition view
ab94cb0 03-13  Major refactor: PerformanceManager, ImageCacheManager, UI overhaul
ca489ae 03-14  Knowledge-base feature + Vulkan acceleration attempt
f1406b5 03-15  Vulkan unsupported on Snapdragon → OpenCL switch
afd9f0e 03-15  Vulkan removal cleanup
ac23667 03-16  History deduplication, anti-prompt stop conditions, format fixes
4cf0097 03-18  Resolution tiering + O(n²)→O(1) incremental token decoding + TTS
```

## 9. Running

```bash
git checkout Android-Debug
./gradlew assembleDebug
# Load a local GGUF model (and mmproj file for multimodal features) in-app
```

## 10. Demo Screenshots

Screenshots / demo video: coming soon.
