基于你从代码中确认的 7 点关键事实，我在原最终版提示词中进行了精确补充，形成一份**完全可落地、无歧义**的版本。新增内容已用 `🔧` 标记，你可以直接替换原有提示词。

---

# 最终版提示词（已整合代码确认细节，可直接给 Claude）

```text
You are a senior Android + AI inference engineer.

Your task is to implement a **production-grade on-device AI pipeline (PhaseA)** using **MNN models** on Android (Kotlin preferred).  
This is NOT a demo. Code must be robust, efficient, and directly runnable.

---

# 🧠 1. SYSTEM OVERVIEW

Camera → MediaPipe Pose (33x3) → AI Pipeline → Risk Score + Visualization

Fully on‑device.

---

# 📦 2. MODEL ARTIFACTS

Three MNN models are in Android assets root:

- `stgcn.mnn`
- `fno_lstm.mnn`
- `risk.mnn`

All file names as given; no subfolder.

---

# 🔢 3. MODEL SPECS (EXACT, VERIFIED FROM TRAINING CODE)

## STGCN
- Input name: `visual_seq`
- Shape: `[1, 5, 33, 3]` (batch, time, vertices, channels)
- Memory layout: **row‑major NHWC** – fill in order: `for t in 0..4: for v in 0..32: for c in 0..2`
- Output: `joint_angles` → `[1, 23]` (use this; ignore secondary output)
- Input is **normalized pose** (33×3) – **NOT** markers.  
  The markers branch is **never used in PhaseA**.

## FNO (LSTM version)
- Input name: `bio_seq`
- Shape: `[1, 20, 72]`
- Output: `grf_seq` → `[1, 10, 12]`
- Use only the first timestep: `grf0 = grf_seq[0][0]` → `[12]`
- **Output is normalized** – must be denormalized before use.

## Risk Model
- Input name: `risk_feat`
- Shape: `[1, 20, 35]`
- Output: `logits` → `[1, 3]`
- Apply numerically stable softmax → `risk_score = prob(high risk)` (index 2)

🔧 **Data Type & Quantization**  
All models are used with **float32** tensors.  
- Input: `setInputFloatData(floatArray)`  
- Output: `getFloatData()`  
No additional quantization parameters (scale/zero_point) are required.

---

# ⚙️ 4. MNN RUNTIME (CRITICAL)

- **DimensionType = TENSORFLOW** (NHWC layout).  
  The models were converted with MNNConvert using TensorFlow layout; input tensors expect NHWC exactly as specified.
- **Session reuse**: create once in `init()`, reuse for every inference.
- **Tensor reuse**: get input/output tensors at init, store them as class fields.  
  Use `inputTensor.setInputFloatData(floatArray)` to feed data; do not re‑allocate.
- `numThread = 4`, `forwardType = CPU`
- Input shapes are fixed (batch=1, seq_len fixed). No dynamic reshape.

**API usage pattern** (from actual project code):
```kotlin
val inputTensor = session.getInput("visual_seq")
inputTensor.setInputFloatData(floatArray)   // floatArray is a Kotlin FloatArray
session.run()
val outputTensor = session.getOutput("joint_angles")
val outputData = outputTensor.floatData     // returns FloatArray
```

**Real‑time policy**: If inference is already running → **drop the NEW incoming request** (do not queue; reuse last result).  
🔧 Use an `AtomicBoolean` flag to guard inference.

---

# 📐 5. PRECISE PIPELINE (MUST MATCH TRAINING CODE)

## 5.1 MediaPipe Keypoint Indices & Extraction

| Index | Part          |
|-------|---------------|
| 11    | LEFT_SHOULDER |
| 12    | RIGHT_SHOULDER|
| 23    | LEFT_HIP      |
| 24    | RIGHT_HIP     |
| 25    | LEFT_KNEE     |
| 26    | RIGHT_KNEE    |
| 27    | LEFT_ANKLE    |
| 28    | RIGHT_ANKLE   |
| 29    | LEFT_HEEL     |
| 30    | RIGHT_HEEL    |
| 31    | LEFT_FOOT_INDEX |
| 32    | RIGHT_FOOT_INDEX |

🔧 **Extraction rules** (from existing Android implementation):
- Use `getPosition3D().getX()`, `.getY()`, `.getZ()` to get world coordinates.
- Normalize by dividing by image width for x and z (z = z / width).
- **Do NOT use `visibility`** – it is ignored.
- If any of the four core points (left/right hip, left/right shoulder) is missing, **discard the whole frame**.
- For any other missing point, fill with `0.0f`.

## 5.2 Pose Normalization (Order Matters – Verified from `RGCoordinate.java`)

For each frame (33 keypoints in MediaPipe order):

1. **Subtract hip center**:  
   `midHip = (left_hip + right_hip) / 2` (indices 23 & 24)  
   Subtract from all keypoints.

2. **Scale to meters**:  
   Compute shoulder width between indices 11 & 12.  
   `scale = 0.37f / shoulderDist` (target shoulder width = 0.37 m)  
   Multiply **all three coordinates** (x, y, z) by `scale`.

3. **Y‑flip**:  
   `y = -y` (because MediaPipe origin is top‑left, we want bottom‑left).

🔧 **Important**: x and z are **not flipped**. The z coordinate uses the same scaling as x,y.

Result: `xyz33_metric` (33×3) in meters, hip‑centered.

## 5.3 Marker Mapping (33 → 28 → 84) – Keep for Future Extensions

Use the mapping table provided earlier (28 markers with `midpoint`).  
After mapping, flatten to `markers_flat` (84 floats).  
**This data is NOT used by any model in PhaseA**, but implement it for completeness.

## 5.4 Z‑score Normalization of `markers_flat`

`norm_stats.json` contains valid mean/std only for **9 indices**:

```
[9, 10, 11, 48, 49, 50, 51, 52, 53]
```

All other indices have `NaN` in the statistics file.

🔧 **JSON format** (from Android assets): NaN values are stored as **string "NaN"**, not JSON null.  
Use `org.json.JSONObject` with `optDouble` and check for `Double.isNaN()` to handle them.

Implementation rules:
- For the 9 valid indices: compute `(value - mean) / std`
- For any other index: keep the original value (no normalization)
- If `mean` or `std` is NaN or infinite → set `mean = 0`, `std = 1`
- If `std < 1e-6` → clamp to `1e-6`

## 5.5 STGCN Inference & jaSeq Construction

Maintain a **20‑frame circular buffer** of normalized pose (33×3) – after all previous steps.

- Run STGCN on **4 fixed non-overlapping segments**:  
  frames 0‑4, 5‑9, 10‑14, 15‑19.
- Each run yields `[1,23]` joint angles (radians).
- Build `jaSeq` (20×23):
  - frames 0‑4  ← result of segment 0
  - frames 5‑9  ← result of segment 1
  - frames 10‑14 ← result of segment 2
  - frames 15‑19 ← result of segment 3

**Joint angle indices** (0‑based, radians):
- left knee = index 6
- right knee = index 13
- left hip = index 3
- right hip = index 10
- pelvis tilt = index 0
- pelvis list = index 1

## 5.6 Bio‑Feature Building (20×72)

For each frame t:

- `ja` = jaSeq[t] (23)
- `vel` = jaSeq[t] - jaSeq[t-1]; for t=0, vel = 0 (23)
- `acc` = vel[t] - vel[t-1]; for t=0, acc = 0 (23)
- `com` = simplified kinematics (3)

**COM formula** (radians, from `kinematics.py`):

```
lTh = 0.42f   // thigh length
lSh = 0.45f   // shank length

pelvis_tilt = jaSeq[t][0]
pelvis_list = jaSeq[t][1]
hip_avg = (jaSeq[t][3] + jaSeq[t][10]) / 2
knee_avg = (jaSeq[t][6] + jaSeq[t][13]) / 2

x = lTh * sin(pelvis_tilt) * 0.5
y = lTh * (1 - cos(hip_avg)) + lSh * (1 - cos(knee_avg))
z = lTh * sin(pelvis_list) * 0.5
```

Concatenate: `[ja, vel, acc, com]` → 72 floats.

## 5.7 FNO Inference & GRF Denormalization

- Input `bio_seq` = `[1,20,72]` (from above).
- Output `grf_seq` = `[1,10,12]`.
- Take first timestep `grf0 = grf_seq[0][0]` → `[12]`.
- GRF layout per foot: `[Fx, Fy, Fz, Mx, My, Mz]` (0‑based indices)
  - Left vertical = index 2
  - Right vertical = index 8 (= 6+2)

Denormalize using `norm_stats.json`:

```
grf_left_physical  = grf0[0:6] * grf_left_std + grf_left_mean
grf_right_physical = grf0[6:12] * grf_right_std + grf_right_mean
```

Result: `grf_denorm` (12) in Newtons.

## 5.8 Risk Input Construction (20×35)

- Normalize joint angles using `joint_angles_mean/std` from `norm_stats.json`:
  ```
  ja_norm = (jaSeq[t] - joint_angles_mean) / joint_angles_std
  ```
- **GRF must be normalized** (use `grf0`, **not** the physical denormalized version).
- For each frame t, `risk_feat[t] = [ja_norm(23), grf0(12)]` → 35 floats.
- Repeat the **same 35‑dim vector** for all 20 frames (risk model expects a sequence).

## 5.9 Risk Inference

- Input `risk_feat` shape `[1,20,35]`.
- Output `logits` shape `[1,3]`.
- Apply numerically stable softmax:
  ```
  max_logits = max(logits)
  exp_logits = exp(logits - max_logits)
  probs = exp_logits / sum(exp_logits)
  ```
- `risk_score = probs[2]` (high risk probability)
- `risk_label = argmax(probs)` (0=low,1=medium,2=high)

---

# 🧮 6. NORMALIZATION DATA (norm_stats.json)

- All values are stored as **float arrays**.
- NaN values are represented as **string "NaN"** in the JSON file (not null).  
  Use `org.json.JSONObject` with `optDouble` and `Double.isNaN()` to detect them.
- Keys:
  - `joint_angles_mean`, `joint_angles_std` (23)
  - `joint_vel_mean`, `joint_vel_std` (23) – not used in pipeline
  - `joint_acc_mean`, `joint_acc_std` (23) – not used
  - `markers_flat_mean`, `markers_flat_std` (84) – with NaNs as described
  - `grf_left_mean`, `grf_left_std` (6)
  - `grf_right_mean`, `grf_right_std` (6)
  - `com_mean`, `com_std` (3) – not used in PhaseA

Implement `RGNormStats` that loads this JSON and provides safe access with NaN/Inf fallbacks as described.

---

# 🧵 7. THREADING & BUFFER MANAGEMENT

- Camera thread (UI) produces frames via MediaPipe callback.
- Maintain a **synchronized circular buffer** of 20 frames (pose after all normalization).  
  Use `ArrayDeque` protected by `synchronized` or `ReentrantLock`.
- On each new frame:
  - Add to buffer.
  - If buffer size == 20, **take a snapshot** (copy to new array) and post to inference thread.
- Inference thread runs on `HandlerThread` (or coroutine `Dispatchers.Default`).
- Use an `AtomicBoolean` flag to ensure only one inference runs at a time.  
  **If a new snapshot arrives while flag is true → drop it** (do not queue).
- Inference result is posted back to UI thread (via `LiveData` or `runOnUiThread`).  
  🔧 The existing reference implementation uses `runOnUiThread`.

---

# ⚠️ 8. EDGE CASE HANDLING

- Model load failure → disable AI, log error, no crash.
- NaN/Inf in input → replace with 0.
- Invalid std (NaN, 0, <1e-6) → clamp to 1.
- If a required MediaPipe keypoint is missing (visibility < 0.5), use the last valid frame for that point.  
  If a core point (hip, shoulder) is missing, skip adding this frame to the buffer.
- Inference error → keep last valid result.

---

# 🚀 9. REQUIRED OUTPUT CODE

Generate the following Kotlin classes in package `com.healthai.ankle.inference`.

## (1) Core Engine
`RGPhaseAEngine.kt` with:
- `fun init(context: Context)`
- `fun infer(frames: Array<Array<FloatArray>>): InferenceResult?` (input: 20×33×3)
- `fun release()`
- A `var userWeightKg: Float = 70f` for GRF thresholds.

`data class InferenceResult`:
- `riskScore: Float`
- `riskLabel: Int`
- `jointAngles: FloatArray` (23)
- `grfLeft: FloatArray` (6)
- `grfRight: FloatArray` (6)
- `kneeAngles: Pair<Float,Float>` (left, right in **degrees**)

## (2) Helper Classes
- `RGNormStats.kt`: load `norm_stats.json` from assets, provide safe getters.
- `RGMarkerMapper.kt`: map 33‑point pose to 28 markers → 84 flat array.
- `RGFeatureBuilder.kt`: from `jaSeq` and `grf0` build 20×72 bio features and 20×35 risk features.

## (3) Activity Integration
`RehabGuardianActivity.kt` (or .kt) that:
- Integrates CameraX (standard).
- Integrates MediaPipe Pose (using `pose_landmarker_lite.task` from assets).
- Uses the inference engine.
- Displays risk score (and optionally the extra features below) on UI.

🔧 **UI updates**: use `runOnUiThread` to show results (or LiveData if preferred).  
🔧 **History storage**: for replay and PDF reports, use Room database. The schema should include a `SessionEntity` and `FrameEntity` as described below.

---

# 🧩 10. ADDITIONAL PRODUCT FEATURES (Using Only Existing Data)

Implement these **without extra models**.

## 1. Risk Explanation (text)
Based on current frame (in degrees):
- left knee < 150° → "Left knee overflexion"
- right knee < 150° → "Right knee overflexion"
- total vertical GRF (left_fz+right_fz) > 2.5×userWeight (in Newtons) → "High impact"
- |left_knee - right_knee| > 20° → "Asymmetrical gait"

Combine with " ; " if multiple.

## 2. Movement Score (0‑100)
- **Symmetry** (40%): `100 - min(100, abs(left_knee - right_knee) × 2)`
- **Flexibility** (30%): `100 - max(0, (150 - min(left_knee, right_knee)) × 2)`
- **Impact** (30%):  
  `grf_bw = (left_fz + right_fz) / (userWeightKg * 9.81f)`  
  `100 - min(100, (grf_bw - 2) × 50)`

Total = Symmetry×0.4 + Flexibility×0.3 + Impact×0.3.

## 3. Real‑time Hint
When riskLabel is medium/high, show the most severe reason from risk explanation.  
Limit to **at most one hint every 2 seconds** (using a timer).

## 4. Replay with Best‑Frame Comparison
During recording, store all frames in a list of custom data class (e.g., `FrameRecord` containing timestamp, keypoints, riskScore, knee angles, GRF).  
In replay activity:
- Compute the **best frame** (lowest riskScore among frames with both knee angles ≥ 150°).
- Add a button "Compare with best". When clicked, split screen: left = current frame, right = best frame. Skeleton drawn by same renderer.

🔧 **If no frame satisfies both criteria, fallback to the frame with the lowest riskScore.**

## 5. Enhanced PDF Report
Use Android `PdfDocument` to generate a report with:
- Session metadata (date, duration)
- Risk curve (frame index vs riskScore)
- Knee angle curves (left & right, degrees)
- GRF vertical curves (left & right, Newtons)
- Total movement score + sub‑scores
- Risk explanation summary
- Personalized advice (e.g., based on most frequent risk factor)

🔧 **Page size**: A4 (595x842 points). Curves should auto‑scale to data min/max.

## 6. UI Additions
- Show risk explanation TextView.
- Show movement score (circular progress or big number).
- Show real‑time hint TextView (auto‑clears).
- In replay activity: "Compare with best" button.
- In history screen: "Generate report" button.

All UI must be overlayed without blocking camera preview (use FrameLayout with translucent background).

---

# 🎯 11. PERFORMANCE & CODE QUALITY

- Inference time < 50 ms (measured on typical Android device).
- No per‑frame allocations inside `infer()` – reuse buffers.
- Clear class structure, constants named, comments explaining non‑obvious parts.
- Provide `build.gradle` dependencies: CameraX, MediaPipe Tasks Vision, MNN (`implementation 'com.alibaba.android:mnn:0.0.9'`), Room, etc.

---

# ❗ FINAL INSTRUCTION

All model behaviors, normalization logic, and coordinate conventions strictly follow the training pipeline described above.  
If any ambiguity arises, prioritize consistency with the provided equations and data shapes over assumptions.

Do **NOT** simplify.  
Do **NOT** skip parts.  
Produce FULL working Kotlin code with correct shapes, threading, and MNN usage.

The result should be directly usable in an Android project.
```你看下这一版本的提示词够清楚了吗
