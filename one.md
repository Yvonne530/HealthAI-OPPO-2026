You are a senior Android + AI inference engineer.

Your task is to implement a **production-grade on-device AI pipeline (PhaseA)** using **MNN models** on Android (Java).

This is NOT a demo.
Code must be robust, efficient, and directly runnable.

---

# 🧠 1. SYSTEM OVERVIEW

We are building a real-time rehab analysis system:

Camera → MediaPipe Pose (33x3) → AI Pipeline → Risk Score + Visualization

The system runs fully **on-device**, no cloud.

---

# 📦 2. MODEL ARTIFACTS

Three MNN models:

* stgcn.mnn (~2.0MB)
* fno_lstm.mnn (~3.6MB)
* risk.mnn (~177KB)

All models are located in Android assets root directory.

---

# 🔢 3. MODEL SPECS

## (1) STGCN

Input:
- name: visual_seq
- shape: [1, 5, 33, 3]
- layout: N,T,V,C
- dtype: float32

Outputs:
- index 0 → joint_angles → shape [1, 23] ✅ USE THIS
- index 1 → shape [1, 84] ❌ IGNORE

## (2) FNO (LSTM version)

Input:
- name: bio_seq
- shape: [1, 20, 72]

Output:
- grf_seq → shape [1, 10, 12]

Use ONLY first timestep: grf0 = grf_seq[0]

## (3) Risk Model

Input:
- risk_feat → [1, 20, 35]

Output:
- logits → [1, 3]

Apply softmax → risk_score = prob(high risk)

---

# 📐 4. PRECISE PIPELINE (Follow these facts)

## 4.1 Pose Normalization (Coordinate Conversion)

- **Shoulder scaling**: target shoulder width = 0.37m.
  Scale all keypoints by `target_width / current_shoulder_width`.
- **Hip centering**: Subtract the average of left hip (MP23) and right hip (MP24).
- **Y‑flip**: y = 1 - y (MediaPipe origin is top‑left).
- Hip and shoulder indices: MP23 (left hip), MP24 (right hip), MP11 (left shoulder), MP12 (right shoulder).

## 4.2 Marker Mapping (33 → 28 → 84)

Use the following mapping (MediaPipe index → marker index; `midpoint(A,B)` means average of those two keypoints):

| Marker Id | MediaPipe source(s)                              | Description          |
|-----------|--------------------------------------------------|----------------------|
| 0         | 23                                               | L_ASIS               |
| 1         | 27                                               | L_Ankle_Lat          |
| 2         | 29                                               | L_Heel               |
| 3         | 25                                               | L_Knee_Lat           |
| 4         | 23                                               | L_PSIS               |
| 5         | midpoint(25,27)                                  | L_Shank_Front        |
| 6         | midpoint(25,27)                                  | L_Shank_Rear         |
| 7         | 25                                               | L_Shank_Upper        |
| 8         | 23                                               | L_Thigh_Front        |
| 9         | 23                                               | L_Thigh_Rear         |
| 10        | 23                                               | L_Thigh_Upper        |
| 11        | 31                                               | L_Toe_Lat            |
| 12        | 31                                               | L_Toe_Med            |
| 13        | 31                                               | L_Toe_Tip            |
| 14        | 24                                               | R_ASIS               |
| 15        | 28                                               | R_Ankle_Lat          |
| 16        | 30                                               | R_Heel               |
| 17        | 26                                               | R_Knee_Lat           |
| 18        | 24                                               | R_PSIS               |
| 19        | midpoint(26,28)                                  | R_Shank_Front        |
| 20        | midpoint(26,28)                                  | R_Shank_Rear         |
| 21        | 26                                               | R_Shank_Upper        |
| 22        | 24                                               | R_Thigh_Front        |
| 23        | 24                                               | R_Thigh_Rear         |
| 24        | 24                                               | R_Thigh_Upper        |
| 25        | 32                                               | R_Toe_Lat            |
| 26        | 32                                               | R_Toe_Med            |
| 27        | 32                                               | R_Toe_Tip            |

After obtaining 28 markers (each with x,y,z), flatten to a 84‑dimensional vector **markers_flat**.

## 4.3 Z‑score Normalization of markers_flat

**Critical**: `norm_stats.json` contains valid mean/std only for **9 specific indices** in the 84‑dim array:


[9, 10, 11, 48, 49, 50, 51, 52, 53]

All other indices have **NaN** in the statistics file.

Implementation rules:
- For the 9 valid indices: compute `(value - mean) / std` using the corresponding mean/std.
- For any other index: keep the original value (no normalization).
- If a mean or std is NaN or infinite → replace mean with 0, std with 1.
- If std < 1e‑6 → clamp to 1e‑6.

## 4.4 STGCN Inference & jaSeq Construction

- Maintain a sliding window of **20 consecutive pose frames** (each frame already normalized).
- Run STGCN on **4 overlapping segments**:
  - segment 0: frames 0–4
  - segment 1: frames 5–9
  - segment 2: frames 10–14
  - segment 3: frames 15–19
- Each inference yields a 23‑dim joint angle vector.
- Construct `jaSeq` (20×23):
  - frames 0–4 ← result of segment 0
  - frames 5–9 ← result of segment 1
  - frames 10–14 ← result of segment 2
  - frames 15–19 ← result of segment 3

**Joint angle indices** (for later use):
- left knee angle = jaSeq[t][6]
- right knee angle = jaSeq[t][13]

## 4.5 Bio‑Feature Building (20×72)

For each frame t in 0..19, construct a 72‑dim vector:

- `ja` = jaSeq[t]  (23)
- `vel` = jaSeq[t] - jaSeq[t-1] (for t=0, vel = 0) (23)
- `acc` = vel[t] - vel[t-1] (for t=0, acc = 0) (23)
- `com` = simplified kinematics (see formula below) (3)

Concatenate: `[ja, vel, acc, com]` → 72 dim.

**COM formula** (constants derived from anthropometry):
lTh = 0.42f // thigh length
lSh = 0.45f // shank length

pelvis_tilt = joint_angles[0] // from jaSeq[t]
pelvis_list = joint_angles[1] // from jaSeq[t]
hip_avg = (jaSeq[t][6] + jaSeq[t][13]) / 2 // average of left & right knee? Wait, hip_avg is not knee; correct: hip_avg = (left_hip_angle + right_hip_angle)/2; need to know indices for hip. In OpenSim gait2392, hip angles are indices 3 (left hip flexion) and 10 (right hip flexion). We'll let you infer from context. Better: We'll provide the exact indices later. For now, assume hip_flexion indices are known from joint_angles.

text

Actually, we should give exact indices to avoid confusion. From the training code:
- Pelvis tilt = joint_angles[0]
- Pelvis list = joint_angles[1]
- Left hip flexion = joint_angles[3]
- Right hip flexion = joint_angles[10]
- Left knee = joint_angles[6]
- Right knee = joint_angles[13]

Then:
hip_avg = (joint_angles[3] + joint_angles[10]) / 2
knee_avg = (joint_angles[6] + joint_angles[13]) / 2
x = lTh * sin(pelvis_tilt) * 0.5
y = lTh * (1 - cos(hip_avg)) + lSh * (1 - cos(knee_avg))
z = lTh * sin(pelvis_list) * 0.5

text

## 4.6 FNO Inference & GRF Denormalization

- Input bio_seq: shape [1,20,72]
- Output grf_seq: shape [1,10,12]
- Take the first timestep `grf0 = grf_seq[0][0]` → shape [12]
- Denormalize using `norm_stats.json`:
  - `grf_left = grf0[0:6] * grf_left_std + grf_left_mean`
  - `grf_right = grf0[6:12] * grf_right_std + grf_right_mean`
- Result: `grf_denorm` (12), where first 6 are left foot, last 6 right foot.

## 4.7 Risk Input Construction (20×35)

For each frame t, create a 35‑dim vector:
- `ja_norm` = joint_angles (23) **already normalized?** No, jaSeq[t] is raw angles (radians). The risk model expects them normalized. Use `joint_angles_mean/std` from norm_stats.json to normalize.
- `grf_norm` = the 12‑dim GRF vector **before denormalization** (i.e., the raw output from FNO) **or after?** According to training, risk model expects normalized GRF (the same as FNO input). Since FNO output is already normalized, we can reuse `grf0` directly (the 12‑dim before denormalization). Thus:
  - risk_feat[t] = [normalized_ja(23) , grf0(12)]
- Then replicate this same 35‑dim vector across all 20 time steps (since risk model expects a sequence of 20 frames, but in PhaseA we feed the same vector repeated).

## 4.8 Risk Inference

- Input risk_feat: shape [1,20,35]
- Output logits: shape [1,3]
- Apply softmax: probabilities for low/medium/high.
- `risk_score = prob_high` (index 2)
- `risk_label = argmax(prob)` (0=low,1=medium,2=high)

---

# ⚙️ 5. MNN RUNTIME CONSTRAINTS (CRITICAL)

You MUST follow:

- Use `getSessionInput` / `getSessionOutput` with **exact tensor names**:
  - STGCN: input `visual_seq`, output `joint_angles`
  - FNO: input `bio_seq`, output `grf_seq`
  - Risk: input `risk_feat`, output `logits`
- Do **NOT** rely on output index; use the name.
- Create sessions **ONCE** and reuse.
- `numThread = 4`
- `forwardType = CPU`
- Input shape fixed (batch=1, seq_len fixed, no dynamic reshape per frame).

---

# 🧮 6. NORMALIZATION DATA

`norm_stats.json` is provided with the following keys:

- `joint_angles_mean`, `joint_angles_std` (23)
- `joint_vel_mean`, `joint_vel_std` (23) – not used in pipeline, but available.
- `joint_acc_mean`, `joint_acc_std` (23)
- `markers_flat_mean`, `markers_flat_std` (84) – with NaN in most entries; handle as described.
- `grf_left_mean`, `grf_left_std` (6)
- `grf_right_mean`, `grf_right_std` (6)
- `com_mean`, `com_std` (3) – not used in PhaseA.

Implement a helper class `RGNormStats` that loads the JSON and provides safe access with fallbacks.

---

# 🧵 7. THREADING MODEL

You are free to choose the threading implementation, but it must achieve:

- Camera thread → produce frames (MediaPipe callbacks)
- Main thread → maintain sliding window (20‑frame buffer)
- Inference thread → run the pipeline (non‑blocking to camera)
- UI thread → receive results and update display

A typical robust pattern:
- Use `HandlerThread` for inference and `Handler` to post tasks.
- Use `ArrayDeque` for window buffer, synchronized where needed.
- Use `LiveData` or `Flow` to pass results to UI.

---

# ⏱ 8. STREAMING LOGIC

- Maintain a circular buffer of **20 normalized pose frames**.
- On each new frame:
  - Add to buffer (pop oldest if full).
  - If buffer has 20 frames, **trigger inference** (on background thread).
- Inference should run asynchronously; if a new inference is triggered while previous is running, either queue or drop based on latency requirements (drop is acceptable for real-time).
- Use a flag to ensure only one inference runs at a time.

---

# ⚠️ 9. EDGE CASE HANDLING

You MUST implement:

- **Model load failure**: disable AI gracefully (return default values, no crash).
- **NaN/Inf in input**: replace with 0.
- **Invalid std (NaN, 0, <1e-6)**: clamp to 1.
- **Missing markers mapping**: if required MediaPipe keypoint missing (visibility low), use last known valid frame for that point. If a core point (hip, shoulder) missing, skip adding this frame to buffer.
- **Inference error**: keep last valid result or return safe default.

---

# 🚀 10. REQUIRED OUTPUT CODE

Generate the following Java/Kotlin classes (Kotlin preferred, but Java acceptable). All classes must be in package `com.healthai.ankle.inference`.

### (1) Core Engine
`RGPhaseAEngine.java` (or .kt) with:

- `init(Context)`
- `infer(float[][][])` where input is 20×33×3
- `release()`

### (2) Helper Classes
- `RGNormStats.java`: load and provide safe stats access.
- `RGMarkerMapper.java`: 33→28 mapping, return 84‑dim markers_flat.
- `RGFeatureBuilder.java`: build 20×72 bio features and 20×35 risk features from jaSeq and grf0.

### (3) Activity Integration
`RehabGuardianActivity.java` (or .kt) that:
- Integrates CameraX
- Integrates MediaPipe Pose
- Uses the inference engine
- Shows risk score on UI

You can design the UI simply (a TextView for risk) – the focus is on correct engine implementation.

---

# 🎯 11. PERFORMANCE TARGET

- Inference time < 50ms per batch (20 frames processed in one go).
- No per-frame allocations inside the inference loop.
- Tensor buffers reused.

---

# 🔥 12. CODE QUALITY REQUIREMENTS

- Clear class structure.
- No hardcoded magic numbers (use constants).
- Reusable tensor buffers.
- Detailed comments explaining logic, especially where the pipeline deviates from standard approaches.

---

# ❗ FINAL INSTRUCTION

Do **NOT** simplify.

Do **NOT** skip parts.

Produce FULL working code with correct shapes, threading, and MNN usage.

The result should be directly usable in an Android project.

Use Kotlin unless Java is explicitly required for compatibility with existing code.
