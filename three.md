You are a senior Android + AI inference engineer.

Implement a production-grade on-device PhaseA pipeline on Android (Kotlin first) using the MNN models in this project.
Do not write pseudo code. Write complete runnable code.

================================================================
1) Project Facts You Must Follow (Aligned with current export)
================================================================

- Pipeline: Camera -> MediaPipe Pose(33x3) -> STGCN -> FNO -> Risk -> UI/History/Report
- Fully on-device.

- Model artifacts in assets/models:
  - stgcn_phaseA.mnn
  - stgcn_phaseA.mnn.weight
  - fno_lstm_phaseA.mnn
  - fno_lstm_phaseA.mnn.weight
  - risk_phaseA.mnn
  - risk_phaseA.mnn.weight

- Tensor names and shapes (must match exactly):
  - STGCN input: visual_seq, shape [1,5,33,3]
  - STGCN output: joint_angles, shape [1,23]
  - FNO input: bio_seq, shape [1,20,72]
  - FNO output: grf_seq, shape [1,10,12]
  - Risk input: risk_seq, shape [1,20,35]
  - Risk outputs: risk_logits [1,3], risk_confidence [1,1]

- Data type: float32 end-to-end.
- No extra quantization scale/zero_point handling.
- Current FNO deployment path is LSTM-compatible export; use it directly.

================================================================
2) Runtime Rules (MNN + threading)
================================================================

- MNN dependency: implementation "com.alibaba.android:mnn:1.2.0" (or newer stable).
- Use CPU forward type first, numThread=4.
- Create interpreter/session/tensor handles once in init(); reuse every inference.
- No per-frame large allocations inside infer(); pre-allocate and reuse buffers.
- Use AtomicBoolean for single-flight inference:
  - if previous inference is running, drop new request (no queue).
  - keep last valid result as fallback.
- Publish UI result via runOnUiThread.

================================================================
3) Feature Flow (must be implemented)
================================================================

3.1 Input frame buffer
- Maintain rolling buffer of 20 frames, each frame is 33x3 float landmarks.
- If frame count < 20, do not run full pipeline yet.

3.2 STGCN stage
- STGCN consumes 5-frame windows.
- Use four non-overlapping chunks from 20 frames:
  - 0..4, 5..9, 10..14, 15..19
- For each chunk run STGCN once, get one [23] joint vector.
- Expand to jaSeq[20,23] by repeating each chunk result to its 5 frames.

3.3 Build bio_seq for FNO
- For each t in 0..19:
  - ja: jaSeq[t] (23)
  - vel: jaSeq[t]-jaSeq[t-1], t=0 -> zeros(23)
  - acc: vel[t]-vel[t-1], t=0 -> zeros(23)
  - com: 3 dims from simplified kinematics
- Concatenate [ja, vel, acc, com] -> 72 dims.
- Final bio_seq shape [1,20,72].

3.4 FNO stage
- Run FNO, get grf_seq [1,10,12].
- Use first timestep grf0 = grf_seq[0][0] (12 dims).
- grf0 is normalized model output.

3.5 Build risk_seq
- For each t in 0..19:
  - normalize joint angles with joint_angles_mean/std from norm stats.
  - concatenate [ja_norm(23), grf0(12)] -> 35 dims.
- Repeat same 35-dim vector across 20 steps if needed.
- Final risk_seq shape [1,20,35].

3.6 Risk stage
- Run Risk model, get risk_logits [1,3], risk_confidence [1,1].
- Use numerically stable softmax on logits.
- riskScore = prob[2] (high risk).
- riskLabel = argmax(prob), mapping: 0 low, 1 medium, 2 high.

================================================================
4) Normalization and stats loading
================================================================

- Implement RGNormStats to load norm_stats.json from assets.
- JSON may contain "NaN" string; robust parse required.
- For invalid mean/std (NaN/Inf), fallback mean=0, std=1.
- Clamp std to at least 1e-6.

Use these fields if present:
- joint_angles_mean/std (required)
- grf_left_mean/std and grf_right_mean/std (for optional denorm display/report)

If optional fields are missing, keep pipeline running with safe defaults.

================================================================
5) Required Kotlin output files
================================================================

Package: com.healthai.ankle.inference

1. RGPhaseAEngine.kt
- fun init(context: Context)
- fun infer(frames20: Array<Array<FloatArray>>): InferenceResult?
- fun release()
- var userWeightKg: Float = 70f

InferenceResult fields:
- riskScore: Float
- riskLabel: Int
- confidence: Float
- jointAngles: FloatArray (23)
- grfLeft: FloatArray (6)
- grfRight: FloatArray (6)
- kneeAnglesDeg: Pair<Float, Float>

2. RGNormStats.kt
- load and sanitize stats json with safe getters.

3. RGFeatureBuilder.kt
- build jaSeq, bio_seq, risk_seq.

4. RGMarkerMapper.kt
- keep 33->28->84 mapping utility for future extension.

5. RehabGuardianActivity.kt integration
- CameraX + MediaPipe Pose callback
- 20-frame buffer maintenance
- call engine.infer
- show risk score/label in UI
- use runOnUiThread for updates

6. Room persistence (for replay + report)
- SessionEntity
- FrameEntity
- DAO + Database class

================================================================
6) Product features to implement
================================================================

1. Risk explanation text
- knee flexion < 150 deg, asymmetry > 20 deg, high impact threshold.
- join multiple reasons with ";".

2. Motion score (0-100)
- combine symmetry/flexion/impact weights exactly as specified.

3. Real-time hints
- show most severe reason when label is medium/high.
- throttle: at most one hint every 2 seconds.

4. Best-frame comparison in replay
- select best frame by lowest risk among frames with both knees >= 150 deg.
- fallback: global lowest-risk frame.

5. Enhanced PDF report
- AndroidPdfDocument, A4 size 595x842.
- include risk curve, knee curves, GRF vertical curves, score summary, suggestions.

================================================================
7) Error handling and quality bar
================================================================

- Model load failure: disable AI gracefully, no crash.
- NaN/Inf inputs: replace with 0.
- Inference exceptions: return last valid result.
- Performance target: < 50ms per end-to-end inference on typical Android CPU.
- Keep code modular, constants centralized, and comments only for non-obvious logic.

================================================================
8) Output format expected from you
================================================================

Provide full Kotlin code files with package/imports, not snippets.
Also provide required build.gradle dependencies:
- CameraX
- MediaPipe Tasks Vision
- MNN Android runtime
- Room

Do not skip any required class.
If any assumption is uncertain, explicitly list assumption and still provide runnable default behavior.

