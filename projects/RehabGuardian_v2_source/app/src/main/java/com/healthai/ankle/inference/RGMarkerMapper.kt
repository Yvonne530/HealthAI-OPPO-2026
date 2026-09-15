package com.healthai.ankle.inference

/**
 * MediaPipe Pose landmark index mappings.
 * Provides 33 → 28 anatomical subset and future 84-channel extension.
 *
 * MediaPipe 33-point schema:
 *   0  nose           11 L shoulder   22 R heel
 *   1  L eye inner    12 R shoulder   23 L foot index
 *   2  L eye          13 L elbow      24 R foot index
 *   3  L eye outer    14 R elbow      25–32 hand points
 *   4  R eye inner    15 L wrist
 *   5  R eye          16 R wrist
 *   6  R eye outer    17 L pinky
 *   7  L ear          18 R pinky
 *   8  R ear          19 L index
 *   9  mouth L        20 R index
 *  10  mouth R        21 L thumb
 *  11 L shoulder      ...
 *  13 L elbow
 *  14 R elbow
 *  15 L wrist
 *  16 R wrist
 *  23 L hip           27 L knee      31 L ankle
 *  24 R hip           28 R knee      32 R ankle
 */
object RGMarkerMapper {

    /**
     * 28 anatomically-meaningful indices from MediaPipe 33-point output.
     * Drops face/hand landmarks, keeps body + lower extremity.
     */
    val IDX_28: IntArray = intArrayOf(
        11, 12,          // shoulders
        13, 14,          // elbows
        15, 16,          // wrists
        23, 24,          // hips
        25, 26,          // knees (corrected: 25=L_knee, 26=R_knee in MP)
        27, 28,          // ankles
        29, 30,          // heels
        31, 32,          // foot index
        0,               // nose (head proxy)
        7,  8,           // ears (head width)
        9, 10,           // mouth
        17, 18,          // pinkies
        19, 20,          // index fingers
        21, 22           // thumbs
    )

    // Specific semantic indices in the full 33-point set
    const val L_SHOULDER = 11
    const val R_SHOULDER = 12
    const val L_ELBOW    = 13
    const val R_ELBOW    = 14
    const val L_WRIST    = 15
    const val R_WRIST    = 16
    const val L_HIP      = 23
    const val R_HIP      = 24
    const val L_KNEE     = 25
    const val R_KNEE     = 26
    const val L_ANKLE    = 27
    const val R_ANKLE    = 28
    const val L_HEEL     = 29
    const val R_HEEL     = 30
    const val L_FOOT_IDX = 31
    const val R_FOOT_IDX = 32

    /**
     * Extract 28-point subset from a full 33×3 frame.
     * Returns [28][3] float array.
     */
    fun extract28(frame33x3: Array<FloatArray>): Array<FloatArray> {
        require(frame33x3.size >= 33) { "Expected 33 landmarks, got ${frame33x3.size}" }
        return Array(28) { i -> frame33x3[IDX_28[i]].copyOf() }
    }

    /**
     * Placeholder for future 84-channel kinematic embedding.
     * Currently returns a zero-padded [84] descriptor for each frame.
     */
    fun toKinematic84(frame33x3: Array<FloatArray>): FloatArray {
        val out = FloatArray(84)
        // Fill first 28*3 = 84 slots from the 28-point subset
        val pts = extract28(frame33x3)
        for (i in 0 until 28) {
            out[i * 3 + 0] = pts[i][0]
            out[i * 3 + 1] = pts[i][1]
            out[i * 3 + 2] = if (pts[i].size > 2) pts[i][2] else 0f
        }
        return out
    }

    /**
     * Returns knee angle indices in the STGCN [23]-dim joint_angles output.
     * Convention (from training):
     *   index 6 = left knee flexion angle (deg, 0=straight, 180=fully flexed)
     *   index 7 = right knee flexion angle
     */
    const val JA_L_KNEE = 6
    const val JA_R_KNEE = 7
}
