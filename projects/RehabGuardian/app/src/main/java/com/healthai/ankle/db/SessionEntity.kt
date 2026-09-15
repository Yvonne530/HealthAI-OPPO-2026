package com.healthai.ankle.db

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * One rehabilitation session (e.g., one exercise bout).
 */
@Entity(tableName = "sessions")
data class SessionEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val startTimeMs: Long = System.currentTimeMillis(),
    val endTimeMs: Long = 0L,
    val userWeightKg: Float = 70f,
    val avgMotionScore: Float = 0f,
    val maxRiskLabel: Int = 0,       // 0=low 1=medium 2=high
    val totalFrames: Int = 0,
    val notes: String = ""
)

/**
 * Single inference frame within a session.
 */
@Entity(
    tableName = "frames",
    foreignKeys = [ForeignKey(
        entity = SessionEntity::class,
        parentColumns = ["id"],
        childColumns = ["sessionId"],
        onDelete = ForeignKey.CASCADE
    )],
    indices = [Index("sessionId")]
)
data class FrameEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val sessionId: Long,
    val timestampMs: Long = System.currentTimeMillis(),

    // Risk
    val riskScore: Float,
    val riskLabel: Int,
    val confidence: Float,

    // Knee angles (degrees)
    val lKneeDeg: Float,
    val rKneeDeg: Float,

    // Motion quality
    val motionScore: Float,

    // Compact serialization of joint angles [23] as comma-separated string
    val jointAnglesJson: String = "",

    // GRF vertical components
    val grfLeftFz: Float = 0f,
    val grfRightFz: Float = 0f,

    // Risk explanation bitmask or short text
    val explanations: String = ""
)
