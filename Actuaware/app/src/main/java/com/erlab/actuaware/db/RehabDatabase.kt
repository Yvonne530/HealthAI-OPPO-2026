package com.healthai.ankle.db

import androidx.room.*
import kotlinx.coroutines.flow.Flow

// ── DAO ──────────────────────────────────────────────────────────────────────

@Dao
interface RehabDao {

    // Sessions
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSession(session: SessionEntity): Long

    @Update
    suspend fun updateSession(session: SessionEntity)

    @Query("SELECT * FROM sessions ORDER BY startTimeMs DESC")
    fun allSessions(): Flow<List<SessionEntity>>

    @Query("SELECT * FROM sessions WHERE id = :id")
    suspend fun sessionById(id: Long): SessionEntity?

    @Delete
    suspend fun deleteSession(session: SessionEntity)

    // Frames
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertFrame(frame: FrameEntity): Long

    @Query("SELECT * FROM frames WHERE sessionId = :sid ORDER BY timestampMs ASC")
    suspend fun framesForSession(sid: Long): List<FrameEntity>

    /** Best frame = lowest risk among frames where both knees ≥ 150°; fallback = global min risk. */
    @Query("""
        SELECT * FROM frames WHERE sessionId = :sid
        AND lKneeDeg >= 150 AND rKneeDeg >= 150
        ORDER BY riskScore ASC LIMIT 1
    """)
    suspend fun bestFrameForSession(sid: Long): FrameEntity?

    @Query("SELECT * FROM frames WHERE sessionId = :sid ORDER BY riskScore ASC LIMIT 1")
    suspend fun lowestRiskFrame(sid: Long): FrameEntity?

    @Query("SELECT COUNT(*) FROM frames WHERE sessionId = :sid")
    suspend fun frameCount(sid: Long): Int

    @Query("""
        SELECT AVG(motionScore) FROM frames WHERE sessionId = :sid
    """)
    suspend fun avgMotionScore(sid: Long): Float
}

// ── Database ──────────────────────────────────────────────────────────────────

@Database(
    entities = [SessionEntity::class, FrameEntity::class],
    version = 1,
    exportSchema = false
)
abstract class RehabDatabase : RoomDatabase() {
    abstract fun dao(): RehabDao

    companion object {
        @Volatile private var INSTANCE: RehabDatabase? = null

        fun getInstance(context: android.content.Context): RehabDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    RehabDatabase::class.java,
                    "rehab_guardian.db"
                ).build().also { INSTANCE = it }
            }
    }
}