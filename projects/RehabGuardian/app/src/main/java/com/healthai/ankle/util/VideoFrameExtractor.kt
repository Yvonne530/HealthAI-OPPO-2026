package com.healthai.ankle.util

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log

/**
 * Simple video frame extractor using MediaMetadataRetriever.
 * Extracts frames at a fixed interval from an MP4 file.
 */
object VideoFrameExtractor {

    private const val TAG = "VideoFrameExtractor"

    data class FrameInfo(
        val bitmap: Bitmap,
        val index: Int,
        val timestampUs: Long
    )

    /**
     * Extract frames from a video file at approximately the given interval (in seconds).
     * Returns a list of decoded RGB bitmaps with timestamps.
     *
     * @param videoPath Absolute path to the MP4 file
     * @param intervalSec Frame extraction interval (default 0.1s = ~10fps)
     * @param maxWidth  Maximum width (video is scaled proportionally)
     * @param maxHeight Maximum height (video is scaled proportionally)
     */
    fun extractFrames(
        videoPath: String,
        intervalSec: Double = 0.1,
        maxWidth: Int = 640,
        maxHeight: Int = 480
    ): List<FrameInfo> {
        val results = mutableListOf<FrameInfo>()
        val retriever = MediaMetadataRetriever()

        try {
            retriever.setDataSource(videoPath)

            val durationUs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                ?.toIntOrNull() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                ?.toIntOrNull() ?: 0

            val intervalUs = (intervalSec * 1_000_000).toLong()
            val totalFrames = if (durationUs > 0 && intervalUs > 0) (durationUs / intervalUs).toInt() else 0

            Log.i(TAG, "Video: ${width}x${height}, duration=${durationUs / 1_000_000}s, interval=${intervalSec}s, ~$totalFrames frames")

            var frameIndex = 0
            for (timestampUs in 0L until durationUs step intervalUs) {
                val bitmap = retriever.getFrameAtTime(
                    timestampUs,
                    MediaMetadataRetriever.OPTION_CLOSEST
                )

                if (bitmap != null) {
                    // Scale down if needed while preserving aspect ratio
                    val scaled = scaleBitmap(bitmap, maxWidth, maxHeight, width, height)
                    results.add(FrameInfo(scaled, frameIndex, timestampUs))
                    frameIndex++

                    if (frameIndex % 50 == 0) {
                        Log.i(TAG, "Extracted $frameIndex/$totalFrames frames...")
                    }
                }
            }

            Log.i(TAG, "Total frames extracted: ${results.size}")

        } catch (e: Exception) {
            Log.e(TAG, "Frame extraction error: ${e.message}", e)
        } finally {
            try { retriever.release() } catch (_: Exception) {}
        }

        return results
    }

    private fun scaleBitmap(
        bitmap: Bitmap,
        maxWidth: Int,
        maxHeight: Int,
        origWidth: Int,
        origHeight: Int
    ): Bitmap {
        if (origWidth <= maxWidth && origHeight <= maxHeight) return bitmap

        val scale = minOf(maxWidth.toFloat() / origWidth, maxHeight.toFloat() / origHeight)
        val newW = (origWidth * scale).toInt()
        val newH = (origHeight * scale).toInt()

        return Bitmap.createScaledBitmap(bitmap, newW, newH, true)
    }
}
