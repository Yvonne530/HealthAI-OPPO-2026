package com.erlab.actuaware.inference

import java.util.*

class SlidingWindowBuffer<T>(private val windowSize: Int) {
    private val buffer = ArrayDeque<T>(windowSize)

    fun addFrame(frame: T) {
        if (buffer.size >= windowSize) buffer.removeFirst()
        buffer.addLast(frame)
    }

    fun isFull(): Boolean = buffer.size >= windowSize

    fun getSequence(): List<T>? = if (isFull()) buffer.toList() else null

    fun clear() = buffer.clear()

    val size: Int get() = buffer.size
}