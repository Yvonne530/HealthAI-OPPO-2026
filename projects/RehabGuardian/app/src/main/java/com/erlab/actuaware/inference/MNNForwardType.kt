package com.healthai.ankle.inference

/**
 * MNN ForwardType constants.
 * MNN 1.x / 2.x both expose these values but via different class paths.
 * This shim centralises the constant so the engine stays version-agnostic.
 *
 * MNNNetInstance.Config.forwardType accepts an Int:
 *   0 = CPU   1 = Metal   2 = CUDA   3 = OpenCL   6 = OpenGL   7 = Vulkan
 */
enum class MNNForwardType(val type: Int) {
    FORWARD_CPU(0),
    FORWARD_METAL(1),
    FORWARD_CUDA(2),
    FORWARD_OPENCL(3),
    FORWARD_OPENGL(6),
    FORWARD_VULKAN(7)
}