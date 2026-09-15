cmake_minimum_required(VERSION 3.22.1)
project(rehab_mnn_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Only arm64-v8a (OPPO Reno15 Pro) ────────────────────────────────────────
if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Only arm64-v8a is supported. Got: ${ANDROID_ABI}")
endif()

# ── MNN 2.9.0 paths ───────────────────────────────────────────────────────────
# Headers: extracted by setup_mnn_290.sh → app/src/main/cpp/include/MNN/
# Library: extracted by setup_mnn_290.sh → app/src/main/jniLibs/arm64-v8a/libMNN.so
set(MNN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MNN_SO_PATH     "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so")

# Validate headers exist before continuing
if(NOT EXISTS "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp")
    message(FATAL_ERROR
        "MNN headers not found at ${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp\n"
        "Run: bash scripts/setup_mnn_290.sh  (from project root)\n"
        "This extracts MNN 2.9.0 headers and libMNN.so into the correct paths.")
endif()

# Validate libMNN.so exists
if(NOT EXISTS "${MNN_SO_PATH}")
    message(FATAL_ERROR
        "libMNN.so not found at ${MNN_SO_PATH}\n"
        "Run: bash scripts/setup_mnn_290.sh  (from project root)")
endif()

# ── Import libMNN.so as pre-built ─────────────────────────────────────────────
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES
    IMPORTED_LOCATION "${MNN_SO_PATH}"
)

# ── Our JNI bridge shared library ────────────────────────────────────────────
add_library(rehab_mnn_jni SHARED
    mnn_jni.cpp
)

target_include_directories(rehab_mnn_jni PRIVATE
    ${MNN_INCLUDE_DIR}
)

target_compile_options(rehab_mnn_jni PRIVATE
    -O2
    -fvisibility=hidden         # hide all symbols except JNI_OnLoad + Java_ exports
    -Wall
    -Wno-unused-parameter
)

target_link_libraries(rehab_mnn_jni
    MNN
    android   # ANativeWindow, AAssetManager
    log       # __android_log_print
)

# Strip debug symbols in release — reduces .so size by ~40%
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET rehab_mnn_jni POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-unneeded
            $<TARGET_FILE:rehab_mnn_jni>
    )
endif()
