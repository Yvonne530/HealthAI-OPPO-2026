cmake_minimum_required(VERSION 3.22.1)
project("llama_android")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 性能优化选项
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 启用NEON指令集（ARM优化）
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a72")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a72")
endif()

# 查找必要的包
find_package(Threads REQUIRED)

# 添加llama.cpp库
# 注意：需要先下载llama.cpp源代码到这个目录
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
    message(STATUS "llama.cpp source not found, please download it first")
endif()

# 包含llama.cpp的源文件
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
    add_subdirectory(llama.cpp)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/src)
endif()

# 添加mtmd库（多模态支持）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd")
    # mtmd源文件
    set(MTMD_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/mtmd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/mtmd-audio.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/mtmd-helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/clip.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/models.h
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/cogvlm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/conformer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/glm4v.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/internvl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/kimivl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/llama4.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/llava.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/minicpmv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/pixtral.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/qwen2vl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/qwen3vl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/siglip.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/whisper-enc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/mobilenetv5.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models/youtuvl.cpp
    )

    add_library(mtmd STATIC ${MTMD_SOURCES})
    target_include_directories(mtmd PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/models
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
    )
    target_link_libraries(mtmd PUBLIC ggml llama Threads::Threads)
    target_compile_features(mtmd PRIVATE cxx_std_17)
endif()

# 添加JNI库
add_library(llama_jni SHARED
    llama_jni.cpp
)

# 定义LLAMA_CPP_EXISTS宏
if(TARGET llama)
    target_compile_definitions(llama_jni PRIVATE LLAMA_CPP_EXISTS)
endif()

# 链接llama.cpp库
if(TARGET llama)
    target_link_libraries(llama_jni PRIVATE
        llama
        ggml
        Threads::Threads
        log
        android
    )
endif()

# 链接mtmd库（如果存在）
if(TARGET mtmd)
    target_link_libraries(llama_jni PRIVATE mtmd)
    target_include_directories(llama_jni PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
    )
endif()

# 查找log库
find_library(log-lib log)
target_link_libraries(llama_jni PRIVATE ${log-lib})