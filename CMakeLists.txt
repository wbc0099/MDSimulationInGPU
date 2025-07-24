cmake_minimum_required(VERSION 3.18)
project(QSP LANGUAGES CXX CUDA)

# 方法1：优先使用现代CMake的CUDAToolkit包
find_package(CUDAToolkit REQUIRED)

# 方法2：如果方法1失败，手动指定CUDA路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")  # 确保这个路径包含include和lib64
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# 设置CUDA架构（根据你的GPU调整）
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")  # 例如75对应Turing架构

add_executable(QSP
    main.cpp
    particle.cu
    parameters.cpp
    gpuConfig.cpp
)

# 现代链接方式（优先尝试）
target_link_libraries(QSP PRIVATE
    CUDA::cudart
    CUDA::curand
)

target_include_directories(QSP PRIVATE include)

# 传统链接方式（如果上面失败）
if(NOT TARGET CUDA::cudart)
    message(WARNING "Falling back to manual CUDA linking")
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_link_libraries(QSP PRIVATE
        cudart
        curand
    )
endif()

# 设置C++和CUDA标准
set_target_properties(QSP PROPERTIES
    CUDA_STANDARD 17
    CXX_STANDARD 17
)
