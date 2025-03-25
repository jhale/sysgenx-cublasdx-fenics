cmake_minimum_required(VERSION 3.8)

# Enable CUDA
enable_language(CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect CUDA architecture
include(CheckCUDACompilerFlag)
check_cuda_compiler_flag("-arch=sm_80" CUDA_SUPPORTS_SM_80)
if(CUDA_SUPPORTS_SM_80)
    set(CMAKE_CUDA_ARCHITECTURES 80)
else()
    message(WARNING "CUDA architecture detection failed. Using default architecture.")
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
set(CMAKE_CUDA_ARCHITECTURES 70)

# Add executable
add_executable(vector_add vector_add.cu)

# Set CUDA architecture
set_target_properties(vector_add PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
) 
