cmake_minimum_required(VERSION 3.8)
project(sysgenx-cublasdx-fenics)

# Enable CUDA
enable_language(CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG PATHS "nvidia-mathdx-25.01.1/nvidia/mathdx/25.01")

# Add executable
add_executable(mass_example mass_example.cu)

# Set CUDA architecture
set_target_properties(mass_example PROPERTIES
    CUDA_ARCHITECTURES native
)

target_link_libraries(mass_example mathdx::cublasdx)
