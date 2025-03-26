cmake_minimum_required(VERSION 3.8)
project(sysgenx-cublasdx-fenics C CXX CUDA)
include(FetchContent)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number --expt-relaxed-constexpr") # Show error/warning numbers
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CUBLASDX_CUDA_CXX_FLAGS}\"")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr")

FetchContent_Declare(nvidia-mathdx
    URL https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/nvidia-mathdx-25.01.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(nvidia-mathdx)
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG PATHS "${nvidia-mathdx_SOURCE_DIR}/nvidia/mathdx/25.01")

find_package(Basix REQUIRED CONFIG)
find_package(DOLFINX REQUIRED)

add_executable(mass_example mass_example.cu)

set_property(TARGET mass_example PROPERTY CXX_STANDARD 20)
set_target_properties(mass_example PROPERTIES
    CUDA_ARCHITECTURES native
)

target_link_libraries(mass_example mathdx::cublasdx Basix::basix dolfinx)
