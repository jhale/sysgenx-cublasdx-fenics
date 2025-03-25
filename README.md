# sysgenx-cublasdx-fenics

This repository contains experiments assembling simple matrix-free finite
element kernels using CUBlasDX

## Instructions

### Building the Dockerfile

    export ENGINE=podman
    cd docker
    ${ENGINE} build -t jhale/sysgenx-cublasdx-fenics .
    ${ENGINE} run --rm -ti --device nvidia.com/gpu=0 jhale/sysgenx-cublasdx-fenics:latest

### Launching a container

    ${ENGINE} run -v $(pwd):/shared -w /shared --rm -ti --device nvidia.com/gpu=0 jhale/sysgenx-cublasdx-fenics:latest

### Downloading cuBLASDx

    cd test_vector_add/
    wget https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/nvidia-mathdx-25.01.1.tar.gz
    tar -xvf nvidia-mathdx-25.01.1.tar.gz 
    
### Build and run test

    cd test_vector_add
    mkdir build
    cd build
    cmake ../
