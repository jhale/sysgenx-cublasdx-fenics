# Experiments with cuBLASDx and FEniCS

This repository contains experimental code for assembling simple matrix-free
finite element kernels using GEMM functionality in
[cuBLASDx](https://docs.nvidia.com/cuda/cublasdx/). Basix data structures are
implemented using [DOLFINx](https://github.com/fenics/dolfinx). cuBLASDx allows
for BLAS kernels to be called within a kernel - this allows for fusing linear
algebra routines with other operations in a kernel, and even fusing linear
algebra routines together.

## Authors

- Igor Baratta, NVIDIA Corporation.
- Jack S. Hale, University of Luxembourg.

## Instructions

### Building the docker image

On the system

    export ENGINE=podman
    cd docker
    ${ENGINE} build -t jhale/sysgenx-cublasdx-fenics .
    ${ENGINE} run --rm -ti --device nvidia.com/gpu=0 jhale/sysgenx-cublasdx-fenics:latest

### Launching a container

    ${ENGINE} run -v $(pwd):/shared -w /shared --rm -ti --device nvidia.com/gpu=0 jhale/sysgenx-cublasdx-fenics:latest

### Build and run test

In the running container

    cd test_vector_add
    mkdir build
    cd build
    cmake ../
    make
    ./mass_example
