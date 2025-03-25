# sysgenx-cublasdx-fenics

## Building the Dockerfile

    export ENGINE=podman
    cd docker
    ${ENGINE} build -t jhale/sysgenx-cublasdx-fenics .
    ${ENGINE} run --rm -ti --device nvidia.com/gpu=0 jhale/sysgenx-cublasdx-fenics:latest

## Running a test

    ${ENGINE} run -v $(pwd):/shared -w /shared --rm -ti --device nvidia.com/gpu=0 jhale/sysgenx-cublasdx-fenics:latest
    cd test_vector_add
    mkdir build
    cd build
    cmake ../
