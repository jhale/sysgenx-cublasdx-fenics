#include <stdio.h>
#include <stdlib.h>

// Size of array
#define N 1048576

// Error checking macro
#define CUDA_CHECK(call) \
	do { \
		cudaError_t error = call; \
		if (error != cudaSuccess) { \
			fprintf(stderr, "CUDA error at %s:%d: %s\n", \
					__FILE__, __LINE__, cudaGetErrorString(error)); \
			exit(EXIT_FAILURE); \
		} \
	} while(0)


int main(void) {
    int P = 3;
    int nq = (P+1)*(P+2)*(P+3)/6;
    int nq_squared = (nq+1)*(nq+1);

    // Allocate host memory for basis function values
    double *phi = (double *)malloc(nq_squared * sizeof(double));
    if (phi == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Number of elements
    int num_elements = 100000;
    int num_dofs = num_elements * nq;

    // Allocate host memory for solution vector
    double *u = (double *)malloc(num_dofs * sizeof(double));
    if (u == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Memory allocation for device arrays
    double *d_phi, *d_u;
    CUDA_CHECK(cudaMalloc(&d_phi, nq_squared * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u, num_dofs * sizeof(double)));

    // Copy basis function values from host to device
    CUDA_CHECK(cudaMemcpy(d_phi, phi, nq_squared * sizeof(double), cudaMemcpyHostToDevice));
    
}