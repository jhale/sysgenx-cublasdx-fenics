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

// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
	// Number of bytes to allocate for N doubles
	size_t bytes = N*sizeof(double);

	// Allocate memory for arrays A, B, and C on host
	double *A = (double*)malloc(bytes);
	double *B = (double*)malloc(bytes);
	double *C = (double*)malloc(bytes);
	
	if (A == NULL || B == NULL || C == NULL) {
		fprintf(stderr, "Failed to allocate host memory\n");
		exit(EXIT_FAILURE);
	}

	// Allocate memory for arrays d_A, d_B, and d_C on device
	double *d_A, *d_B, *d_C;
	CUDA_CHECK(cudaMalloc(&d_A, bytes));
	CUDA_CHECK(cudaMalloc(&d_B, bytes));
	CUDA_CHECK(cudaMalloc(&d_C, bytes));

	// Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = 1.0;
		B[i] = 2.0;
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

	// Set execution configuration parameters
	//		thr_per_blk: number of CUDA threads per grid block
	//		blk_in_grid: number of blocks in grid
	int thr_per_blk = 256;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

	// Launch kernel
	add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);
	
	// Check for kernel launch errors
	CUDA_CHECK(cudaGetLastError());
	
	// Copy data from device array d_C to host array C
	CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());

	// Verify results
	double tolerance = 1.0e-14;
	for(int i=0; i<N; i++)
	{
		if( fabs(C[i] - 3.0) > tolerance)
		{ 
			printf("\nError: value of C[%d] = %f instead of 3.0\n\n", i, C[i]);
			exit(EXIT_FAILURE);
		}
	}	

	// Free CPU memory
	free(A);
	free(B);
	free(C);

	// Free GPU memory
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", thr_per_blk);
	printf("Blocks In Grid    = %d\n", blk_in_grid);
	printf("---------------------------\n\n");

	return 0;
}