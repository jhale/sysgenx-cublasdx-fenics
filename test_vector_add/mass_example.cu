#include <chrono>
#include <iostream>
#include <vector>

#include <basix/finite-element.h>

#include <cublasdx.hpp>
#include <cuda_runtime_api.h>

using value_type = double;

constexpr int P = 4;
constexpr std::size_t num_dofs = (P + 1) * (P + 2) * (P + 3) / 6;
constexpr std::size_t batch_size = 64;

constexpr std::size_t m = num_dofs;
constexpr std::size_t n = batch_size;
constexpr std::size_t k = num_dofs;

constexpr std::size_t num_elements = 1 << 20;

template <class GEMM>
__global__ void gemm_kernel_shared(const value_type* phi, const value_type* u,
                                   const value_type* c, const value_type alpha,
                                   const value_type beta, value_type* output,
                                   size_t num_dofs)
{
  extern __shared__ __align__(16) char smem[];

  // Get block index
  const int block_idx = blockIdx.x;

  const size_t u_offset = block_idx * batch_size * num_dofs;
  const size_t c_offset = block_idx * batch_size * num_dofs;

  auto phi_tensor = cublasdx::make_tensor(phi, GEMM::get_layout_gmem_a());
  auto u_tensor
      = cublasdx::make_tensor(u + u_offset, GEMM::get_layout_gmem_b());
  auto c_tensor
      = cublasdx::make_tensor(c + c_offset, GEMM::get_layout_gmem_c());

  auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
  auto a_shared_tensor
      = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared_tensor
      = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());

  using alignment = cublasdx::alignment_of<GEMM>;

  // First GEMM: phi * U
  cublasdx::copy<GEMM, alignment::a>(phi_tensor, a_shared_tensor);
  cublasdx::copy<GEMM, alignment::b>(u_tensor, b_shared_tensor);
  cublasdx::copy_wait();

  auto [c_frag, partitioner] = GEMM().execute(a_shared_tensor, b_shared_tensor);

  // Second GEMM: phi^T * (phi * U)
  auto d_frag = partitioner.make_accumulator_fragment();
  cublasdx::copy_fragment<alignment::c>(c_tensor, d_frag, partitioner);
  cublasdx::axpby(alpha, c_frag, beta, d_frag);

  // Copy result back to global memory
  auto out_global_tensor
      = cublasdx::make_tensor(output + c_offset, GEMM::get_layout_gmem_c());
  cublasdx::copy_fragment<alignment::c>(d_frag, out_global_tensor, partitioner);
}

int main(int, char**)
{
  constexpr int Arch = 700;

  // Tabulation of basis functions
  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::triangle, P,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::simplex_equispaced, false);

  // GEMM definition using cuBLASDx operators
  using GEMM
      = decltype(cublasdx::Size<m, n, k>() + cublasdx::Precision<value_type>()
                 + cublasdx::Type<cublasdx::type::real>()
                 + cublasdx::Arrangement<cublasdx::row_major,
                                         cublasdx::col_major>()
                 + cublasdx::Function<cublasdx::function::MM>()
                 + cublasdx::SM<Arch>() + cublasdx::Block()
                 + cublasdx::BlockDim<256>());

  constexpr auto global_a_size = m * k;
  constexpr auto global_b_size = num_elements * k * n;
  constexpr auto global_c_size = num_elements * m * n;

  // U = phi^T * phi * U

  value_type* a;
  value_type* b;
  value_type* c;

  cudaMallocManaged(&a, global_a_size * sizeof(value_type));
  cudaMallocManaged(&b, global_b_size * sizeof(value_type));
  cudaMallocManaged(&c, global_c_size * sizeof(value_type));

  // Initialize matrices
  for (int i = 0; i < global_a_size; ++i)
  {
    a[i] = static_cast<value_type>(1);
  }
  for (int i = 0; i < global_b_size; ++i)
  {
    b[i] = static_cast<value_type>(1);
  }

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Calculate grid and block dimensions
  constexpr int block_dim = 256; // Use the GEMM block size
  constexpr int num_batches = (num_elements + batch_size - 1) / batch_size;
  constexpr int grid_size = num_batches;

  // Record start event
  cudaEventRecord(start);

  // Launch kernel

  for (int i = 0; i < 10; ++i)
  {
    gemm_kernel_shared<GEMM>
        <<<grid_size, block_dim, cublasdx::get_shared_storage_size<GEMM>()>>>(
            a, b, c, 1.0, 0.0, c, num_dofs);
    cudaDeviceSynchronize();
  }

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  // Synchronize to ensure kernel completes
  cudaDeviceSynchronize();

  // Record stop event
  cudaEventRecord(stop);

  // Synchronize to ensure timing is accurate
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Print timing information
  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
  std::cout << "Throughput: "
            << 10
                   * (3 * global_c_size * sizeof(value_type)
                      / (milliseconds / 1000.0))
                   / 1e9
            << " GB/s" << std::endl;
  std::cout << "Grid size: " << grid_size << ", Block size: " << block_dim
            << std::endl;
  std::cout << "Number of batches: " << num_batches << std::endl;

  // Clean up CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Clean up memory
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return 0;
}
