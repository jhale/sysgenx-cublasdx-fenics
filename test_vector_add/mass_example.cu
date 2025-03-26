#include <chrono>
#include <iostream>
#include <vector>

#include <basix/finite-element.h>
#include <basix/quadrature.h>

#include <dolfinx.h>
#include <dolfinx/fem/Function.h>

#include <cublasdx.hpp>
#include <cuda_runtime_api.h>

using T = double;

constexpr int P = 4;
constexpr std::size_t num_dofs = (P + 1) * (P + 2) * (P + 3) / 6;
constexpr std::size_t batch_size = 64;
constexpr std::size_t m = num_dofs;
constexpr std::size_t n = batch_size;
constexpr std::size_t k = num_dofs;
constexpr std::size_t num_quadrature_points = num_dofs;

constexpr std::size_t num_elements = 32 * 32 * 32 * 6;

template <class GEMM, class GEMM_T>
__global__ void gemm_kernel_shared(const T* phi, const T* u, const T* c,
                                   const T* detJ, T* output, size_t num_dofs)
{
  extern __shared__ __align__(16) char smem[];

  // Get block index
  const int block_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  const size_t u_offset = block_idx * batch_size * num_dofs;
  const size_t c_offset = block_idx * batch_size * num_dofs;
  const size_t detJ_offset = block_idx * batch_size * num_quadrature_points;

  //  Copy tensors to shared memory
  auto phi_tensor = cublasdx::make_tensor(phi, GEMM::get_layout_gmem_a());
  auto u_tensor
      = cublasdx::make_tensor(u + u_offset, GEMM::get_layout_gmem_b());
  auto c_tensor
      = cublasdx::make_tensor(c + c_offset, GEMM::get_layout_gmem_c());

  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
  auto a_shared_tensor
      = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared_tensor
      = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared_tensor
      = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  using alignment = cublasdx::alignment_of<GEMM>;

  // First GEMM: phi * U
  cublasdx::copy<GEMM, alignment::a>(phi_tensor, a_shared_tensor);
  cublasdx::copy<GEMM, alignment::b>(u_tensor, b_shared_tensor);
  cublasdx::copy_wait();

  T alpha1 = 1.0;
  T beta1 = 0.0;
  GEMM().execute(alpha1, a_shared_tensor, b_shared_tensor, beta1,
                 c_shared_tensor);

  //  Scale by Jacobian determinant
  for (int i = thread_idx; i < num_quadrature_points * batch_size;
       i += blockDim.x)
  {
    c_shared_tensor[i] = c_shared_tensor[i] * detJ[i];
  }

  __syncthreads();

  T alpha2 = 1.0;
  T beta2 = 0.0;
  GEMM_T().execute(alpha2, a_shared_tensor, c_shared_tensor, beta2,
                   b_shared_tensor);

  // Copy result back to global memory
  auto out_global_tensor
      = cublasdx::make_tensor(output + c_offset, GEMM::get_layout_gmem_c());
  cublasdx::copy<GEMM, alignment::c>(b_shared_tensor, out_global_tensor);
}

int main(int argc, char* argv[])
{
  {
    MPI_Init(&argc, &argv);
    [[maybe_unused]] constexpr int Arch = 700;

    dolfinx::init_logging(argc, argv);
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box<T>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {32, 32, 32},
        mesh::CellType::hexahedron, part));

    // Tabulation of basis functions
    basix::FiniteElement element = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::tetrahedron, P,
        basix::element::lagrange_variant::equispaced,
        basix::element::dpc_variant::unset, false);

    auto [x, weights] = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::Default, basix::cell::type::tetrahedron,
        basix::polyset::type::standard, 2 * P);

    auto [table, shape] = element.tabulate(1, points, {weights.size(), 3});

    auto V
        = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
            mesh, std::make_shared<fem::FiniteElement<T>>(element)));

    auto u_function = std::make_shared<fem::Function<T>>(V);
    u_function->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            f.push_back(x(0, p) + x(1, p) + x(2, p));
          }

          return {f, {f.size()}};
        });

    auto arrangement
        = cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major,
                                cublasdx::row_major>();
    auto size = cublasdx::Size<m, n, k>();
    auto precision = cublasdx::Precision<T>();
    auto type = cublasdx::Type<cublasdx::type::real>();
    auto function = cublasdx::Function<cublasdx::function::MM>();
    auto sm = cublasdx::SM<Arch>();
    auto block = cublasdx::Block();
    auto block_dim = cublasdx::BlockDim<256>();

    using GEMM = decltype(size + precision + type + arrangement + function + sm
                          + block + block_dim);

    auto transpose = cublasdx::Arrangement<cublasdx::arrangement::col_major,
                                           cublasdx::arrangement::row_major,
                                           cublasdx::arrangement::col_major>();
    using GEMM_T = decltype(size + precision + type + transpose + function + sm
                            + block + block_dim);

    // Allocate memory
    constexpr auto num_quadrature_points = num_dofs;
    constexpr auto phi_size
        = num_elements * num_quadrature_points; // Batch size
    constexpr auto u_size = num_elements * num_dofs;
    constexpr auto global_c_size = num_elements * num_dofs;
    constexpr auto detJ_size = num_elements * num_quadrature_points;

    T* phi;  // basis functions
    T* u;    // coefficients DG style
    T* detJ; // Jacobian determinant
    T* c;    // result

    cudaMallocManaged(&phi, phi_size * sizeof(T));
    cudaMallocManaged(&u, u_size * sizeof(T));
    cudaMallocManaged(&c, global_c_size * sizeof(T));
    cudaMallocManaged(&detJ, detJ_size * sizeof(T));

    // Initialize phi with basis functions
    for (int i = 0; i < phi_size; ++i)
    {
      phi[i] = table[i];
    }

    // Initialize detJ with Jacobian determinant (Assumed to be 1 for now)
    for (int i = 0; i < detJ_size; ++i)
    {
      detJ[i] = 1.0 / num_elements; // TODO: compute
    }

    // Copy u coefficients to device
    std::shared_ptr dofmap = V->dofmap();
    auto [dof_indices, unrolled] = dofmap->dof_indices();

    for (int i = 0; i < u_size; ++i)
    {
      u[i] = u_function->vector()->mutable_array()[dof_indices[i]];
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
      gemm_kernel_shared<GEMM, GEMM_T>
          <<<grid_size, block_dim, cublasdx::get_shared_storage_size<GEMM>()>>>(
              phi, u, c, detJ, c, u_size);
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
    std::cout << "Kernel execution time: " << milliseconds << " ms"
              << std::endl;
    std::cout << "Throughput: "
              << 10 * (3 * global_c_size * sizeof(T) / (milliseconds / 1000.0))
                     / 1e9
              << " GB/s" << std::endl;
    std::cout << "Grid size: " << grid_size << ", Block size: " << block_dim
              << std::endl;
    std::cout << "Number of batches: " << num_batches << std::endl;

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Clean up memory
    cudaFree(phi);
    cudaFree(u);
    cudaFree(c);
    cudaFree(detJ);
  }

  MPI_Finalize();
  return 0;
}
