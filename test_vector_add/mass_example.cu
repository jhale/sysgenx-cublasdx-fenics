#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>


template<class GEMM>
__global__ void gemm_kernel_shared(const typename GEMM::c_value_type  alpha,
                                   const typename GEMM::a_value_type* a,
                                   const typename GEMM::b_value_type* b,
                                   const typename GEMM::c_value_type  beta,
                                   typename GEMM::c_value_type* c) {
    extern __shared__ __align__(16) char smem[];

    // Make global memory tensor
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

    // Make shared memory tensor
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

    // Load data from global memory tensor to shared memory tensor
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<GEMM, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    // Execute GEMM
    GEMM().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
    __syncthreads();

    // Store data from shared memory tensor to global memory tensor
    cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
}



int main(int, char**) {

    using value_type = double;

    constexpr int m = 32;
    constexpr int n = 32;
    constexpr int k = 32;

    constexpr int block_size = 256;
    constexpr int Arch = 800;
    
    // GEMM definition using cuBLASDx operators
    using GEMM = decltype(cublasdx::Size<m, n, k>()
                  + cublasdx::Precision<value_type>()
                  + cublasdx::Type<cublasdx::type::real>()
                  + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
                  + cublasdx::Function<cublasdx::function::MM>()
                  + cublasdx::SM<700>()
                  + cublasdx::Block()
                  + cublasdx::BlockDim<256>());


    constexpr auto global_a_size = m * k;
    constexpr auto global_b_size = k * n;
    constexpr auto global_c_size = m * n;

    value_type* a;
    value_type* b;
    value_type* c;

    cudaMallocManaged(&a, global_a_size * sizeof(value_type));
    cudaMallocManaged(&b, global_b_size * sizeof(value_type));
    cudaMallocManaged(&c, global_c_size * sizeof(value_type));

    // Initialize matrices
    for (int i = 0; i < global_a_size; ++i) {
        a[i] = static_cast<value_type>(1);
    }
    for (int i = 0; i < global_b_size; ++i) {
        b[i] = static_cast<value_type>(1);
    }
    gemm_kernel_shared<GEMM><<<1, GEMM::block_dim, cublasdx::get_shared_storage_size<GEMM>()>>>(1.0, a, b, 1.0, c);
    cudaPeekAtLastError();

    cudaDeviceSynchronize();

    // Print results
    for (int i = 0; i < global_c_size; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
