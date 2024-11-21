#include <torch/extension.h>
#include <cuda_runtime.h>

void launch_matmul_kernel(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block);

at::Tensor matmul_kernel_wrapper(
    at::Tensor A,
    at::Tensor B,
    int grid_x, int grid_y,
    int block_x, int block_y) {
    
    // check that all inputs are on CUDA
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // check dimensions
    TORCH_CHECK(A.dim() == 2, "A must be 2-dimensional");
    TORCH_CHECK(B.dim() == 2, "B must be 2-dimensional");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    // create output tensor
    auto C = torch::zeros({A.size(0), B.size(1)}, 
                          torch::TensorOptions().device(A.device()).dtype(A.dtype()));

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    launch_matmul_kernel(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        A.size(0),
        grid,
        block);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_kernel", &matmul_kernel_wrapper, "Matrix multiplication kernel wrapper");
}
