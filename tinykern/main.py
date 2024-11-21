import torch
from torch.utils.cpp_extension import load_inline
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MatMulConfig:
    size: int = 16
    threads_per_block: int = 16
    
    @property
    def blocks(self) -> int:
        return (self.size + self.threads_per_block - 1) // self.threads_per_block
    
    @property
    def grid_block_dims(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (self.blocks, self.blocks),
            (self.threads_per_block, self.threads_per_block),
        )

cuda_source = """
extern "C" __global__ void matmul_kernel_cuda(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_matmul_kernel(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmul_kernel_cuda<<<grid, block>>>(A, B, C, N);
}
"""

cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

void launch_matmul_kernel(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block);

at::Tensor matmul_kernel_wrapper(
    at::Tensor A,
    at::Tensor B,
    int grid_x, int grid_y,
    int block_x, int block_y) {
    
    // Check that all inputs are on CUDA
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    // Check dimensions
    TORCH_CHECK(A.dim() == 2, "A must be 2-dimensional");
    TORCH_CHECK(B.dim() == 2, "B must be 2-dimensional");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    
    // Create output tensor
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
    m.def("matmul_kernel", &matmul_kernel_wrapper, "matmul kernel wrapper");
}
"""

matmul_kernel = load_inline(
    name="matmul_kernel",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    verbose=True,
)

def main():
    config = MatMulConfig()
    A = torch.rand(config.size, config.size, device="cuda", dtype=torch.float32)
    B = torch.rand(config.size, config.size, device="cuda", dtype=torch.float32)
    
    grid, block = config.grid_block_dims
    C = matmul_kernel.matmul_kernel(
        A, B,
        grid[0], grid[1], 
        block[0], block[1]
    )
    
    torch_result = torch.mm(A, B)
    max_diff = torch.max(torch.abs(C - torch_result)).item()
    print(f"Max difference: {max_diff:.2e}")
    
if __name__ == "__main__":
    main()