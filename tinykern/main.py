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
__global__ void matmul_kernel(float* A, float* B, float* C, int N);

__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
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
"""

cpp_source = """
#include <torch/extension.h>

void matmul_kernel_wrapper(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    dim3 grid,
    dim3 block,
    int N) {
    
    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_kernel", &matmul_kernel_wrapper, "matmul kernel");
}
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=["matmul_kernel"],
    verbose=True,
)


def main():
    config = MatMulConfig()

    A = torch.rand(config.size, config.size, device="cuda", dtype=torch.float32)
    B = torch.rand(config.size, config.size, device="cuda", dtype=torch.float32)
    C = torch.zeros(config.size, config.size, device="cuda", dtype=torch.float32)

    grid, block = config.grid_block_dims
    matmul_cuda.matmul_kernel(A, B, C, dim3(grid), dim3(block), config.size)

    torch_result = torch.mm(A, B)
    max_diff = torch.max(torch.abs(C - torch_result)).item()
    print(f"Max difference: {max_diff:.2e}")


if __name__ == "__main__":
    main()
