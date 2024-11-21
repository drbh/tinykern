import torch
from dataclasses import dataclass
from typing import Tuple
import matmul_kernel

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

def main():
    config = MatMulConfig()
    A = torch.rand(config.size, config.size, device="cuda", dtype=torch.float32)
    B = torch.rand(config.size, config.size, device="cuda", dtype=torch.float32)
    
    grid, block = config.grid_block_dims
    C = matmul_kernel.matmul_kernel(
        A, B,
        grid[0], grid[1],  # grid dimensions
        block[0], block[1]  # block dimensions
    )
    
    torch_result = torch.mm(A, B)
    max_diff = torch.max(torch.abs(C - torch_result)).item()
    print(f"Max difference: {max_diff:.2e}")

if __name__ == "__main__":
    main()
