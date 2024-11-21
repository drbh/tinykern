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
