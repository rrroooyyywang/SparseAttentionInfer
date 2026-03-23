// ── kernel ────────────────────────────────────────────────────────────────────
__global__ void helloWorldKernel(int* arr) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from thread %d, arr[%d] = %d\n",
           threadId, threadId % 5, arr[threadId % 5]);
}

// ── torch wrapper ─────────────────────────────────────────────────────────────
torch::Tensor cuda_helloworld_impl() {
    int hostArray[] = {1, 2, 3, 4, 5};
    int* deviceArray = nullptr;

    cudaMalloc(&deviceArray, 5 * sizeof(int));
    cudaMemcpy(deviceArray, hostArray, 5 * sizeof(int), cudaMemcpyHostToDevice);

    helloWorldKernel<<<10, 4>>>(deviceArray);
    cudaDeviceSynchronize();

    cudaFree(deviceArray);
    return torch::empty({1}, torch::kFloat32);  // dummy return to satisfy Tensor signature
}