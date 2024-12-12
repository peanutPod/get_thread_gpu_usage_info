// FILE: add.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    cudaSetDevice(1);
    const int arraySize = 1000000; // 增加数组大小
    int *a = new int[arraySize];
    int *b = new int[arraySize];
    int *c = new int[arraySize];

    // 初始化数组
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, arraySize * sizeof(int));
    cudaMalloc((void**)&d_c, arraySize * sizeof(int));

    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;

    // 启动 CUDA 内核
    add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, arraySize);

    cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 输出部分结果
    std::cout << "Result: ";
    for (int i = 0; i < 10; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
