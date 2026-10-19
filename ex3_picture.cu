#include "cuda_utils.cuh"

constexpr int WIDTH = 800, HEIGHT = 600;

__global__ void picture_kernel(const unsigned char *in, unsigned char *out,
                               int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        int i = row * width + col;
        int doubled = 2 * in[i];
        out[i] = (unsigned char)(doubled > 255 ? 255 : doubled);
    }
}

int main() {
    size_t count = (size_t)WIDTH * HEIGHT;
    size_t bytes = count * sizeof(unsigned char);
    auto *in = (unsigned char *)host_alloc(bytes);
    auto *out = (unsigned char *)host_alloc(bytes);
    for (size_t i = 0; i < count; ++i) in[i] = rand() % 256;
    unsigned char *din, *dout;
    CHECK(cudaMalloc((void **)&din, bytes));
    CHECK(cudaMalloc((void **)&dout, bytes));
    CHECK(cudaMemcpy(din, in, bytes, cudaMemcpyHostToDevice));
    dim3 threads(16, 16);
    dim3 blocks((WIDTH + threads.x - 1) / threads.x,
                (HEIGHT + threads.y - 1) / threads.y);
    printf("Grid: %u x %u blocks of %u x %u threads\n",
           blocks.x, blocks.y, threads.x, threads.y);
    double start = cpu_second();
    picture_kernel<<<blocks, threads>>>(din, dout, WIDTH, HEIGHT);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double elapsed = cpu_second() - start;
    CHECK(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost));
    puts("Original top-left 50 x 50:");
    for (int y = 0; y < 50; ++y) {
        for (int x = 0; x < 50; ++x) printf("%3d ", in[y * WIDTH + x]);
        putchar('\n');
    }
    puts("Updated top-left 50 x 50:");
    for (int y = 0; y < 50; ++y) {
        for (int x = 0; x < 50; ++x) printf("%3d ", out[y * WIDTH + x]);
        putchar('\n');
    }
    size_t errors = 0, capped = 0;
    for (size_t i = 0; i < count; ++i) {
        int doubled = 2 * in[i];
        capped += doubled > 255;
        errors += out[i] != (doubled > 255 ? 255 : doubled);
    }
    printf("Verification: %zu mismatches out of %zu\nPixels capped: %zu\nTime taken - %.7f\n",
           errors, count, capped, elapsed);
    CHECK(cudaFree(din)); CHECK(cudaFree(dout));
    free(in); free(out);
    return errors ? 1 : 0;
}
