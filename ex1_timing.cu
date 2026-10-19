#include "cuda_utils.cuh"

__global__ void add_blocks(const int *a, const int *b, int *c) {
    int i = blockIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void add_threads(const int *a, const int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void multiply_vector(const int *a, const int *b, int *c, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}
__global__ void multiply_matrix(const int *a, const int *b, int *c,
                                int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        size_t i = (size_t)y * cols + x;
        c[i] = a[i] * b[i];
    }
}

void run_case(const char *label, size_t n, int mode) {
    size_t bytes = n * sizeof(int);
    int *a = (int *)host_alloc(bytes), *b = (int *)host_alloc(bytes);
    int *c = (int *)host_alloc(bytes);
    random_ints(a, n); random_ints(b, n);
    int *da, *db, *dc;
    CHECK(cudaMalloc((void **)&da, bytes));
    CHECK(cudaMalloc((void **)&db, bytes));
    CHECK(cudaMalloc((void **)&dc, bytes));
    printf("%s: %zu elements\n", label, n);
    for (int rep = 1; rep <= 3; ++rep) {
        double whole_start = cpu_second();
        CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
        double kernel_start = cpu_second();
        if (mode == 0) add_blocks<<<512, 1>>>(da, db, dc);
        else if (mode == 1) add_threads<<<1, 512>>>(da, db, dc);
        else if (mode == 2) multiply_vector<<<(n + 511) / 512, 512>>>(da, db, dc, n);
        else multiply_matrix<<<dim3(313, 625), dim3(32, 16)>>>(da, db, dc, 10000, 10000);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        double kernel_time = cpu_second() - kernel_start;
        CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
        double whole_time = cpu_second() - whole_start;
        size_t errors = 0;
        for (size_t i = 0; i < n; ++i)
            errors += c[i] != (mode <= 1 ? a[i] + b[i] : a[i] * b[i]);
        printf("run %d: kernel=%.7f s, copy+kernel+copy=%.7f s, mismatches=%zu\n",
               rep, kernel_time, whole_time, errors);
        if (errors) exit(1);
    }
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
    free(a); free(b); free(c);
}

int main() {
    run_case("Lab 9 Ex 3: 512 blocks x 1 thread", 512, 0);
    run_case("Lab 9 Ex 4: 1 block x 512 threads", 512, 1);
    run_case("Lab 9 Ex 5: 10 million vector", 10000000, 2);
    run_case("Lab 9 Ex 6: 100 million matrix", 100000000, 3);
    run_case("Transfer comparison: 100 million vector", 100000000, 2);
    return 0;
}
