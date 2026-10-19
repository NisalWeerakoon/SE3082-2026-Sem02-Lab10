#include "cuda_utils.cuh"
#include <cstring>

__global__ void add(const int *a, const int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main(int argc, char **argv) {
    const char *test = argc > 1 ? argv[1] : "normal";
    const int n = 512;
    const size_t bytes = n * sizeof(int);
    int *da, *db, *dc;
    // These modes intentionally fail to prove CHECK reports errors.
    if (!strcmp(test, "oom")) CHECK(cudaMalloc((void **)&da, (size_t)1 << 50));
    CHECK(cudaMalloc((void **)&da, bytes));
    CHECK(cudaMalloc((void **)&db, bytes));
    CHECK(cudaMalloc((void **)&dc, bytes));
    int a[n], b[n], c[n];
    random_ints(a, n); random_ints(b, n);
    CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
    double start = cpu_second();
    if (!strcmp(test, "invalid")) add<<<n, 2048>>>(da, db, dc);
    else add<<<n, 1>>>(da, db, dc);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double elapsed = cpu_second() - start;
    CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i) errors += c[i] != a[i] + b[i];
    printf("Verification: %d mismatches out of %d\nTime taken - %.7f\n", errors, n, elapsed);
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
    return errors ? 1 : 0;
}
