#include "cuda_utils.cuh"

constexpr int NX = 500, NY = 500, NZ = 500;

__global__ void multiply_volume(const int *a, const int *b, int *c,
                                int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        size_t i = ((size_t)z * ny + y) * nx + x;
        c[i] = a[i] * b[i];
    }
}

int main() {
    size_t count = (size_t)NX * NY * NZ;
    size_t bytes = count * sizeof(int);
    int *a = (int *)host_alloc(bytes), *b = (int *)host_alloc(bytes);
    int *c = (int *)host_alloc(bytes);
    random_ints(a, count); random_ints(b, count);
    int *da, *db, *dc;
    CHECK(cudaMalloc((void **)&da, bytes));
    CHECK(cudaMalloc((void **)&db, bytes));
    CHECK(cudaMalloc((void **)&dc, bytes));
    CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
    dim3 threads(8, 8, 8);
    dim3 blocks((NX + 7) / 8, (NY + 7) / 8, (NZ + 7) / 8);
    printf("Grid: %u x %u x %u blocks\n", blocks.x, blocks.y, blocks.z);
    double start = cpu_second();
    multiply_volume<<<blocks, threads>>>(da, db, dc, NX, NY, NZ);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double elapsed = cpu_second() - start;
    CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = count - 1000; i < count; ++i) {
        size_t z = i / (NX * NY), y = (i / NX) % NY, x = i % NX;
        printf("C[%zu][%zu][%zu] = %d x %d = %d\n", z, y, x, a[i], b[i], c[i]);
    }
    size_t errors = 0;
    for (size_t i = 0; i < count; ++i) errors += c[i] != a[i] * b[i];
    printf("Verification: %zu mismatches out of %zu\nTime taken - %.7f\n",
           errors, count, elapsed);
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
    free(a); free(b); free(c);
    return errors ? 1 : 0;
}
