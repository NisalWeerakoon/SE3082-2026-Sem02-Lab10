#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, code:%d, reason: %s\n", \
                __FILE__, __LINE__, (int)e, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

inline double cpu_second() {
    timeval t;
    gettimeofday(&t, nullptr);
    return t.tv_sec + t.tv_usec * 1e-6;
}

inline void *host_alloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "Host allocation failed: %zu bytes\n", n); exit(1); }
    return p;
}

inline void random_ints(int *p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = rand() % 100;
}
#endif
