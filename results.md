# Lab 10 — real Colab T4 results

Executed on 19 October 2026 in Google Colab using a T4 GPU. Times are wall-clock seconds, include `cudaDeviceSynchronize()` for kernel measurements, and naturally vary between runs. The first trial includes warm-up effects.

| Lab 9 program | Launch | Kernel run 1 | Run 2 | Run 3 |
| --- | --- | ---: | ---: | ---: |
| Exercise 3, 512 additions | 512 × 1 | 0.0005541 | 0.0000150 | 0.0000122 |
| Exercise 4, 512 additions | 1 × 512 | 0.0000401 | 0.0000129 | 0.0000110 |
| Exercise 5, 10 million products | 19,532 × 512 | 0.0005419 | 0.0005310 | 0.0005231 |
| Exercise 6, 100 million products | 313 × 625 blocks of 32 × 16 | 0.0049431 | 0.0049088 | 0.0049112 |

For the 100-million-element **vector** test, the three kernel-only times were 0.0046461, 0.0046279, and 0.0046329 s. The corresponding copy-in + kernel + copy-out times were 0.4407589, 0.2602382, and 0.2676840 s. All trials reported zero mismatches. Transfer overhead dominates this simple element-wise computation.

The `CHECK` normal run reported 0/512 mismatches. The intentional invalid launch was caught as `code:1, reason: invalid argument` in this CUDA runtime; the lab sheet's example says code 9, but the descriptive error and program stop confirm the check worked. The impossible one-petabyte allocation was caught as `code:2, reason: out of memory`.

The picture used a 50 × 38 grid of 16 × 16 threads; 240,121 pixels were capped, with 0/480,000 mismatches and kernel time 0.0001490 s. The volume used a 63 × 63 × 63 grid of 8 × 8 × 8 threads; it reported 0/125,000,000 mismatches and kernel time 0.0076661 s. The final volume element was `C[499][499][499] = 47 × 68 = 3196`.

The executed notebook, with complete program outputs including the required picture region and last 1,000 volume elements, is available in [Colab](https://colab.research.google.com/drive/1p5KRNL1d0HWmedRLvD4moE17BJNoXE2M).
