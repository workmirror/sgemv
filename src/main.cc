#include <chrono>
#include <cstdlib>
#include <iostream>

#if defined(_MSC_VER)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#include "sgemm.h"

void* AlignedMalloc(size_t alignment, size_t nbytes) {
  void* data = nullptr;

  if (nbytes == 0) return data;

#if defined(_MSC_VER)
  data = _aligned_malloc(nbytes, alignment);
#else
  int ret = posix_memalign(&data, alignment, nbytes);
  if (ret != 0) {
    // TODO: throw error!
    return nullptr;
  }
#endif
  return data;
}

void AlignedFree(void* data) {
#ifdef _MSC_VER
  _aligned_free(data);
#else
  free(data);
#endif
  data = nullptr;
}

void RefGemv(float* C, const float* A, const float* B, size_t K, size_t N) {
  for (int j = 0; j < N; ++j) {
    C[j] = 0;

    for (int i = 0; i < K; ++i) {
      C[j] += A[i] * B[i * N + j];
    }
  }
}

std::pair<float, float> GetRange(const float* A, size_t n) {
  float min_val = A[0];
  float max_val = A[0];

  for (size_t i = 1; i < n; ++i) {
    float x = A[i];

    if (x < min_val) min_val = x;
    if (x > max_val) max_val = x;
  }

  return {min_val, max_val};
}

float GetMaxDiff(const float* A, const float* B, size_t n) {
  float diff = -1.0f;

  for (size_t i = 0; i < n; ++i) {
    float d = A[i] - B[i];
    if (d < 0) d = -d;
    if (d > diff) diff = d;
  }

  return diff;
}

int main(int argc, char* argv[]) {
  size_t K = 0;
  size_t N = 0;

  if (argc == 3) {
    K = std::atoi(argv[1]);
    N = std::atoi(argv[2]);

    std::cout << "K = " << K << std::endl;
    std::cout << "N = " << N << std::endl;

  } else {
    std::cerr << "Usage: " << argv[0] << " k, n" << std::endl;
    return 1;
  }

  constexpr size_t kAlignment = 32;

  float* A = (float*)AlignedMalloc(kAlignment, sizeof(float) * K);
  float* B = (float*)AlignedMalloc(kAlignment, sizeof(float) * K * N);
  float* C0 = (float*)AlignedMalloc(kAlignment, sizeof(float) * N);
  float* C1 = (float*)AlignedMalloc(kAlignment, sizeof(float) * N);

  // init A and B
  for (int i = 0; i < K; ++i) A[i] = (float)i / K;
  for (int i = 0; i < K * N; ++i) B[i] = (float)i / (K * N);

  // test corrections
  RefGemv(C0, A, B, K, N);
  MlasSgemv(C1, A, B, K, N);

  // get range max difference
  auto r0 = GetRange(C0, N);
  auto r1 = GetRange(C1, N);
  auto d = GetMaxDiff(C0, C1, N);

  std::cout << "ref  : " << r0.first << ", " << r0.second << std::endl;
  std::cout << "opt  : " << r1.first << ", " << r1.second << std::endl;
  std::cout << "diff : " << d << std::endl << std::endl;

  // do benchmark
  constexpr int kRepeats = 10;
  float time_v1[kRepeats], time_v2[kRepeats];

  for (int i = 0; i < kRepeats; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    RefGemv(C0, A, B, K, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    MlasSgemv(C1, A, B, K, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    time_v1[i] = std::chrono::duration<float, std::micro>(t1 - t0).count();
    time_v2[i] = std::chrono::duration<float, std::micro>(t2 - t1).count();
  }

  // show result
  std::cout << "Ref version (μs)" << std::endl;
  for (int i = 0; i < kRepeats - 1; ++i) std::cout << time_v1[i] << ", ";
  std::cout << time_v1[kRepeats - 1] << std::endl;

  std::cout << "Opt version μs" << std::endl;
  for (int i = 0; i < kRepeats - 1; ++i) std::cout << time_v2[i] << ", ";
  std::cout << time_v2[kRepeats - 1] << std::endl;

  AlignedFree(A);
  AlignedFree(B);
  AlignedFree(C0);
  AlignedFree(C1);

  return 0;
}
