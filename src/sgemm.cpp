#include "sgemm.h"

#include <cstdint>

#if (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#define MLASCALL __stdcall
#else
#define MLASCALL
#endif

extern "C" void MLASCALL MlasSgemmKernelM1Avx(const float *, const float *, float *, size_t, size_t, size_t,
                                              float);

extern "C" alignas(32) const uint32_t MlasMaskMoveAvx[8] = {0, 1, 2, 3, 4, 5, 6, 7};

void MlasSgemv(float *C, const float *A, const float *B, size_t K, size_t N)
{
  size_t ldb = N;

  MlasSgemmKernelM1Avx(A, B, C, K, N, ldb, 0.0f);
}
