#pragma once

#include <cstddef>

void MlasSgemv(float* C, const float* A, const float* B, size_t K, size_t N);
