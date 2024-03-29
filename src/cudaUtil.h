#pragma once

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define ERRORCHECK 1

#define FILENAME                                                               \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

static void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err) {
    return;
  }

  fprintf(stderr, "CUDA error");
  if (file) {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
  getchar();
#endif
  exit(EXIT_FAILURE);
#endif
}

template <typename T> T *cudaMalloc(size_t numElements) {
  T *devPtr;
  cudaMalloc(&devPtr, sizeof(T) * numElements);
  return devPtr;
}

template <typename T> void cudaSafeFree(T *&ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

template <typename T> size_t byteSizeOf(const std::vector<T> &vec) {
  return vec.size() * sizeof(T);
}

static cudaError_t __stdcall cudaMemcpyHostToDev(void *device, const void *host,
                                                 size_t size) {
  return cudaMemcpy(device, host, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

static cudaError_t __stdcall cudaMemcpyDevToHost(void *host, const void *device,
                                                 size_t size) {
  return cudaMemcpy(host, device, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

static cudaError_t __stdcall cudaMemcpyDevToDev(void *deviceDst,
                                                const void *deviceSrc,
                                                size_t size) {
  return cudaMemcpy(deviceDst, deviceSrc, size,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}