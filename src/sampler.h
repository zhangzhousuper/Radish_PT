#pragma once

#include "mathUtil.h"
#include <thrust/random.h>

#define SAMPLER_USE_SOBOL false

#if SAMPLER_USE_SOBOL
#else
using Sampler = thrust::default_random_engine;
#endif

__host__ __device__ static thrust::default_random_engine
makeSeededRandomEngine(int iter, int index, int depth) {
  int h =
      Math::utilhash((1 << 31) | (depth << 22) | iter) ^ Math::utilhash(index);
  return thrust::default_random_engine(h);
}

__device__ inline float sample1D(thrust::default_random_engine &rng) {
  return thrust::uniform_real_distribution<float>(0.f, 1.f)(rng);
}

__device__ inline glm::vec2 sample2D(Sampler &sampler) {
  return glm::vec2(sample1D(sampler), sample1D(sampler));
}

__device__ inline glm::vec3 sample3D(Sampler &sampler) {
  return glm::vec3(sample2D(sampler), sample1D(sampler));
}

__device__ inline glm::vec4 sample4D(Sampler &sampler) {
  return glm::vec4(sample3D(sampler), sample1D(sampler));
}