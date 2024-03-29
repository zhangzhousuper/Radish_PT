#pragma once

#include <device_launch_parameters.h>
#include <sys/stat.h>

#include "gBuffer.h"
#include "scene.h"

#define RESERVOIR_SIZE 32
template <typename SampleT> struct Reservoir {
  __host__ __device__ Reservoir() = default;

  static __host__ __device__ float toScalar(glm::vec3 x) {
    return glm::length(x);
  }

  __host__ __device__ void update(const SampleT &newSample, float newWeight,
                                  float rand) {
    weight += newWeight;
    numSamples++;
    if (rand * weight / newWeight) {
      sample = newSample;
    }
  }

  __host__ __device__ void clear() {
    weight = 0.f;
    numSamples = 0;
  }

  __device__ glm::vec3 pHat(const Intersection &intersec,
                            const Material &material) const {
    return sample.Li * material.BSDF(intersec.norm, intersec.wo, sample.wi) *
           Math::satDot(intersec.norm, sample.wi);
  }

  __device__ float W(const Intersection &intersec, const Material &material) {
    return weight / (toScalar(pHat(intersec, material)) *
                     static_cast<float>(numSamples));
  }

  __device__ bool invalid() { return Math::isNanOrInf(weight) || weight < 0.f; }

  __device__ void checkValidity() {
    if (invalid()) {
      // weight = 0.f;
      clear();
    }
  }

  __device__ void merge(const Reservoir &rhs, float rand) {
    weight += rhs.weight;
    numSamples += rhs.numSamples;

    if (rand * weight < rhs.weight) {
      sample = rhs.sample;
    }
  }

  __device__ void clampedMerge(const Reservoir &rhs, int threshold,
                               float rand) {
    int clamp = threshold - numSamples;
    if (rhs.numSamples > clamp) {
      rhs.weight = static_cast<float>(clamp) / rhs.numSamples;
      rhs.numSamples = clamp;
    }
    merge(rhs, rand);
  }

  template <int M> __device__ void preClampedMerge(Reservoir rhs, float r) {
    static_assert(M > 0, "M <= 0");
    if (rhs.numSamples > 0 && rhs.numSamples > (M - 1) * numSamples &&
        numSamples > 0) {
      rhs.weight *= static_cast<float>(M - 1) * numSamples / rhs.numSamples;
      rhs.numSamples = (M - 1) * numSamples;
    }
    merge(rhs, r);
  }

  template <int M> __device__ void postClampedMerge(Reservoir rhs, float r) {
    static_assert(M > 0, "M <= 0");
    int curNumSample = numSamples;
    merge(rhs, r);
    if (curNumSample > 0 && numSamples > 0 && numSamples > M * curNumSample) {
      weight *= static_cast<float>(M) * curNumSample / numSamples;
      numSamples = M * curNumSample;
    }
  }

  SampleT sample = SampleT();
  int numSamples = 0;
  float weight = 0.f;
};

struct LightLiSample {
  glm::vec3 Li;
  glm::vec3 wi;
  float dist;
};

using DirectReservoir = Reservoir<LightLiSample>;

void ReSTIRInit();
void ReSTIRFree();

void ReSTIRDirect(glm::vec3 *directIllum, int iter, const GBuffer &gBuffer);