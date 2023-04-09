#pragma once

#include <device_launch_parameters.h>

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