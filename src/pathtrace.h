#pragma once

#include <device_launch_parameters.h>

#include "common.h"
#include "glm/fwd.hpp"
#include "scene.h"
#include <vector>

#define RESERVOIR_SIZE 16

__host__ __device__ static bool operator<(float x, glm::vec3 lum) {
  return x * x < glm::dot(lum, lum);
}

__host__ __device__ inline bool operator<(glm::vec3 x, glm::vec3 y) {
  return glm::dot(x, x) < glm::dot(y, y);
}

template <typename SampleT> struct Reservoir {
  __host__ __device__ Reservoir()
      : sample({}), sumWeight(1e-6f), reservoirWeight(0.f){};

  __host__ __device__ void update(const SampleT &sampled,
                                  const glm::vec3 &weight, float rand) {
    sumWeight += weight;
    numSamples++;
    if (rand < weight / sumWeight) {
      sample = sampled;
    }
  }

  __host__ __device__ void clear() {
    sumWeight = glm::vec3(1e-6f);
    reservoirWeight = glm::vec3(0.f);
    numSamples = 0;
  }

  __device__ glm::vec3 directPHat(const Intersection &intersec,
                                  const Material &material) const {
    return sample.Li * material.BSDF(intersec.norm, intersec.wo, sample.wi) *
           Math::satDot(intersec.norm, sample.wi);
  }

  __device__ void calcReservoirWeight(const Intersection &intersec,
                                      const Material &material) {
    reservoirWeight = sumWeight / (directPHat(intersec, material) *
                                   static_cast<float>(numSamples));
  }

  __device__ void merge(const Reservoir &rhs, const Intersection &intersec,
                        const Material &material, float rand) {
    glm::vec3 weight = directPHat(intersec, material) * reservoirWeight *
                       static_cast<float>(numSamples);
    glm::vec3 rhsWeight = rhs.directPHat(intersec, material) *
                          rhs.reservoirWeight *
                          static_cast<float>(rhs.numSamples);
    sumWeight = weight + rhsWeight;

    if (rand * rhsWeight < sumWeight) {
      sample = rhs.sample;
    }
    numSamples += rhs.numSamples;
    calcReservoirWeight(intersec, material);
  }

  SampleT sample = SampleT();
  glm::vec3 sumWeight = glm::vec3(1e-6f);
  glm::vec3 reservoirWeight = glm::vec3(0.f);
  int numSamples = 0;
};

struct LightLiSample {
  glm::vec3 Li;
  glm::vec3 wi;
  float dist;
};

using DirectReservoir = Reservoir<LightLiSample>;

void InitDataContainer(GuiDataContainer *guiData);
void pathTraceInit(Scene *scene);
void pathTraceFree();
// void pathTrace(glm::vec3 *DirectIllum, glm::vec3 *IndirectIllum);
void pathTrace(glm::vec3 *DirectIllum, glm::vec3 *IndirectIllum, int iter);

void ReSTIRDirect(glm::vec3 *DirectOutput, int iter, bool useReservoir);

void copyImageToPBO(uchar4 *devPBO, glm::vec3 *devImage, int width, int height,
                    int toneMapping, float scale = 1.f);
void copyImageToPBO(uchar4 *devPBO, glm::vec2 *devImage, int width, int height);
void copyImageToPBO(uchar4 *devPBO, float *devImage, int width, int height);
void copyImageToPBO(uchar4 *devPBO, int *devImage, int width, int height);
