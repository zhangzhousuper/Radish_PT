#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

#include "common.h"
#include "pathtrace.h"
#include "scene.h"

#include "gbuffer.h"

struct EAWaveletFilter {
  EAWaveletFilter() = default;

  EAWaveletFilter(int width, int height, float sigLumin, float sigNormal,
                  float sigDepth)
      : width(width), height(height), sigLumin(sigLumin), sigNormal(sigNormal),
        sigDepth(sigDepth) {}

  void filter(glm::vec3 *colorOut, glm::vec3 *colorIn, const GBuffer &gBuffer,
              const Camera &cam, int level);
  void filter(glm::vec3 *colorOut, glm::vec3 *colorIn, float *varianceOut,
              float *varianceIn, float *filteredVar, const GBuffer &gBuffer,
              const Camera &cam, int level);

  float sigLumin;
  float sigNormal;
  float sigDepth;

  int width = 0;
  int height = 0;
};

struct LeveledEAWFilter {
  LeveledEAWFilter() = default;
  void create(int width, int height, int level);
  void destroy();

  void filter(glm::vec3 *&colorOut, glm::vec3 *colorIn, const GBuffer &gBuffer,
              const Camera &cam);

  EAWaveletFilter waveletFilter;
  int level = 0;
  glm::vec3 *tmpImg = nullptr;
};

struct SpatioTemporalFilter {
  SpatioTemporalFilter() = default;
  void create(int width, int height, int level);
  void destroy();

  void temporalAccumulate(glm::vec3 *colorIn, const GBuffer &gBuffer);
  void estimateVariance();
  void filterVariance();

  void filter(glm::vec3 *&colorOut, glm::vec3 *colorIn, const GBuffer &gBuffer,
              const Camera &cam);

  void nextFrame();

  EAWaveletFilter waveletFilter;
  int level = 0;

  glm::vec3 *accumColor[2] = {nullptr};
  glm::vec3 *accumMoment[2] = {nullptr};
  float *variance = nullptr;
  bool firstTime = true;

  glm::vec3 *tmpColor = nullptr;
  float *tmpVar = nullptr;
  float *filteredVar = nullptr;
  int frameIdx = 0;
};

void modulateAlbedo(glm::vec3 *devImage, const GBuffer &gBuffer);
void addImage(glm::vec3 *devImage, glm::vec3 *in, int width, int height);
void addImage(glm::vec3 *out, glm::vec3 *in1, glm::vec3 *in2, int width,
              int height);