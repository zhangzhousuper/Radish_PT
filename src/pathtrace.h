#pragma once

#include <device_launch_parameters.h>

#include "common.h"
#include "glm/fwd.hpp"
#include "scene.h"
#include <vector>

__host__ __device__ static bool operator<(float x, glm::vec3 lum) {
  return x * x < glm::dot(lum, lum);
}

__host__ __device__ inline bool operator<(glm::vec3 x, glm::vec3 y) {
  return glm::dot(x, x) < glm::dot(y, y);
}

void InitDataContainer(GuiDataContainer *guiData);
void pathTraceInit();
void pathTraceFree();
// void pathTrace(glm::vec3 *DirectIllum, glm::vec3 *IndirectIllum);
void pathTrace(glm::vec3 *directIllum, glm::vec3 *indirectIllum, int iter);
void pathTraceDirect(glm::vec3 *directIllum, int iter);

void copyImageToPBO(uchar4 *devPBO, glm::vec3 *devImage, int width, int height,
                    int toneMapping, float scale = 1.f);
void copyImageToPBO(uchar4 *devPBO, glm::vec2 *devImage, int width, int height);
void copyImageToPBO(uchar4 *devPBO, float *devImage, int width, int height);
void copyImageToPBO(uchar4 *devPBO, int *devImage, int width, int height);
