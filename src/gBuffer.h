#pragma once

#include <device_launch_parameters.h>

#include "scene.h"

#if DENOISER_ENCODE_NORMAL
#define ENCODE_NORM(x) Math::encodeNormalHemiOct32(x)
#define DECODE_NORM(x) Math::decodeNormalHemiOct32(x)
#else
#define ENCODE_NORM(x) x
#define DECODE_NORM(x) x
#endif

struct GBuffer {
#if DENOISER_ENCODE_NORMAL
  using NormT = glm::vec2;
#else
  using NormT = glm::vec3;
#endif

  GBuffer() = default;

  void create(int width, int height);
  void destroy();
  void render(DevScene *scene, const Camera &cam);
  void update(const Camera &cam);

  __device__ NormT *getNormal() const { return normal[frameIdx]; }
  __device__ NormT *lastNormal() const { return normal[frameIdx ^ 1]; }
  __device__ int *getPrimId() const { return primId[frameIdx]; }
  __device__ int *lastPrimId() const { return primId[frameIdx ^ 1]; }

#if DENOISER_ENCODE_POSITION
  __device__ float *getDepth() const { return depth[frameIdx]; }
  __device__ float *lastDepth() const { return depth[frameIdx ^ 1]; }
#else
  __device__ glm::vec3 *getPos() const { return position[frameIdx]; }
  __device__ glm::vec3 *lastPos() const { return position[frameIdx ^ 1]; }
#endif

  glm::vec3 *albedo = nullptr;
  NormT *normal[2] = {nullptr};

  int *motion = nullptr;

#if DENOISER_ENCODE_POSITION
  float *depth[2] = {nullptr};
#else
  glm::vec3 *position[2] = {nullptr};
#endif
  int *primId[2] = {nullptr};
  int frameIdx = 0;

  Camera lastCam;
  int width;
  int height;
};
