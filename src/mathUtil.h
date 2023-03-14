#pragma once

#include "glm/ext/vector_float2.hpp"
#include "glm/fwd.hpp"
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define INV_PI 1.f / PI
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON 0.00001f

namespace Math {
bool epsilonCheck(float a, float b);

glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation,
                                    glm::vec3 scale);

__host__ __device__ inline float pow5(float x) {
  float x2 = x * x;
  return x2 * x2 * x;
}

__host__ __device__ inline float square(float x) { return x * x; }

template <typename T> __host__ __device__ inline T calcFilmic(T c) {
  return (c * (c * 0.22f + 0.03f) + 0.002f) / (c * (c * 0.22f + 0.3f) + 0.06f) -
         1.f / 30.f;
}
__host__ __device__ inline glm::vec3 filmic(glm::vec3 c) {
  return calcFilmic(c * 1.6f) / calcFilmic(11.2f);
}

__host__ __device__ inline float satDot(glm::vec3 a, glm::vec3 b) {
  return glm::max(glm::dot(a, b), 0.f);
}

__host__ __device__ inline float absDot(glm::vec3 a, glm::vec3 b) {
  return glm::abs(glm::dot(a, b));
}

__host__ __device__ inline glm::vec3 ACES(glm::vec3 color) {
  return glm::clamp((color * (2.51f * color + 0.03f)) /
                        (color * (2.43f * color + 0.59f) + 0.14f),
                    0.f, 1.f);
  // need clamp?
}

__host__ __device__ inline glm::vec3 gammaCorrection(glm::vec3 color) {
  return glm::pow(color, glm::vec3(1.f / 2.2f));
}

__device__ static glm::vec2 concentricSampleDisk(float x, float y) {
  float r = glm::sqrt(x);
  float theta = TWO_PI * y;
  return glm::vec2(r * glm::cos(theta), r * glm::sin(theta));
}

__device__ static glm::mat3 localRefMatrix(glm::vec3 n) {
  glm::vec3 t = (glm::abs(n.y) > 0.9999f) ? glm::vec3(0.f, 0.f, 1.f)
                                          : glm::vec3(0.f, 1.f, 0.f);
  glm::vec3 b = glm::normalize(glm::cross(n, t));
  t = glm::cross(b, n);
  return glm::mat3(t, b, n);
}
__device__ static glm::vec3 localToWorld(glm::vec3 n, glm::vec3 v) {
  return glm::normalize(localRefMatrix(n) * v);
}

__device__ static glm::vec3 cosineSampleHemisphere(glm::vec3 n, float rx,
                                                   float ry) {
  glm::vec2 d = concentricSampleDisk(rx, ry);
  float z = glm::sqrt(1.f - glm::dot(d, d));
  return localToWorld(n, glm::vec3(d, z));
}

__device__ static bool refract(glm::vec3 n, glm::vec3 wi, float ior,
                               glm::vec3 &wt) {
  float conIn = glm::dot(n, wi);
  if (conIn < 0.f) {
    ior = 1.f / ior;
  }
  float sin2In = glm::max(0.f, 1.f - conIn * conIn);
  float sin2Tr = sin2In / (ior * ior);

  if (sin2Tr >= 1.f) {
    return false;
  }

  float cosTr = glm::sqrt(1.f - sin2Tr);
  if (conIn < 0.f) {
    cosTr = -cosTr;
  }
  wt = glm::normalize(-wi / ior + n * (conIn / ior - cosTr));
  return true;
}

__device__ inline float areaPdfToSolidAngle(float pdf, glm::vec3 ref,
                                            glm::vec3 y, glm::vec3 ny) {
  glm::vec3 yToRef = ref - y;
  float dist2 = glm::dot(yToRef, yToRef);

  return pdf * absDot(ny, glm::normalize(yToRef)) / dist2;
}
/**
 * Handy-dandy hash function that
 * provides seeds for random number
 * generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}
} // namespace Math