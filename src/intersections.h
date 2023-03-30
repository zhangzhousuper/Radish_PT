#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"

#include "mathUtil.h"

#include "utilities.h"

__host__ __device__ inline Ray makeRay(glm::vec3 ori, glm::vec3 dir) {
  return {ori, dir};
}

__host__ __device__ inline Ray makeOffsetedRay(glm::vec3 ori, glm::vec3 dir) {
  return {ori + dir * 1e-5f, dir};
}

__host__ __device__ static bool intersectTriangle(Ray ray, glm::vec3 v0,
                                                  glm::vec3 v1, glm::vec3 v2,
                                                  glm::vec2 &bary,
                                                  float &dist) {
  glm::vec3 e01 = v1 - v0;
  glm::vec3 e02 = v2 - v0;
  glm::vec3 ori = ray.origin;
  glm::vec3 dir = ray.direction;
  glm::vec3 pvec = glm::cross(dir, e02);

  float det = glm::dot(e01, pvec);

  if (glm::abs(det) < FLT_EPSILON) {
    // because the ray is parallel to the triangle.
    return false;
  }

  glm::vec3 v0ToOri = ori - v0;

  // the determinant needs to be positive for the barycentric coordinates to be
  // valid.
  if (det < 0.f) {
    det = -det;
    v0ToOri = -v0ToOri;
  }

  bary.x = glm::dot(v0ToOri, pvec);

  if (bary.x < 0.f || bary.x > det) {
    // . If bary.x is outside the range [0, det] (where det is the determinant),
    // the function returns false because the intersection point is outside the
    // triangle.
    return false;
  }

  glm::vec3 qvec = glm::cross(v0ToOri, e01);
  bary.y = glm::dot(dir, qvec);

  if (bary.y < 0.f || bary.x + bary.y > det) {
    return false;
  }

  float invDet = 1.f / det;
  bary *= invDet; //  get the barycentric coordinates.
  dist = glm::dot(e02, qvec) * invDet;
  // If dist is negative, the intersection point is behind the ray origin and
  // the function returns false.
  return dist > 0.f;
}
