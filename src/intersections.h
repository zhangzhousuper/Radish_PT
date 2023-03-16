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

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ static glm::vec3 getPointOnRay(Ray r, float dist) {
  return r.origin + (dist - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ static glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
  return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ static float
boxIntersectionTest(Geom box, Ray r, glm::vec3 &intersectionPoint,
                    glm::vec3 &normal, bool &outside) {
  Ray q;
  q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
  q.direction = glm::normalize(
      multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

  float tmin = -1e38f;
  float tmax = 1e38f;
  glm::vec3 tmin_n;
  glm::vec3 tmax_n;
  for (int xyz = 0; xyz < 3; ++xyz) {
    float qdxyz = q.direction[xyz];
    /*if (glm::abs(qdxyz) > 0.00001f)*/ {
      float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
      float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
      float ta = glm::min(t1, t2);
      float tb = glm::max(t1, t2);
      glm::vec3 n;
      n[xyz] = t2 < t1 ? +1 : -1;
      if (ta > 0 && ta > tmin) {
        tmin = ta;
        tmin_n = n;
      }
      if (tb < tmax) {
        tmax = tb;
        tmax_n = n;
      }
    }
  }

  if (tmax >= tmin && tmax > 0) {
    outside = true;
    if (tmin <= 0) {
      tmin = tmax;
      tmin_n = tmax_n;
      outside = false;
    }
    intersectionPoint =
        multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
    normal =
        glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
    return glm::length(r.origin - intersectionPoint);
  }
  return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ static float
sphereIntersectionTest(Geom sphere, Ray r, glm::vec3 &intersectionPoint,
                       glm::vec3 &normal, bool &outside) {
  float radius = .5;

  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(
      multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

  Ray rt;
  rt.origin = ro;
  rt.direction = rd;

  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection -
                   (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
  if (radicand < 0) {
    return -1;
  }

  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;

  float dist = 0;
  if (t1 < 0 && t2 < 0) {
    return -1;
  } else if (t1 > 0 && t2 > 0) {
    dist = glm::min(t1, t2);
    // need check glm
    outside = true;
  } else {
    dist = glm::max(t1, t2);
    outside = false;
  }

  glm::vec3 objspaceIntersection = getPointOnRay(rt, dist);

  intersectionPoint =
      multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
  normal = glm::normalize(
      multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
  if (!outside) {
    normal = -normal;
  }

  return glm::length(r.origin - intersectionPoint);
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
    return false;
  }

  glm::vec3 v0ToOri = ori - v0;

  if (det < 0.f) {
    det = -det;
    v0ToOri = -v0ToOri;
  }

  bary.x = glm::dot(v0ToOri, pvec);

  if (bary.x < 0.f || bary.x > det) {
    return false;
  }

  glm::vec3 qvec = glm::cross(v0ToOri, e01);
  bary.y = glm::dot(dir, qvec);

  if (bary.y < 0.f || bary.x + bary.y > det) {
    return false;
  }

  float invDet = 1.f / det;
  bary.x *= invDet;
  dist = glm::dot(e02, qvec) * invDet;
  return dist > 0.f;
}
