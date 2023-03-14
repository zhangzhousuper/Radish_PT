#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"

#define BVHNodeNonLeaf -1
struct AABB {
  AABB() = default;

  AABB(glm::vec3 pMin, glm::vec3 pMax) : pMin(pMin), pMax(pMax) {}

  AABB(glm::vec3 va, glm::vec3 vb, glm::vec3 vc) {
    pMin = glm::min(glm::min(va, vb), vc);
    pMax = glm::max(glm::max(va, vb), vc);
  }

  AABB(const AABB &a, const AABB &b) {
    pMin = glm::min(a.pMin, b.pMin);
    pMax = glm::max(a.pMax, b.pMax);
  }

  AABB operator()(glm::vec3 p) {
    return {glm::min(pMin, p), glm::max(pMax, p)};
  }

  AABB operator()(const AABB &rhs) {
    return {glm::min(pMin, rhs.pMin), glm::max(pMax, rhs.pMax)};
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "[AABB "
       << "pMin = " << pMin.x << " " << pMin.y << " " << pMin.z;
    ss << ", pMax = " << pMax.x << " " << pMax.y << " " << pMax.z << "]";
    return ss.str();
  }
  __host__ __device__ glm::vec3 center() const { return (pMin + pMax) * 0.5f; }

  __host__ __device__ float surfaceArea() const {
    glm::vec3 size = pMax - pMin;
    return 2.f * (size.x * size.y + size.x * size.z + size.y * size.z);
  }

  __host__ __device__ int longestAxis() const {
    glm::vec3 size = pMax - pMin;
    if (size.x < size.y) {
      return size.y > size.z ? 1 : 2;
    } else {
      return size.x > size.z ? 0 : 2;
    }
  }

  __host__ __device__ bool intersect(Ray ray, float &dist) {
    const float eps = 1e-6f;

    glm::vec3 ori = ray.origin;
    glm::vec3 dir = ray.direction;

    glm::vec3 t1 = (pMin - ori) / dir;
    glm::vec3 t2 = (pMax - ori) / dir;

    glm::vec3 ta = glm::min(t1, t2);
    glm::vec3 tb = glm::max(t1, t2);

    float tMin = -FLT_MAX;
    float tMax = FLT_MAX;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
      if (glm::abs(dir[i]) > eps) {
        if (tb[i] >= 0.f && ta[i] <= tb[i]) {
          tMin = glm::max(tMin, ta[i]);
          tMax = glm::min(tMax, tb[i]);
        }
      }
    }
    dist = tMin;

    if (tMax >= 0.f && tMin <= tMax) {
      glm::vec3 mid = ray.getPoint((tMin + tMax) * 0.5f);
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        if (mid[i] < pMin[i] || mid[i] > pMax[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  glm::vec3 pMin = glm::vec3(FLT_MAX);
  glm::vec3 pMax = glm::vec3(-FLT_MAX);
};

struct MTBVHNode {
  MTBVHNode() = default;
  MTBVHNode(int primId, int boxId, int next)
      : primitiveId(primId), boundingBoxId(boxId), nextNodeIfMiss(next) {}

  int primitiveId;
  int boundingBoxId;
  int nextNodeIfMiss;
};

class BVHBuilder {
private:
  struct NodeInfo {
    bool isLeaf;
    int primIdOrSize;
  };

  struct PrimInfo {
    int primId;
    AABB bound;
    glm::vec3 center;
  };

  struct BuildInfo {
    int offset;
    int start;
    int end;
  };

public:
  static int build(const std::vector<glm::vec3> &vertices,
                   std::vector<AABB> &boundingBoxes,
                   std::vector<std::vector<MTBVHNode>> &BVHNodes);

private:
  static void buildMTBVH(const std::vector<AABB> &boundingBoxes,
                         const std::vector<NodeInfo> &nodeInfo, int BVHSize,
                         std::vector<std::vector<MTBVHNode>> &BVHNodes);
};
