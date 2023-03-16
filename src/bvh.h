#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

#include "mathUtil.h"
#include "sceneStructs.h"

#define NullPrimitive -1
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
       << "pMin = " << vec3ToString(pMin);
    ss << ", pMax = " << vec3ToString(pMax);
    ss << ", center = " << vec3ToString(this->center()) << "]";
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

  __host__ __device__ bool getDistMinMax(float tMin1, float tMin2, float tMax1,
                                         float tMax2, float &tMin) {
    tMin = fminf(tMin1, tMin2);
    float tMax = fmaxf(tMax1, tMax2);
    return tMax >= 0.f && tMax >= tMin;
  }

  __host__ __device__ bool getDistMaxMin(float tMin1, float tMin2, float tMax1,
                                         float tMax2, float &tMin) {
    tMin = fmaxf(tMin1, tMin2);
    float tMax = fminf(tMax1, tMax2);
    return tMax >= 0.f && tMax >= tMin;
  }

  __host__ __device__ bool intersect(Ray ray, float &tMin) {
    const float eps = 1e-6f;
    float tMax;

    glm::vec3 ori = ray.origin;
    glm::vec3 dir = ray.direction;

    if (glm::abs(dir.x) > 1.f - eps) {
      if (Math::between(ori.y, pMin.y, pMax.y) &&
          < Math::between(ori.z, pMin.z, pMax.z)) {
        float dirInvX = 1.f / dir.x;
        float t1 = (pMin.x - ori.x) * dirInvX;
        float t2 = (pMax.x - ori.x) * dirInvX;
        return getDistMinMax(t1, t2, t1, t2, tMin);
      } else {
        return false;
      }
    }

    if (tMax >= 0.f && tMax >= tMin - eps) {
      glm::vec3 mid = ray.getPoint((tMin + tMax) * .5f);
#pragma unroll
      for (int i = 0; i < 3; i++) {
        if (mid[i] <= pMin[i] - eps || mid[i] >= pMax[i] + eps) {
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
