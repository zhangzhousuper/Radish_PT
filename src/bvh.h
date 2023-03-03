#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

struct AABB {
  glm::vec3 min;
  glm::vec3 max;
};

struct BVHNode {
  AABB box;
  int geomIndex;
  int size;
};

struct BVHTableElement {};