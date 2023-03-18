#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <iomanip>

#include <string>
#include <vector>

#include "bvh.h"
#include "material.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
  SPHERE = 0,
  CUBE = 1,
  MESH = 2,
};

struct Ray {
  __host__ __device__ glm::vec3 getPoint(float dist) {
    return origin + direction * dist;
  }
  glm::vec3 origin;
  glm::vec3 direction;
};

struct Geom {
  GeomType type;
  int materialId;
  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;
  glm::mat4 transform;
  glm::mat4 inverseTransform;
  glm::mat4 invTranspose;
};

struct Camera {
  glm::ivec2 resolution;
  glm::vec3 position;
  glm::vec3 lookAt;
  glm::vec3 view;
  glm::vec3 up;
  glm::vec3 right;
  glm::vec2 fov;
  glm::vec2 pixelLength;
  float lensRadius;
  float focalDist;
  float tanFovY;
};

struct RenderState {
  Camera camera;
  unsigned int iterations;
  int traceDepth;
  std::vector<glm::vec3> image;
  std::string imageName;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct PathSegment {
  Ray ray;
  glm::vec3 throughput;
  glm::vec3 radiance;
  float BSDFPdf;
  bool deltaSample;
  int pixelIndex;
  int remainingBounces;
};

struct Material;
struct Intersection {
  __device__ Intersection() {}

  __device__ Intersection(const Intersection &rhs) { *this = rhs; }

  __device__ void operator=(const Intersection &rhs) {
    primId = rhs.primId;
    matId = rhs.matId;
    pos = rhs.pos;
    norm = rhs.norm;
    uv = rhs.uv;
    wo = rhs.wo;
  }
  int primId;
  int matId;

  glm::vec3 pos;
  glm::vec3 norm;
  glm::vec2 uv;

  union {
    glm::vec3 wo;
    glm::vec3 prevPos;
  };
};
