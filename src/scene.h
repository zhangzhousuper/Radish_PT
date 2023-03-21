#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "glm/geometric.hpp"
#include "tiny_obj_loader.h"

#include "glm/glm.hpp"

#include "cudaUtil.h"

#include "intersections.h"

#include "bvh.h"
#include "image.h"
#include "material.h"
#include "sampler.h"
#include "sceneStructs.h"
#include "utilities.h"

#include "common.h"

using namespace std;

struct MeshData {
  void clear() {
    vertices.clear();
    normals.clear();
    texcoords.clear();
  }

  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> texcoords;

#if MESH_DATA_INDEXED
  std::vector<glm::ivec3> indices;
#endif
};

struct ModelInstance {
  glm::vec3 translation;
  glm::vec3 rotation;
  glm::vec3 scale;

  glm::mat4 transform;
  glm::mat4 inverseTransform;
  glm::mat3 normalMatrix;

  int materialId;
  MeshData *meshData;
};

class Resource {
public:
  static MeshData *loadOBJMesh(const std::string &filename);
  static MeshData *loadGLTFMesh(const std::string &filename);
  static MeshData *loadModelMeshData(const std::string &filename);
  static Image *loadTexture(const std::string &filename);

  static void clear();

public:
  static std::map<std::string, MeshData *> meshDataPool;
  static std::map<std::string, Image *> texturePool;
};
class Scene;

struct DevScene {
  void create(const Scene &scene);
  void destroy();

  __device__ Material getTexturedMaterialAndSurface(Intersection &intersec) {
    Material mat = devMaterials[intersec.matId];
    if (mat.baseColorMapId > NullTextureId) {
      mat.baseColor =
          devTextureObjs[mat.baseColorMapId].linearSample(intersec.uv);
    }

    if (mat.normalMapId > NullTextureId) {
      glm::vec3 mapped =
          devTextureObjs[mat.normalMapId].linearSample(intersec.uv);
      glm::vec3 localNorm =
          glm::normalize(glm::vec3(mapped.x, mapped.y, mapped.z) * 1.f - 0.5f);
      intersec.norm = Math::localToWorld(intersec.norm, localNorm);
    }
    return mat;
  }

  __device__ int getMTBVHId(glm::vec3 dir) {
    glm::vec3 absDir = glm::abs(dir);
    if (absDir.x > absDir.y) {
      if (absDir.x > absDir.z) {
        return dir.x > 0 ? 0 : 1;
      } else {
        return dir.z > 0 ? 4 : 5;
      }
    } else {
      if (absDir.y > absDir.z) {
        return dir.y > 0 ? 2 : 3;
      } else {
        return dir.z > 0 ? 4 : 5;
      }
    }
  }

  __device__ glm::vec3 getPrimitivePlainNormal(int primId) {
    glm::vec3 va = devVertices[primId * 3];
    glm::vec3 vb = devVertices[primId * 3 + 1];
    glm::vec3 vc = devVertices[primId * 3 + 2];

    return glm::normalize(glm::cross(vb - va, vc - va));
  }

  __device__ void getIntersecGeomInfo(int primId, glm::vec2 bary,
                                      Intersection &intersec) {
    glm::vec3 va = devVertices[primId * 3];
    glm::vec3 vb = devVertices[primId * 3 + 1];
    glm::vec3 vc = devVertices[primId * 3 + 2];

    glm::vec3 na = devNormals[primId * 3];
    glm::vec3 nb = devNormals[primId * 3 + 1];
    glm::vec3 nc = devNormals[primId * 3 + 2];

    glm::vec2 ta = devTexcoords[primId * 3];
    glm::vec2 tb = devTexcoords[primId * 3 + 1];
    glm::vec2 tc = devTexcoords[primId * 3 + 2];

    intersec.pos = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
    intersec.norm = nb * bary.x + nc * bary.y + na * (1.f - bary.x - bary.y);
    intersec.uv = tb * bary.x + tc * bary.y + ta * (1.f - bary.x - bary.y);
  }

  __device__ bool intersectPrim(int primId, Ray ray, float &dist,
                                glm::vec2 &bary) {
    glm::vec3 va = devVertices[primId * 3];
    glm::vec3 vb = devVertices[primId * 3 + 1];
    glm::vec3 vc = devVertices[primId * 3 + 2];

    if (!intersectTriangle(ray, va, vb, vc, bary, dist)) {
      return false;
    }
    glm::vec3 hitPoint =
        vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
    return true;
  }

  __device__ bool intersectPrim(int primId, Ray ray, float distRange) {
    glm::vec3 va = devVertices[primId * 3];
    glm::vec3 vb = devVertices[primId * 3 + 1];
    glm::vec3 vc = devVertices[primId * 3 + 2];
    glm::vec2 bary;
    float dist;
    bool hit = intersectTriangle(ray, va, vb, vc, bary, dist);
    return (hit && dist < distRange);
  }

  __device__ bool intersectPrimDetailed(int primId, Ray ray,
                                        Intersection &intersec) {
    glm::vec3 va = devVertices[primId * 3];
    glm::vec3 vb = devVertices[primId * 3 + 1];
    glm::vec3 vc = devVertices[primId * 3 + 2];
    float dist;
    glm::vec2 bary;

    if (!intersectTriangle(ray, va, vb, vc, bary, dist)) {
      return false;
    }

    glm::vec3 na = devNormals[primId * 3 + 0];
    glm::vec3 nb = devNormals[primId * 3 + 1];
    glm::vec3 nc = devNormals[primId * 3 + 2];

    glm::vec2 ta = devTexcoords[primId * 3 + 0];
    glm::vec2 tb = devTexcoords[primId * 3 + 1];
    glm::vec2 tc = devTexcoords[primId * 3 + 2];

    intersec.pos = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
    intersec.norm = nb * bary.x + nc * bary.y + na * (1.f - bary.x - bary.y);
    intersec.uv = tb * bary.x + tc * bary.y + ta * (1.f - bary.x - bary.y);
    return true;
  }

  __device__ void naiveIntersect(Ray ray, Intersection &intersec) {
    float closestDist = FLT_MAX;
    glm::vec2 closestBary;
    int closestPrimId = NullPrimitive;

    for (int i = 0; i < (BVHSize + 1) / 2; i++) {
      float dist;
      glm::vec2 bary;
      bool hit = intersectPrim(i, ray, dist, bary);

      if (hit && dist < closestDist) {
        closestDist = dist;
        closestBary = bary;
        closestPrimId = i;
      }
    }

    if (closestPrimId != NullPrimitive) {
      intersec.primId = closestPrimId;
      intersec.matId = devMaterialIds[closestPrimId];
      getIntersecGeomInfo(closestPrimId, closestBary, intersec);
    } else {
      intersec.primId = NullPrimitive;
    }
  }

  __device__ bool naiveTestOcclusion(glm::vec3 x, glm::vec3 y) {
    const float eps = 1e-4f;

    glm::vec3 dir = y - x;
    float dist = glm::length(dir);
    dir /= dist;
    dist -= eps;

    Ray ray = makeOffsetedRay(x, dir);

    for (int i = 0; i < (BVHSize + 1) / 2; i++) {
      if (intersectPrim(i, ray, dist)) {
        return true;
      }
    }
    return false;
  }

  __device__ void intersect(Ray ray, Intersection &intersec) {
    int closestPrimId = NullPrimitive;
    glm::vec2 closestBary;
    float closestDist = FLT_MAX;

    MTBVHNode *nodes = devBVHNodes[getMTBVHId(-ray.direction)];
    int node = 0;

    while (node != BVHSize) {
      AABB &bound = devBoundingBoxes[nodes[node].boundingBoxId];
      float boundDist;
      bool boundHit = bound.intersect(ray, boundDist);

      // Only intersect a primitive if its bounding box is hit and
      // that box is closer than previous hit record
      if (boundHit && boundDist < closestDist) {
        int primId = nodes[node].primitiveId;
        if (primId != NullPrimitive) {
          float dist;
          glm::vec2 bary;
          bool hit = intersectPrim(primId, ray, dist, bary);

          if (hit && dist < closestDist) {
            closestPrimId = primId;
            closestDist = dist;
            closestBary = bary;
          }
        }
        node++;
      } else {
        node = nodes[node].nextNodeIfMiss;
      }
    }

    if (closestPrimId != NullPrimitive) {
      getIntersecGeomInfo(closestPrimId, closestBary, intersec);
      intersec.matId = devMaterialIds[closestPrimId];
    }
    intersec.primId = closestPrimId;
  }

  __device__ bool testOcclusion(glm::vec3 x, glm::vec3 y) {
    const float eps = 1e-4f;

    glm::vec3 dir = y - x;
    float dist = glm::length(dir);
    dir /= dist;
    dist -= eps;

    Ray ray = makeOffsetedRay(x, dir);

    MTBVHNode *nodes = devBVHNodes[getMTBVHId(-ray.direction)];
    int node = 0;

    while (node != BVHSize) {
      AABB &bound = devBoundingBoxes[nodes[node].boundingBoxId];
      float boundDist;
      bool boundHit = bound.intersect(ray, boundDist);

      if (boundHit && boundDist < dist) {
        int primId = nodes[node].primitiveId;

        if (primId != NullPrimitive) {
          if (intersectPrim(primId, ray, dist))
            return true;
        }
        node++;
      } else {
        node = nodes[node].nextNodeIfMiss;
      }
    }
    return false;
  }

  __device__ void visualizedIntersect(Ray ray, Intersection &intersec) {
    float closestDist = FLT_MAX;
    int closestPrimId = NullPrimitive;
    glm::vec2 closestBary;

    MTBVHNode *nodes = devBVHNodes[getMTBVHId(-ray.direction)];
    int node = 0;
    int maxDepth = 0;

    while (node != BVHSize) {
      AABB &bound = devBoundingBoxes[nodes[node].boundingBoxId];
      float boundDist;
      bool boundHit = bound.intersect(ray, boundDist);

      // Only intersect a primitive if its bounding box is hit and
      // that box is closer than previous hit record
      if (boundHit && boundDist < closestDist) {
        int primId = nodes[node].primitiveId;
        if (primId != NullPrimitive) {
          float dist;
          glm::vec2 bary;
          bool hit = intersectPrim(primId, ray, dist, bary);

          if (hit && dist < closestDist) {
            closestPrimId = primId;
            closestDist = dist;
            closestBary = bary;
            maxDepth += 1.f;
          }
        }
        node++;
        maxDepth += 1.f;
      } else {
        node = nodes[node].nextNodeIfMiss;
      }
    }
    intersec.primId = maxDepth;
  }
  /**
   * Returns solid angle probability
   */
  __device__ float sampleDirectLight(glm::vec3 pos, glm::vec4 r,
                                     glm::vec3 &radiance, glm::vec3 &wi) {
    int passId = int(float(numLightPrims) * r.x);
    BinomialDistrib<float> distrib = devLightDistrib[passId];
    int lightId = (r.y < distrib.prob) ? passId : distrib.failId;
    int primId = devLightPrimIds[lightId];

    glm::vec3 v0 = devVertices[primId * 3 + 0];
    glm::vec3 v1 = devVertices[primId * 3 + 1];
    glm::vec3 v2 = devVertices[primId * 3 + 2];
    glm::vec3 sampled = Math::sampleTriangleUniform(v0, v1, v2, r.z, r.w);

#if BVH_DISABLE
    bool occ = naiveTestOcclusion(pos, sampled);
#else
    bool occ = testOcclusion(pos, sampled);
#endif
    if (occ) {
      return INVALID_PDF;
    }

    glm::vec3 normal = Math::triangleNormal(v0, v1, v2);
    glm::vec3 posToSampled = sampled - pos;

#if SCENE_LIGHT_SINGLE_SIDED
    if (glm::dot(normal, posToSampled) > 0.f) {
      return INVALID_PDF;
    }
#endif

    radiance = devLightUnitRadiance[lightId];
    wi = glm::normalize(posToSampled);
    return Math::pdfAreaToSolidAngle(
        Math::luminance(radiance) * sumLightPowerInv, pos, sampled, normal);
  }

  glm::vec3 *devVertices = nullptr;
  glm::vec3 *devNormals = nullptr;
  glm::vec2 *devTexcoords = nullptr;
  AABB *devBoundingBoxes = nullptr;
  MTBVHNode *devBVHNodes[6] = {nullptr};
  int BVHSize;

  int *devMaterialIds = nullptr;
  Material *devMaterials = nullptr;
  glm::vec3 *devTextureData = nullptr;
  DevTextureObj *devTextureObjs = nullptr;

  int *devLightPrimIds = nullptr;
  glm::vec3 *devLightUnitRadiance = nullptr;
  BinomialDistrib<float> *devLightDistrib;
  int numLightPrims;
  float sumLightPowerInv;
};

class Scene {
public:
  friend struct DevScene;

  Scene(const std::string &filename);
  ~Scene();

  void buildDevData();
  void clear();

private:
  void createLightSampler();

  void loadModel(const std::string &objectId);
  void loadMaterial(const std::string &materialId);
  void loadCamera();

  int addMaterial(const Material &material);
  int addTexture(const std::string &filename);

public:
  RenderState state;

  std::vector<ModelInstance> modelInstances;

  std::vector<Image *> textures;
  std::map<Image *, int> textureMap;

  std::vector<Material> materials;
  std::map<std::string, int> materialMap;

  std::vector<int> materialIds;
  int BVHSize;
  std::vector<AABB> boundingBoxes;
  std::vector<std::vector<MTBVHNode>> BVHNodes;
  MeshData meshData;

  std::vector<int> lightPrimIds;
  std::vector<float> lightPower;
  std::vector<glm::vec3> lightUnitRadiance;
  DiscreteSampler<float> lightSampler;
  int numLightPrims = 0;
  float sumLightPower = 0.f;

  DevScene hstScene;
  DevScene *devScene = nullptr;

private:
  std::ifstream fpIn;
};