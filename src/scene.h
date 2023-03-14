#pragma once

#include "glm/glm.hpp"
#include "image.h"
#include "material.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"
#include "utilities.h"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

#define INDEXED_MESH_DATA false

struct Model {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> texcoords;

#if INDEXED_MESH_DATA
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
  Model *meshData;
};

class Resource {
public:
  static Model *loadModel(const std::string &filename);
  static Image *loadTexture(const std::string &filename);

  static void clear();

private:
  static std::map<std::string, Model *> modelPool;
  static std::map<std::string, Image *> texturePool;
};

class Scene {
public:
  Scene(const std::string &filename);
  ~Scene();

  void buildDevData();
  void clear();

private:
  void loadModel(const string &objectId);
  void loadMaterial(const string &materialId);
  void loadCamera();

public:
  RenderState state;
  std::vector<Geom> geoms;

  std::vector<Material> materials;
  std::vector<Image *> textures;
  std::vector<ModelInstance> modelInstances;

  std::map<std::string, int> materialMap;

  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec3> texcoords;

#if INDEXED_MESH_DATA
  std::vector<glm::ivec3> indices;
#endif

  glm::vec3 *dev_vertices = nullptr;
  glm::vec3 *dev_normals = nullptr;
  glm::vec3 *dev_texcoords = nullptr;

#if INDEXED_MESH_DATA
  glm::ivec3 *dev_indices = nullptr;
#endif

  AABB *dev_boundingbox = nullptr;
  glm::vec3 *dev_materials = nullptr;
  glm::vec3 *dev_textures = nullptr;

private:
  std::ifstream fp_in;
};