#include "scene.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "glm/gtx/dual_quaternion.hpp"
#include "glm/trigonometric.hpp"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <iterator>
#include <map>
#include <stack>
#include <string>
#include <vcruntime.h>
#include <vector>

std::map<std::string, int> MaterialTypeTokenMap = {
    {"Lambertian", Material::Type::Lambertian},
    {"Metallic", Material::Type::Metallic},
    {"Dielectric", Material::Type::Dielectric},
    {"Light", Material::Type::Light},
};

std::map<std::string, MeshData *> Resource::meshDataPool;
std::map<std::string, Image *> Resource::texturePool;

MeshData *Resource::loadOBJMesh(const std::string &filename) {
  auto find = meshDataPool.find(filename);
  if (find != meshDataPool.end()) {
    return find->second;
  }

  auto model = new MeshData;

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::string warn;
  std::string err;

  std::cout << "\t\t[Model loading " << filename << " ...]" << std::endl;
  if (!tinyobj::LoadObj(&attrib, &shapes, nullptr, &warn, &err,
                        filename.c_str())) {
    std::cout << "\t\t\t[Fail Error msg [" << err << "]" << std::endl;
    return nullptr;
  }
  bool hasTexcoord = !attrib.texcoords.empty();

#if MESH_DATA_INDEXED
  model->vertices.resize(attrib.vertices.size() / 3);
  model->normals.resize(attrib.normals.size() / 3);
  memcpy(model->vertices.data(), attrib.vertices.data(),
         attrib.vertices.size() * sizeof(float));
  memcpy(model->normals.data(), attrib.normals.data(),
         attrib.normals.size() * sizeof(float));
  if (hasTexcoord) {
    model->texcoords.resize(attrib.texcoords.size() / 2);
    memcpy(model->texcoords.data(), attrib.texcoords.data(),
           attrib.texcoords.size() * sizeof(float));
  } else {
    model->texcoords.resize(attrib.vertices.size() / 3);
    for (const auto &shape : shapes) {
      for (const auto &index : shape.mesh.indices) {
        model->indices.push_back(glm::ivec3(
            index.vertex_index, index.normal_index,
            hasTexcoord ? index.texcoord_index : index.vertex_index));
      }
    }
  }
#else
  for (const auto &shape : shapes) {
    for (const auto &index : shape.mesh.indices) {
      model->vertices.push_back(
          *((glm::vec3 *)attrib.vertices.data() + index.vertex_index));
      model->normals.push_back(
          *((glm::vec3 *)attrib.normals.data() + index.normal_index));

      model->texcoords.push_back(
          hasTexcoord
              ? *((glm::vec2 *)attrib.texcoords.data() + index.texcoord_index)
              : glm::vec2(0.0f));
    }
  }

#endif
  std::cout << "\t\t[Vertex count = " << model->vertices.size() << "]"
            << std::endl;
  meshDataPool[filename] = model;
  return model;
}

MeshData *Resource::loadGLTFMesh(const std::string &filename) {
  return nullptr;
}

MeshData *Resource::loadModelMeshData(const std::string &filename) {
  if (filename.find(".obj") != filename.npos) {
    return loadOBJMesh(filename);
  } else {
    return loadGLTFMesh(filename);
  }
}

Image *Resource::loadTexture(const std::string &filename) {
  auto find = texturePool.find(filename);
  if (find != texturePool.end()) {
    return find->second;
  }

  auto image = new Image(filename);
  texturePool[filename] = image;
  return image;
}

void Resource::clear() {
  for (auto &pair : meshDataPool) {
    delete pair.second;
  }
  meshDataPool.clear();
  for (auto &pair : texturePool) {
    delete pair.second;
  }
  texturePool.clear();
}

Scene::Scene(const std::string &filename) {
  std::cout << "[Reading scene from " << filename << " ]..." << std::endl;
  std::cout << " " << std::endl;
  char *fname = (char *)filename.c_str();

  fp_in.open(fname);
  if (!fp_in.is_open()) {
    cout << "Error reading from file - aborting!" << endl;
    throw;
  }
  while (fp_in.good()) {
    string line;
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty()) {
      vector<string> tokens = utilityCore::tokenizeString(line);
      if (tokens[0] == "MATERIAL") {
        loadMaterial(tokens[1]);
        cout << " " << endl;
      } else if (tokens[0] == "OBJECT") {
        loadModel(tokens[1]);
        cout << " " << endl;
      } else if (tokens[0] == "CAMERA") {
        loadCamera();
        cout << " " << endl;
      }
    }
  }
}

Scene::~Scene() {}

void Scene::createLightSampler() {
  lightSampler = DiscreteSampler<float>(lightPower);
  std::cout << "[Light sampler size = " << lightPower.size() << "]"
            << std::endl;
}

void Scene::buildDevData() {
#if MESH_DATA_INDEXED
#else
  int primId = 0;
  for (const auto &instance : modelInstances) {
    const auto &material = materials[instance.materialId];
    glm::vec3 radianceUnitArea = material.baseColor * material.emittance;
    float powerUnitArea = Math::luminance(radianceUnitArea);

    for (size_t i = 0; i < instance.meshData->vertices.size(); i++) {
      meshData.vertices.push_back(
          glm::vec3(instance.transform *
                    glm::vec4(instance.meshData->vertices[i], 1.0f)));
      meshData.normals.push_back(glm::normalize(instance.normalMatrix *
                                                instance.meshData->normals[i]));
      meshData.texcoords.push_back(instance.meshData->texcoords[i]);
      if (i % 3 == 0) {
        materialIds.push_back(instance.materialId);
      } else if (i % 3 == 2) {
        if (material.type == Material::Light) {
          glm::vec3 v0 = meshData.vertices[i - 2];
          glm::vec3 v1 = meshData.vertices[i - 1];
          glm::vec3 v2 = meshData.vertices[i];
          float area = Math::triangleArea(v0, v1, v2);
          float power = powerUnitArea * area;

          lightPower.push_back(power);
          lightPrimIds.push_back(primId);
          lightUnitRadiance.push_back(radianceUnitArea);
          sumLightPower += power;
          numLightPrims++;
        }
        primId++;
      }
    }
  }
#endif
  createLightSampler();
  BVHSize = BVHBuilder::build(meshData.vertices, boundingBoxes, BVHNodes);
  checkCUDAError("BVH Build");
  hstScene.create(*this);
  cudaMalloc(&devScene, sizeof(DevScene));
  cudaMemcpyHostToDev(devScene, &hstScene, sizeof(DevScene));
  checkCUDAError("Dev Scene");

  meshData.clear();
  boundingBoxes.clear();
  BVHNodes.clear();

  lightPrimIds.clear();
  lightPower.clear();
  lightSampler.binomDistribs.clear();
}

void Scene::clear() {
  hstScene.destroy();
  cudaSafeFree(devScene);
}

void Scene::loadModel(const std::string &objId) {
  cout << "Loading Model ..." << endl;
  ModelInstance instance;

  std::string line;
  utilityCore::safeGetline(fp_in, line);

  std::string filename = line;
  std::cout << "filename: " << filename << std::endl;
  instance.meshData = Resource::loadModelMeshData(filename);

  // link material
  utilityCore::safeGetline(fp_in, line);
  if (!line.empty() && fp_in.good()) {
    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
    if (materialMap.find(tokens[1]) == materialMap.end()) {
      std::cerr << "Material " << tokens[1] << " not found!" << std::endl;
      throw;
    }
    instance.materialId = materialMap[tokens[1]];
    std::cout << "link to material " << tokens[1] << " with id "
              << instance.materialId << std::endl;
  }

  // load transform
  utilityCore::safeGetline(fp_in, line);
  while (!line.empty() && fp_in.good()) {
    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
    if (tokens[0] == "Translate") {
      instance.transform =
          glm::translate(instance.transform,
                         glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]),
                                   std::stof(tokens[3])));
      std::cout << vec3ToString(instance.translation) << "\n";
    } else if (tokens[0] == "Rotate") {
      instance.transform =
          glm::scale(instance.transform,
                     glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]),
                               std::stof(tokens[3])));
    } else if (tokens[0] == "Scale") {
      instance.transform =
          glm::rotate(instance.transform, std::stof(tokens[4]),
                      glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]),
                                std::stof(tokens[3])));
    } else {
      break;
    }
    utilityCore::safeGetline(fp_in, line);
  }

  instance.transform = Math::buildTransformationMatrix(
      instance.translation, instance.rotation, instance.scale);
  instance.inverseTransform = glm::inverse(instance.transform);
  instance.normalMatrix = glm::transpose(instance.inverseTransform);

  std::cout << "Complete loading model" << std::endl;
  modelInstances.push_back(instance);
}

void Scene::loadCamera() {
  cout << "Loading Camera ..." << endl;
  RenderState &state = this->state;
  Camera &camera = state.camera;
  float fovy;

  // load static properties
  for (int i = 0; i < 5; i++) {
    string line;
    utilityCore::safeGetline(fp_in, line);
    vector<string> tokens = utilityCore::tokenizeString(line);
    if (strcmp(tokens[0].c_str(), "Resolution") == 0) {
      camera.resolution.x = std::stoi(tokens[1]);
      camera.resolution.y = std::stoi(tokens[2]);
    } else if (tokens[0] == "FovY") {
      fovy = std::stof(tokens[1]);
    } else if (tokens[0] == "LensRadius") {
      camera.lensRadius = std::stof(tokens[1]);
    } else if (tokens[0] == "FocalDist") {
      camera.focalDist = std::stof(tokens[1]);
    } else if (tokens[0] == "Sample") {
      state.iterations = std::stoi(tokens[1]);
    } else if (tokens[0] == "Depth") {
      Settings::traceDepth = std::stoi(tokens[1]);
    } else if (tokens[0] == "File") {
      state.imageName = tokens[1];
    }
  }

  string line;
  utilityCore::safeGetline(fp_in, line);
  while (!line.empty() && fp_in.good()) {
    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
    if (tokens[0] == "Eye") {
      camera.position = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]),
                                  std::stof(tokens[3]));
    } else if (tokens[0] == "LookAt") {
      camera.lookAt = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]),
                                std::stof(tokens[3]));
    } else if (tokens[0] == "Up") {
      camera.up = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]),
                            std::stof(tokens[3]));
    }

    utilityCore::safeGetline(fp_in, line);
  }

  // calculate fov based on resolution
  float yscaled = tan(fovy * (PI / 180));
  float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
  float fovx = (atan(xscaled) * 180) / PI;
  camera.fov = glm::vec2(fovx, fovy);
  camera.tanFovY = glm::tan(glm::radians(fovy * 0.5f));

  camera.right = glm::normalize(glm::cross(camera.view, camera.up));
  camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                 2 * yscaled / (float)camera.resolution.y);

  camera.view = glm::normalize(camera.lookAt - camera.position);

  // set up render camera stuff
  int arraylen = camera.resolution.x * camera.resolution.y;
  state.image.resize(arraylen);
  std::fill(state.image.begin(), state.image.end(), glm::vec3());

  cout << "Loaded camera!" << endl;
}

void Scene::loadMaterial(const std::string &materialId) {
  std::cout << "Loading Material ..." << std::endl;
  Material material;

  for (int i = 0; i < 6; i++) {
    std::string line;
    utilityCore::safeGetline(fp_in, line);
    auto tokens = utilityCore::tokenizeString(line);
    if (tokens[0] == "Type") {
      material.type = MaterialTypeTokenMap[tokens[1]];
    } else if (tokens[0] == "BaseColor") {
      glm::vec3 baseColor(std::stof(tokens[1]), std::stof(tokens[2]),
                          std::stof(tokens[3]));
      material.baseColor = baseColor;
    } else if (tokens[0] == "Metallic") {
      material.metallic = std::stof(tokens[1]);
    } else if (tokens[0] == "Roughness") {
      material.roughness = std::stof(tokens[1]);
    } else if (tokens[0] == "Ior") {
      material.ior = std::stof(tokens[1]);
    } else if (tokens[0] == "Emittance") {
      material.emittance = std::stof(tokens[1]);
    }
  }
  materialMap[materialId] = materials.size();
  materials.push_back(material);
  std::cout << "Complete loading material" << std::endl;
}

void DevScene::create(const Scene &scene) {
  std::vector<DevTextureObj> textureObjs;

  size_t textureTotalSize = 0;
  for (auto &texture : scene.textures) {
    textureTotalSize += texture->byteSize();
  }
  cudaMalloc(&dev_textures, textureTotalSize);

  size_t textureOffset = 0;
  for (auto texture : scene.textures) {
    cudaMemcpy(dev_textures + textureOffset, texture->data(),
               texture->byteSize(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    textureObjs.push_back({texture, dev_textures + textureOffset});
    textureOffset += texture->byteSize();
  }

  cudaMalloc(&dev_textureObjs, textureObjs.size() * sizeof(DevTextureObj));
  cudaMemcpyHostToDev(dev_textureObjs, textureObjs.data(),
                      textureObjs.size() * sizeof(DevTextureObj));
  checkCUDAError("DevScene::texture");

  cudaMalloc(&dev_materials, byteSizeOf(scene.materials));
  cudaMemcpyHostToDev(dev_materials, scene.materials.data(),
                      byteSizeOf(scene.materials));

  cudaMalloc(&dev_materialIds, byteSizeOf(scene.materials));
  cudaMemcpyHostToDev(dev_materialIds, scene.materials.data(),
                      byteSizeOf(scene.materials));

  cudaMalloc(&dev_vertices, byteSizeOf(scene.meshData.vertices));
  cudaMemcpyHostToDev(dev_vertices, scene.meshData.vertices.data(),
                      byteSizeOf(scene.meshData.vertices));

  cudaMalloc(&dev_normals, byteSizeOf(scene.meshData.normals));
  cudaMemcpyHostToDev(dev_normals, scene.meshData.normals.data(),
                      byteSizeOf(scene.meshData.normals));

  cudaMalloc(&dev_texcoords, byteSizeOf(scene.meshData.texcoords));
  cudaMemcpyHostToDev(dev_texcoords, scene.meshData.texcoords.data(),
                      byteSizeOf(scene.meshData.texcoords));

  cudaMalloc(&dev_aabb, byteSizeOf(scene.boundingBoxes));
  cudaMemcpyHostToDev(dev_aabb, scene.boundingBoxes.data(),
                      byteSizeOf(scene.boundingBoxes));

  for (int i = 0; i < 6; i++) {
    cudaMalloc(&dev_bvh[i], byteSizeOf(scene.BVHNodes[i]));
    cudaMemcpyHostToDev(dev_bvh[i], scene.BVHNodes[i].data(),
                        byteSizeOf(scene.BVHNodes[i]));
  }
  BVHSize = scene.BVHSize;

  cudaMalloc(&devLightPrimIds, byteSizeOf(scene.lightPrimIds));
  cudaMemcpyHostToDev(devLightPrimIds, scene.lightPrimIds.data(),
                      byteSizeOf(scene.lightPrimIds));
  numLightPrims = scene.numLightPrims;

  cudaMalloc(&devLightUnitRadiance, byteSizeOf(scene.lightUnitRadiance));
  cudaMemcpyHostToDev(devLightUnitRadiance, scene.lightUnitRadiance.data(),
                      byteSizeOf(scene.lightUnitRadiance));

  cudaMalloc(&devLightDistrib, byteSizeOf(scene.lightSampler.binomDistribs));
  cudaMemcpyHostToDev(devLightDistrib, scene.lightSampler.binomDistribs.data(),
                      byteSizeOf(scene.lightSampler.binomDistribs));
  sumLightPowerInv = 1.f / scene.sumLightPower;

  checkCUDAError("DevScene::meshData");
}

void DevScene::destroy() {
  cudaFree(dev_textures);
  cudaFree(dev_textureObjs);
  cudaFree(dev_materials);
  cudaFree(dev_materialIds);
  cudaFree(dev_vertices);
  cudaFree(dev_normals);
  cudaFree(dev_texcoords);
  cudaFree(dev_aabb);
  for (int i = 0; i < 6; i++) {
    cudaFree(dev_bvh[i]);
  }

  cudaSafeFree(devLightPrimIds);
  cudaSafeFree(devLightUnitRadiance);
  cudaSafeFree(devLightDistrib);
}