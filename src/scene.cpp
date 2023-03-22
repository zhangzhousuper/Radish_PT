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
    {"MetallicWorkflow", Material::Type::MetallicWorkflow},
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
    std::cout << "\t\t\t[Fail Error msg " << err << "]" << std::endl;
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
    model->texcoord.resize(attrib.texcoords.size() / 2);
    memcpy(model->texcoords.data(), attrib.texcoords.data(),
           attrib.texcoords.size() * sizeof(float));
  } else {
    model->texcoord.resize(attrib.vertices.size() / 3);
  }

  for (const auto &shape : shapes) {
    for (auto idx : shape.mesh.indices) {
      model->indices.push_back(
          {idx.vertex_index, idx.normal_index,
           hasTexcoord ? idx.texcoord_index : idx.vertex_index});
    }
  }
#else
  for (const auto &shape : shapes) {
    for (auto idx : shape.mesh.indices) {
      model->vertices.push_back(
          *((glm::vec3 *)attrib.vertices.data() + idx.vertex_index));
      model->normals.push_back(
          *((glm::vec3 *)attrib.normals.data() + idx.normal_index));
      model->texcoords.push_back(
          hasTexcoord
              ? *((glm::vec2 *)attrib.texcoords.data() + idx.texcoord_index)
              : glm::vec2(0.f));
    }
  }
#endif
  std::cout << "\t\t[Vertex count = " << model->vertices.size() << "]"
            << std::endl;
  meshDataPool[filename] = model;
  return model;
}

MeshData *Resource::loadGLTFMesh(const std::string &filename) {
  auto find = meshDataPool.find(filename);
  if (find != meshDataPool.end()) {
    return find->second;
  }
  auto model = new MeshData;
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
  auto texture = new Image(filename);
  texturePool[filename] = texture;
  return texture;
}

void Resource::clear() {
  for (auto i : meshDataPool) {
    delete i.second;
  }
  meshDataPool.clear();

  for (auto i : texturePool) {
    delete i.second;
  }
  texturePool.clear();
}

Scene::Scene(const std::string &filename) {
  std::cout << "[Reading scene from " << filename << " ]..." << std::endl;
  std::cout << " " << std::endl;
  char *fname = (char *)filename.c_str();

  fpIn.open(fname);
  if (!fpIn.is_open()) {
    std::cout << "Error reading from file - aborting!" << std::endl;
    throw;
  }
  while (fpIn.good()) {
    std::string line;
    utilityCore::safeGetline(fpIn, line);
    if (!line.empty()) {
      std::vector<std::string> tokens = utilityCore::tokenizeString(line);
      if (tokens[0] == "Material") {
        loadMaterial(tokens[1]);
        std::cout << " " << std::endl;
      } else if (tokens[0] == "Object") {
        loadModel(tokens[1]);
        std::cout << " " << std::endl;
      } else if (tokens[0] == "Camera") {
        loadCamera();
        std::cout << " " << std::endl;
      } else if (tokens[0] == "EnvMap") {
        if (tokens[1] != "Null") {
          loadEnvMap(tokens[1]);
        }
      }
    }
  }
}

Scene::~Scene() {}

void Scene::createLightSampler() {
  if (envMapTexId != NullTextureId) {
    lightPower.push_back(envMapSampler.sum);
  }
  lightSampler = DiscreteSampler1D<float>(lightPower);
  std::cout << "[Light sampler size = " << lightPower.size() << "]"
            << std::endl;
}

void Scene::buildDevData() {
#if MESH_DATA_INDEXED
#else
  int primId = 0;
  for (const auto &inst : modelInstances) {
    const auto &material = materials[inst.materialId];
    glm::vec3 radianceUnitArea = material.baseColor;
    float powerUnitArea = Math::luminance(radianceUnitArea);

    for (size_t i = 0; i < inst.meshData->vertices.size(); i++) {

      meshData.vertices.push_back(glm::vec3(
          inst.transform * glm::vec4(inst.meshData->vertices[i], 1.0f)));
      meshData.normals.push_back(
          glm::normalize(inst.normalMatrix * inst.meshData->normals[i]));
      meshData.texcoords.push_back(inst.meshData->texcoords[i]);
      if (i % 3 == 0) {
        materialIds.push_back(inst.materialId);
      } else if (i % 3 == 2) {
        if (material.type == Material::Light) {
          glm::vec3 v0 = meshData.vertices[i - 2];
          glm::vec3 v1 = meshData.vertices[i - 1];
          glm::vec3 v2 = meshData.vertices[i];
          float area = Math::triangleArea(v0, v1, v2);
          float power = powerUnitArea * area;

          lightPrimIds.push_back(primId);
          lightUnitRadiance.push_back(radianceUnitArea);
          lightPower.push_back(power);

          numLightPrims++;
        }
        primId++;
      }
    }
  }
#endif
  if (primId == 0) {
    std::cout << "[No mesh data loaded, quit]" << std::endl;
    exit(-1);
  }

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
  utilityCore::safeGetline(fpIn, line);

  std::string filename = line;
  std::cout << "filename: " << filename << std::endl;
  instance.meshData = Resource::loadModelMeshData(filename);
  if (!instance.meshData) {
    std::cout << "\t\t[Fail to load, skipped]" << std::endl;
    while (!line.empty() && fpIn.good()) {
      utilityCore::safeGetline(fpIn, line);
    }
    return;
  }

  // link material
  utilityCore::safeGetline(fpIn, line);
  if (!line.empty() && fpIn.good()) {
    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
    if (tokens[1] == "Null") {
      // Null material, create new one
      instance.materialId = addMaterial(Material());
    } else {
      if (materialMap.find(tokens[1]) == materialMap.end()) {
        std::cout << "Material " << tokens[1] << " not found!" << std::endl;
        throw;
      }
      instance.materialId = materialMap[tokens[1]];
      std::cout << "link to material " << tokens[1] << " with id "
                << instance.materialId << std::endl;
    }
  }

  // load transform
  utilityCore::safeGetline(fpIn, line);
  while (!line.empty() && fpIn.good()) {
    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
    if (tokens[0] == "Translate") {
      instance.translation = glm::vec3(
          std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
    } else if (tokens[0] == "Rotate") {
      instance.rotation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]),
                                    std::stof(tokens[3]));
    } else if (tokens[0] == "Scale") {
      instance.scale = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]),
                                 std::stof(tokens[3]));
    }
    utilityCore::safeGetline(fpIn, line);
  }

  instance.transform = Math::buildTransformationMatrix(
      instance.translation, instance.rotation, instance.scale);

  instance.inverseTransform = glm::inverse(instance.transform);
  instance.normalMatrix = glm::transpose(glm::mat3(instance.inverseTransform));

  std::cout << "Complete loading model" << std::endl;
  modelInstances.push_back(instance);
}

void Scene::loadCamera() {
  cout << "Loading Camera ..." << endl;
  float fovy;

  // load static properties
  for (int i = 0; i < 7; i++) {
    std::string line;
    utilityCore::safeGetline(fpIn, line);
    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
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

  std::string line;
  utilityCore::safeGetline(fpIn, line);
  while (!line.empty() && fpIn.good()) {
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

    utilityCore::safeGetline(fpIn, line);
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

void Scene::loadEnvMap(const std::string &filename) {
  envMapTexId = addTexture(filename);
  auto envMap = textures[envMapTexId];
  std::vector<float> pdf(envMap->width() * envMap->height());

  for (int i = 0; i < envMap->height(); i++) {
    for (int j = 0; j < envMap->width(); j++) {
      int idx = i * envMap->width() + j;
      pdf[idx] = Math::luminance(envMap->data()[idx]) *
                 glm::sin((0.5f + i) / envMap->height() * PI);
    }
  }
  envMapSampler = DiscreteSampler1D<float>(pdf);
  std::cout << "\t[Environment Map width = " << envMap->width()
            << ", height = " << envMap->height()
            << ", sumPower = " << envMapSampler.sum << "]" << std::endl;
}

int Scene::addMaterial(const Material &material) {
  materials.push_back(material);
  return materials.size() - 1;
}

int Scene::addTexture(const std::string &filename) {
  auto texture = Resource::loadTexture(filename);
  auto find = textureMap.find(texture);
  if (find != textureMap.end()) {
    return find->second;
  } else {
    int size = textureMap.size();
    textureMap[texture] = size;
    textures.push_back(texture);
    return size;
  }
}

void Scene::loadMaterial(const std::string &materialId) {
  std::cout << "\t[Material " << materialId << "]" << std::endl;
  Material material;

  for (int i = 0; i < 6; i++) {
    std::string line;
    utilityCore::safeGetline(fpIn, line);
    auto tokens = utilityCore::tokenizeString(line);
    if (tokens[0] == "Type") {
      material.type = MaterialTypeTokenMap[tokens[1]];
      std::cout << "\t\t[Type " << tokens[1] << "]" << std::endl;
    } else if (tokens[0] == "BaseColor") {
      if (tokens.size() > 2) {
        glm::vec3 baseColor(std::stof(tokens[1]), std::stof(tokens[2]),
                            std::stof(tokens[3]));
        material.baseColor = baseColor;
      } else {
        material.baseColorMapId = addTexture(tokens[1]);
        std::cout << "\t\t[BaseColor use texture " << tokens[1] << "]"
                  << std::endl;
        std::cout << "baseColorMapId: " << material.baseColorMapId << std::endl;
      }
    } else if (tokens[0] == "Metallic") {
      material.metallic = std::stof(tokens[1]);
    } else if (tokens[0] == "Roughness") {
      material.roughness = std::stof(tokens[1]);
    } else if (tokens[0] == "Ior") {
      material.ior = std::stof(tokens[1]);
    } else if (tokens[0] == "NormalMap") {
      if (tokens[1] != "Null") {
        material.normalMapId = addTexture(tokens[1]);
        std::cout << "\t\t[NormalMap use texture " << tokens[1] << "]"
                  << std::endl;
      }
    }
  }
  materialMap[materialId] = materials.size();
  materials.push_back(material);
  std::cout << "Complete loading material" << std::endl;
}

void DevScene::create(const Scene &scene) {
  // Put all texture devData in a big buffer
  // and setup device texture objects to manage
  std::vector<DevTextureObj> textureObjs;

  size_t textureTotalSize = 0;
  for (auto texture : scene.textures) {
    textureTotalSize += texture->byteSize();
  }
  cudaMalloc(&textureData, textureTotalSize);
  checkCUDAError("DevScene::texture");

  int textureOffset = 0;
  for (auto texture : scene.textures) {
    cudaMemcpyHostToDev(textureData + textureOffset, texture->data(),
                        texture->byteSize());
    std::cout << texture->byteSize() << "\n";
    checkCUDAError("DevScene::texture::copy");
    textureObjs.push_back({texture, textureData + textureOffset});
    textureOffset += texture->byteSize() / sizeof(glm::vec3);
  }

  cudaMalloc(&textures, textureObjs.size() * sizeof(DevTextureObj));

  checkCUDAError("DevScene::textureObjs::malloc");
  cudaMemcpyHostToDev(textures, textureObjs.data(),
                      textureObjs.size() * sizeof(DevTextureObj));
  checkCUDAError("DevScene::textureObjs::copy");

  cudaMalloc(&materials, byteSizeOf(scene.materials));
  cudaMemcpyHostToDev(materials, scene.materials.data(),
                      byteSizeOf(scene.materials));

  cudaMalloc(&materialIds, byteSizeOf(scene.materialIds));
  cudaMemcpyHostToDev(materialIds, scene.materialIds.data(),
                      byteSizeOf(scene.materialIds));
  checkCUDAError("DevScene::material");

  cudaMalloc(&vertices, byteSizeOf(scene.meshData.vertices));
  cudaMemcpyHostToDev(vertices, scene.meshData.vertices.data(),
                      byteSizeOf(scene.meshData.vertices));

  cudaMalloc(&normals, byteSizeOf(scene.meshData.normals));
  cudaMemcpyHostToDev(normals, scene.meshData.normals.data(),
                      byteSizeOf(scene.meshData.normals));

  cudaMalloc(&texcoords, byteSizeOf(scene.meshData.texcoords));
  cudaMemcpyHostToDev(texcoords, scene.meshData.texcoords.data(),
                      byteSizeOf(scene.meshData.texcoords));

  cudaMalloc(&boundingBoxes, byteSizeOf(scene.boundingBoxes));
  cudaMemcpyHostToDev(boundingBoxes, scene.boundingBoxes.data(),
                      byteSizeOf(scene.boundingBoxes));

  for (int i = 0; i < 6; i++) {
    cudaMalloc(&BVHNodes[i], byteSizeOf(scene.BVHNodes[i]));
    cudaMemcpyHostToDev(BVHNodes[i], scene.BVHNodes[i].data(),
                        byteSizeOf(scene.BVHNodes[i]));
  }
  BVHSize = scene.BVHSize;

  cudaMalloc(&lightPrimIds, byteSizeOf(scene.lightPrimIds));
  cudaMemcpyHostToDev(lightPrimIds, scene.lightPrimIds.data(),
                      byteSizeOf(scene.lightPrimIds));

  cudaMalloc(&lightUnitRadiance, byteSizeOf(scene.lightUnitRadiance));
  cudaMemcpyHostToDev(lightUnitRadiance, scene.lightUnitRadiance.data(),
                      byteSizeOf(scene.lightUnitRadiance));

  lightSampler.create(scene.lightSampler);
  sumLightPowerInv = 1.f / scene.lightSampler.sum;

  if (scene.envMapTexId != NullTextureId) {
    envMap = textures + scene.envMapTexId;
    envMapSampler.create(scene.envMapSampler);
  }
  checkCUDAError("DevScene::meshData");
}

void DevScene::destroy() {
  cudaFree(textureData);
  cudaFree(textures);
  cudaFree(materials);
  cudaFree(materialIds);
  cudaFree(vertices);
  cudaFree(normals);
  cudaFree(texcoords);
  cudaFree(boundingBoxes);

  for (int i = 0; i < 6; i++) {
    cudaFree(BVHNodes[i]);
  }

  cudaSafeFree(lightPrimIds);
  cudaSafeFree(lightUnitRadiance);
  lightSampler.destroy();
  envMapSampler.destroy();
}