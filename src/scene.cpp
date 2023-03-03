#include "scene.h"
#include "glm/trigonometric.hpp"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <map>

std::map<std::string, int> MaterialMap = {
    {"Lambertian", Material::Type::Lambertian},
    {"Metallic", Material::Type::Metallic},
    {"Dielectric", Material::Type::Dielectric},
    {"Light", Material::Type::Light},
};

Scene::Scene(string filename) {
  std::cout << "Reading scene from " << filename << " ..." << std::endl;
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
        loadGeom(tokens[1]);
        cout << " " << endl;
      } else if (tokens[0] == "CAMERA") {
        loadCamera();
        cout << " " << endl;
      }
    }
  }
}

int Scene::loadGeom(string objectid) {
  int id = atoi(objectid.c_str());
  if (id != geoms.size()) {
    cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
    return -1;
  } else {
    cout << "Loading Geom " << id << "..." << endl;
    Geom newGeom;
    string line;

    // load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
      if (line == "sphere") {
        cout << "Creating new sphere..." << endl;
        newGeom.type = GeomType::SPHERE;
      } else if (line == "cube") {
        cout << "Creating new cube..." << endl;
        newGeom.type = GeomType::CUBE;
      }
    }

    // link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
      vector<string> tokens = utilityCore::tokenizeString(line);
      newGeom.materialid = atoi(tokens[1].c_str());
      cout << "Connecting Geom " << objectid << " to Material "
           << newGeom.materialid << "..." << endl;
    }

    // load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
      vector<string> tokens = utilityCore::tokenizeString(line);

      // load tranformations
      if (tokens[0] == "TRANSLATION") {
        newGeom.translation =
            glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                      atof(tokens[3].c_str()));
      } else if (tokens[0] == "ROTATION") {
        newGeom.rotation =
            glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                      atof(tokens[3].c_str()));
      } else if (tokens[0] == "SCALE") {
        newGeom.scale =
            glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()),
                      atof(tokens[3].c_str()));
      }

      utilityCore::safeGetline(fp_in, line);
    }

    newGeom.transform = Math::buildTransformationMatrix(
        newGeom.translation, newGeom.rotation, newGeom.scale);
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    geoms.push_back(newGeom);
    return 1;
  }
}

int Scene::loadCamera() {
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
      state.traceDepth = std::stoi(tokens[1]);
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
  return 1;
}

int Scene::loadMaterial(string materialid) {
  int id = atoi(materialid.c_str());
  if (id != materials.size()) {
    cout << "ERROR: MATERIAL ID does not match expected number of materials"
         << endl;
    return -1;
  } else {
    cout << "Loading Material " << id << "..." << endl;
    Material newMaterial;

    // load static properties
    for (int i = 0; i < 6; i++) {
      string line;
      utilityCore::safeGetline(fp_in, line);
      auto tokens = utilityCore::tokenizeString(line);
      if (tokens[0] == "Type") {
        newMaterial.type = MaterialMap[tokens[1]];
      } else if (tokens[0] == "BaseColor") {
        glm::vec3 baseColor(std::stof(tokens[1]), std::stof(tokens[2]),
                            std::stof(tokens[3]));
        newMaterial.baseColor = baseColor;
      }

      else if (tokens[0] == "Metallic") {
        newMaterial.metallic = std::stof(tokens[1]);
      } else if (tokens[0] == "Roughness") {
        newMaterial.roughness = std::stof(tokens[1]);
      } else if (tokens[0] == "Ior") {
        newMaterial.ior = std::stof(tokens[1]);
      } else if (tokens[0] == "Emittance") {
        newMaterial.emittance = std::stof(tokens[1]);
      }
    }
    materials.push_back(newMaterial);
    return 1;
  }
}
