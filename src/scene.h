#pragma once

#include "glm/glm.hpp"
#include "material.h"
#include "sceneStructs.h"
#include "utilities.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

class Scene {
private:
  ifstream fp_in;
  int loadMaterial(string materialId);
  int loadGeom(string objectid);
  int loadCamera();

public:
  Scene(string filename);
  ~Scene();

  std::vector<Geom> geoms;
  std::vector<Material> materials;
  RenderState state;
};
