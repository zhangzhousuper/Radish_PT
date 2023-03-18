#pragma once
#include <iostream>

#define MESH_DATA_STRUCT_OF_ARRAY false
#define MESH_DATA_INDEXED false

#define DEV_SCENE_PASS_BY_CONST_MEM false

#define SCENE_LIGHT_SINGLE_SIDED true

#define BVH_DEBUG_VISUALIZATION false
#define BVH_DISABLE false

struct ToneMapping {
  enum { None = 0, Filmic = 1, ACES = 2 };
};

struct Settings {
  static int toneMapping;
  static bool visualizeBVH;
  static bool sortMaterial;
};