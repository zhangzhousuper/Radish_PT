#pragma once
#include <iostream>
#include <sys/stat.h>

#define MESH_DATA_STRUCT_OF_ARRAY false
#define MESH_DATA_INDEXED false

#define DEV_SCENE_PASS_BY_CONST_MEM false

#define SCENE_LIGHT_SINGLE_SIDED true

#define BVH_DISABLE false

#define ENABLE_GBUFFER false

struct ToneMapping {
  enum { None = 0, Filmic = 1, ACES = 2 };
};

struct Tracer {
  enum { Streamed = 0, SingleKernel = 1, BVHVisualize = 2, GBufferPreview = 3 };
};

struct Settings {
  static int traceDepth;
  static int toneMapping;
  static int tracer;
  static bool sortMaterial;
  static int GBufferPreviewOpt;
};

struct State {
  static bool camChanged;
};