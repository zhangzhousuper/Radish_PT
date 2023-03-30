#pragma once
#include <iostream>
#include <sys/stat.h>

#define SAMPLER_USE_SOBOL true

#define SCENE_LIGHT_SINGLE_SIDED true

#define CAMERA_PANORAMA false

#define DENOISER_SPLIT_DIRECT_INDIRECT true
#define DENOISER_DEMODULATE true

#define DEMODULATE_EPS 1e-3f

#define DENOSIE_COMPRESS 1024.f

#define DENOSIE_LIGHT_ID -2

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

  static int ImagePreviewOpt;
};

struct State {
  static bool camChanged;
};