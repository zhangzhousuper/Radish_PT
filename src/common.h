#pragma once
#include <iostream>
#include <sys/stat.h>

#define SAMPLER_USE_SOBOL true

#define SCENE_LIGHT_SINGLE_SIDED true

#define CAMERA_PANORAMA false

#define DENOISER_SPLIT_DIRECT_INDIRECT true
#define DENOISER_DEMODULATE true
#define DENOISER_ENCODE_NORMAL true
#define DENOISER_ENCODE_POSITION true

#define DEMODULATE_EPS 1e-3f

#define DENOISE_CLAMP 128.f
#define DENOISE_COMPRESS 16.f
#define DENOISE_LIGHT_ID -2

struct Scene;
struct ToneMapping {
  enum { None = 0, Filmic = 1, ACES = 2 };
};

struct Tracer {
  enum {
    Streamed = 0,
    SingleKernel = 1,
    BVHVisualize = 2,
    GBufferPreview = 3,
    ReSTIRDI = 4,
  };
};

struct Denoiser {
  enum { None, Gaussian, EAWavelet, SVGF };
};
struct Settings {
  static int traceDepth;
  static int toneMapping;
  static int tracer;
  static int ImagePreviewOpt;
  static int denoiser;
  static bool modulate;

  static bool animateCamera;
  static float animateRadius;
  static float animateSpeed;

  static float meshLightSampleWeight;
  static bool useReservoir;
};

struct State {
  static bool camChanged;
  static int looper;
  static Scene *scene;
};