#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <vector_types.h>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "interactions.h"
#include "intersections.h"
#include "material.h"
#include "mathUtil.h"
#include "pathtrace.h"
#include "sampler.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include "restir.h"

void pathTraceInit() { checkCUDAError("pathTraceInit"); }

void pathTraceFree() {}

__global__ void sendImageToPBO(uchar4 *pbo, glm::vec3 *image, int width,
                               int height, int toneMapping, float scale) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }
  int idx = x + (y * width);

  glm::vec3 color = image[idx] * scale;

  switch (toneMapping) {
  case ToneMapping::Filmic:
    color = Math::filmic(color);
    break;
  case ToneMapping::ACES:
    color = Math::ACES(color);
    break;
  case ToneMapping::None:
    break;
  }
  color = Math::gammaCorrection(color);

  glm::ivec3 icolor =
      glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
  pbo[idx] = make_uchar4(icolor.x, icolor.y, icolor.z, 0);
}

__global__ void sendImageToPBO(uchar4 *pbo, glm::vec2 *image, int width,
                               int height) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }
  int idx = x + (y * width);

  glm::vec3 color = glm::vec3(image[idx], 0.f);

  color = Math::gammaCorrection(color);

  glm::ivec3 icolor =
      glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
  pbo[idx] = make_uchar4(icolor.x, icolor.y, icolor.z, 0);
}

__global__ void sendImageToPBO(uchar4 *pbo, float *image, int width,
                               int height) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }
  int idx = x + (y * width);

  glm::vec3 color = glm::vec3(image[idx]);

  color = Math::gammaCorrection(color);

  glm::ivec3 icolor =
      glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
  pbo[idx] = make_uchar4(icolor.x, icolor.y, icolor.z, 0);
}

__global__ void sendImageToPBO(uchar4 *pbo, int *image, int width, int height) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }
  int idx = x + (y * width);
  int px = image[idx] % width;
  int py = image[idx] / width;

  glm::vec3 color =
      glm::vec3(glm::vec2(px, py) / glm::vec2(width, height), 0.f);

  color = Math::gammaCorrection(color);

  glm::ivec3 icolor =
      glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
  pbo[idx] = make_uchar4(icolor.x, icolor.y, icolor.z, 0);
}

void copyImageToPBO(uchar4 *devPBO, glm::vec3 *devImage, int width, int height,
                    int toneMapping, float scale) {
  const int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
  sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height,
                                          toneMapping, scale);
}
void copyImageToPBO(uchar4 *devPBO, glm::vec2 *devImage, int width,
                    int height) {
  const int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
  sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height);
}
void copyImageToPBO(uchar4 *devPBO, float *devImage, int width, int height) {
  const int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
  sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height);
}

void copyImageToPBO(uchar4 *devPBO, int *devImage, int width, int height) {
  const int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
  sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height);
}

__global__ void singleKernelPT(int looper, int iter, int maxDepth,
                               DevScene *scene, Camera cam,
                               glm::vec3 *directIllum,
                               glm::vec3 *indirectIllum) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= cam.resolution.x || y >= cam.resolution.y) {
    return;
  }
  glm::vec3 direct(0.f);
  glm::vec3 indirect(0.f);

  int index = y * cam.resolution.x + x;
  Sampler rng = makeSeededRandomEngine(looper, index, 0, scene->sampleSequence);

  Ray ray = cam.sample(x, y, sample4D(rng));

  Intersection intersec;
  scene->intersect(ray, intersec);

  if (intersec.primId == NullPrimitive) {
    direct = glm::vec3(1.f);
    goto WriteRadiance;
  }

  Material material = scene->getTexturedMaterialAndSurface(intersec);
#if DENOISER_DEMODULATE
  glm::vec3 albedo = material.baseColor;
  material.baseColor = glm::vec3(1.f);
#endif
  if (material.type == Material::Type::Light) {
    direct = glm::vec3(1.f);
    goto WriteRadiance;
  }

  glm::vec3 throughput(1.f);
  intersec.wo = -ray.direction;

  for (int depth = 1; depth <= maxDepth; depth++) {
    bool deltaBSDF = (material.type == Material::Type::Dielectric);

    if (material.type != Material::Type::Dielectric &&
        glm::dot(intersec.norm, intersec.wo) < 0.f) {
      intersec.norm = -intersec.norm;
    }

    if (!deltaBSDF) {
      glm::vec3 radiance;
      glm::vec3 wi;
      float lightPdf =
          scene->sampleDirectLight(intersec.pos, sample4D(rng), radiance, wi);

      if (lightPdf > 0.f) {
        float BSDFPdf = material.pdf(intersec.norm, intersec.wo, wi);
        (depth == 1 ? direct : indirect) +=
            throughput * material.BSDF(intersec.norm, intersec.wo, wi) *
            radiance * Math::satDot(intersec.norm, wi) / lightPdf *
            Math::powerHeuristic(lightPdf, BSDFPdf);
      }
    }

    BSDFSample sample;
    material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);

    if (sample.type == BSDFSampleType::Invalid) {
      // terminate path if sampling fails
      break;
    } else if (sample.pdf < 1e-8f) {
      break;
    }

    bool deltaSample = (sample.type & BSDFSampleType::Specular);

    throughput *= sample.bsdf / sample.pdf *
                  (deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));

    ray = makeOffsetedRay(intersec.pos, sample.dir);

    glm::vec3 curPos = intersec.pos;
    scene->intersect(ray, intersec);

    intersec.wo = -ray.direction;

    if (intersec.primId == NullPrimitive) {
      if (scene->envMap != nullptr) {
        glm::vec3 radiance =
            scene->envMap->linearSample(Math::toPlane(ray.direction)) *
            throughput;

        float weight =
            deltaSample
                ? 1.f
                : Math::powerHeuristic(sample.pdf,
                                       scene->environmentMapPdf(ray.direction));

        indirect += radiance * weight;
      }
      break;
    }

    material = scene->getTexturedMaterialAndSurface(intersec);

    if (material.type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
      if (glm::dot(intersec.norm, ray.direction) < 0.f) {
        break;
      }
#endif

      glm::vec3 radiance = material.baseColor;

      float weight =
          deltaSample
              ? 1.f
              : Math::powerHeuristic(
                    sample.pdf,
                    Math::pdfAreaToSolidAngle(
                        Math::luminance(radiance) * scene->sumLightPowerInv *
                            scene->getPrimitiveArea(intersec.primId),
                        curPos, intersec.pos, intersec.norm));
      indirect += radiance * throughput * weight;
      break;
    }
  }
WriteRadiance:
#if DENOISER_DEMODULATE
  // direct /= albedo + DEMODULATE_EPS;
  // indirect /= albedo + DEMODULATE_EPS;
#endif

  if (Math::hasNanOrInf(direct)) {
    direct = glm::vec3(0.f);
  }
  if (Math::hasNanOrInf(indirect)) {
    indirect = glm::vec3(0.f);
  }
  direct = Math::HDRToLDR(direct);
  indirect = Math::HDRToLDR(indirect);
  directIllum[index] =
      (directIllum[index] * float(iter) + direct) / float(iter + 1);
  indirectIllum[index] =
      (indirectIllum[index] * float(iter) + indirect) / float(iter + 1);
}

__global__ void PTDirectKernel(int looper, int iter, DevScene *scene,
                               Camera cam, glm::vec3 *directIllum) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam.resolution.x || y >= cam.resolution.y) {
    return;
  }
  glm::vec3 direct(0.f);

  int idx = x + y * cam.resolution.x;
  Sampler rng = makeSeededRandomEngine(looper, idx, 0, scene->sampleSequence);

  Ray ray = cam.sample(x, y, sample4D(rng));
  Intersection intersec;
  scene->intersect(ray, intersec);

  if (intersec.primId == NullPrimitive) {
    if (scene->envMap != nullptr) {
      direct = scene->envMap->linearSample(Math::toPlane(ray.direction));
    }
    goto WriteRadiance;
  }

  Material material = scene->getTexturedMaterialAndSurface(intersec);

  if (material.type == Material::Type::Light) {
    direct = material.baseColor;
    goto WriteRadiance;
  }

  intersec.wo = -ray.direction;

  bool deltaBSDF = (material.type == Material::Type::Dielectric);
  if (!deltaBSDF && glm::dot(intersec.norm, intersec.wo) < 0.f) {
    intersec.norm = -intersec.norm;
  }

  if (!deltaBSDF) {
    glm::vec3 Li;
    glm::vec3 wi;
    float lightPdf =
        scene->sampleDirectLight(intersec.pos, sample4D(rng), Li, wi);

    if (lightPdf > 0.f) {
      direct = Li * material.BSDF(intersec.norm, intersec.wo, wi) *
               Math::satDot(intersec.norm, wi) / lightPdf;
    }
  }

WriteRadiance:
  directIllum[idx] =
      (directIllum[idx] * float(iter) + direct) / float(iter + 1);
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a
 * ton of memory management
 */
void pathTrace(glm::vec3 *directIllum, glm::vec3 *indirectIllum, int iter) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const Camera &cam = State::scene->camera;

  const int BlockSizeSinglePTX = 8;
  const int BlockSizeSinglePTY = 8;

  int blockNumSinglePTX = ceilDiv(cam.resolution.x, BlockSizeSinglePTX);
  int blockNumSinglePTY = ceilDiv(cam.resolution.y, BlockSizeSinglePTY);

  dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
  dim3 singlePTBlockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);
  cudaEventRecord(start, 0);
  singleKernelPT<<<singlePTBlockNum, singlePTBlockSize>>>(
      State::looper, iter, Settings::traceDepth, State::scene->devScene, cam,
      directIllum, indirectIllum);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("PT runtime%.3f ms\n", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  checkCUDAError("pathTrace");
#if SAMPLER_USE_SOBOL
  State::looper = (State::looper + 1) % SobolSampleNum;
#else
  State::looper++;
#endif
}

void pathTraceDirect(glm::vec3 *directIllum, int iter) {
  const Camera &cam = State::scene->camera;

  const int blockSizeSinglePTX = 8;
  const int blockSizeSinglePTY = 8;
  int blockNumSinglePTX = ceilDiv(cam.resolution.x, blockSizeSinglePTX);
  int blockNumSinglePTY = ceilDiv(cam.resolution.y, blockSizeSinglePTY);

  dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
  dim3 singlePTBlockSize(blockSizeSinglePTX, blockSizeSinglePTY);

  PTDirectKernel<<<singlePTBlockNum, singlePTBlockSize>>>(
      State::looper, iter, State::scene->devScene, cam, directIllum);

  checkCUDAError("pathTraceDirect");
#if SAMPLER_USE_SOBOL
  State::looper = (State::looper + 1) % SobolSampleNum;
#else
  State::looper++;
#endif
}