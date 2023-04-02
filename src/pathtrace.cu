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

#define PixelIdxForTerminated -1

static Scene *hstScene = nullptr;
static GuiDataContainer *guiData = nullptr;
static PathSegment *devPaths = nullptr;
static PathSegment *devTerminatedPaths = nullptr;
static Intersection *devIntersections = nullptr;

// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> devPathsThr;
static thrust::device_ptr<PathSegment> devTerminatedPathsThr;

static thrust::device_ptr<Intersection> devIntersectionsThr;

static int looper = 0;

void InitDataContainer(GuiDataContainer *imGuiData) { guiData = imGuiData; }

void pathTraceInit(Scene *scene) {
  hstScene = scene;

  const Camera &cam = hstScene->camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  devPaths = cudaMalloc<PathSegment>(pixelcount);
  cudaMalloc(&devTerminatedPaths, pixelcount * sizeof(PathSegment));
  devPathsThr = thrust::device_ptr<PathSegment>(devPaths);
  devTerminatedPathsThr = thrust::device_ptr<PathSegment>(devTerminatedPaths);

  cudaMalloc(&devIntersections, pixelcount * sizeof(Intersection));
  cudaMemset(devIntersections, 0, pixelcount * sizeof(Intersection));
  devIntersectionsThr = thrust::device_ptr<Intersection>(devIntersections);

  checkCUDAError("pathTraceInit");
}

void pathTraceFree() {
  cudaSafeFree(devPaths);
  cudaSafeFree(devTerminatedPaths);
  cudaSafeFree(devIntersections);

#if ENABLE_GBUFFER
  cudaSafeFree(devGBuffer);
#endif
}

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
                                       scene->enviromentMapPdf(ray.direction));

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

struct CompactTerminatedPaths {
  __host__ __device__ bool operator()(const PathSegment &segment) {
    return !(segment.pixelIndex >= 0 && segment.remainingBounces <= 0);
  }
};

struct RemoveInvalidPaths {
  __host__ __device__ bool operator()(const PathSegment &segment) {
    return segment.pixelIndex < 0 || segment.remainingBounces <= 0;
  }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a
 * ton of memory management
 */
void pathTrace(glm::vec3 *DirectIllum, glm::vec3 *IndirectIllum, int iter) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const Camera &cam = hstScene->camera;

  const int BlockSizeSinglePTX = 8;
  const int BlockSizeSinglePTY = 8;

  int blockNumSinglePTX = ceilDiv(cam.resolution.x, BlockSizeSinglePTX);
  int blockNumSinglePTY = ceilDiv(cam.resolution.y, BlockSizeSinglePTY);

  dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
  dim3 singlePTBlockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);
  cudaEventRecord(start, 0);
  singleKernelPT<<<singlePTBlockNum, singlePTBlockSize>>>(
      looper, iter, Settings::traceDepth, hstScene->devScene, cam, DirectIllum,
      IndirectIllum);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("PT runtime%.3f ms\n", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  checkCUDAError("pathTrace");
  looper = (looper + 1) % SobolSampleNum;
}