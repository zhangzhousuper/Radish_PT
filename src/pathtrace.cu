#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

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

#define BVH_DEBUG_VISUALIZATION false

int ToneMapping::method = ToneMapping::ACES;

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter,
                               glm::vec3 *Image, int toneMapping) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);

    // Tonemapping and gamma correction
    glm::vec3 color = Image[index] / (float)iter;

    switch (toneMapping) {
    case ToneMapping::Filmic:
      color = ToneMapping::Filmic(color);
      break;
    case ToneMapping::ACES:
      color = ToneMapping::ACES(color);
      break;
    case ToneMapping::None:
      break;
    }
    color = Math::correctGamma(color);
    glm::ivec3 iColor =
        glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));

    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 0;
    pbo[index].x = intColor.x;
    pbo[index].y = intColor.y;
    pbo[index].z = intColor.z;
  }
}

#define PixelIdxForTerminated -1

static Scene *hst_scene = nullptr;
static GuiDataContainer *guiData = nullptr;
static glm::vec3 *dev_image = nullptr;
static PathSegment *dev_paths = nullptr;
static PathSegment *dev_terminated_paths = nullptr;
static Intersection *dev_intersections = nullptr;
static int *devIntersecMatKeys = nullptr;
static int *devSegmentMatKeys = nullptr;

// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> dev_path_thrust;
static thrust::device_ptr<PathSegment> dev_terminated_paths_thrust;

static thrust::device_ptr<Intersection> devIntersectionsThr;
static thrust::device_ptr<int> devIntersecMatKeysThr;
static thrust::device_ptr<int> devSegmentMatKeysThr;

void InitDataContainer(GuiDataContainer *imGuiData) { guiData = imGuiData; }

void pathTraceInit(Scene *scene) {
  hst_scene = scene;

  const Camera &cam = hst_scene->state.camera;
  const int pixelCount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelCount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelCount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, pixelCount * sizeof(PathSegment));
  cudaMalloc(&dev_terminated_paths, pixelCount * sizeof(PathSegment));
  dev_path_thrust = thrust::device_ptr<PathSegment>(dev_paths);
  dev_terminated_paths_thrust =
      thrust::device_ptr<PathSegment>(dev_terminated_paths);

  cudaMalloc(&dev_intersections, pixelCount * sizeof(Intersection));
  cudaMemset(dev_intersections, 0, pixelCount * sizeof(Intersection));
  devIntersectionsThr = thrust::device_ptr<Intersection>(dev_intersections);

  cudaMalloc(&devIntersecMatKeys, pixelCount * sizeof(int));
  cudaMemset(&devSegmentMatKeys, 0, pixelCount * sizeof(int));
  devIntersecMatKeysThr = thrust::device_ptr<int>(devIntersecMatKeys);
  devSegmentMatKeysThr = thrust::device_ptr<int>(devSegmentMatKeys);

  checkCUDAError("pathTraceInit");
}

void pathTraceFree() {
  cudaFree(dev_image); // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_terminated_paths);
  cudaFree(dev_intersections);

  cudaFree(devIntersecMatKeys);
  cudaFree(devSegmentMatKeys);
  // TODO: clean up any extra device memory you created
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth,
                                      PathSegment *pathSegments) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {
    int index = x + (y * cam.resolution.x);

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    glm::vec4 r = sample4D(rng);

    PathSegment &segment = pathSegments[index];

    // TODO: implement antialiasing by jittering the ray
    float aspect = float(cam.resolution.x) / float(cam.resolution.y);
    float tanFovY = glm::tan(glm::radians(cam.fov.y));
    glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
    glm::vec2 screenPos = glm::vec2(x, y) * pixelSize;
    glm::vec2 ruv = screenPos + pixelSize * glm::vec2(r.x, r.y);
    // it is important to flip the y axis here
    ruv = 1.f - ruv * 2.f;

    glm::vec3 pLens =
        glm::vec3(Math::concentricSampleDisk(r.z, r.w) * cam.lensRadius, 0.f);
    glm::vec3 pFocus =
        glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
    dir = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);
    // the result is in world space by using the camera matrix

    segment.ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
    segment.ray.direction = dir;

    segment.throughput = glm::vec3(1.0f);
    segment.radiance = glm::vec3(0.0f);

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
  }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int num_paths,
                                     PathSegment *pathSegments, DevScene *scene,
                                     Intersection *intersections,
                                     int *materialKeys, bool sortMaterial) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

#if (pathIdx >= num_paths)
  return;

  Intersection intersec;
  PathSegment segment = pathSegments[path_index];
#if BVH_DISABLE
  scene->naiveIntersect(segment.ray, intersec);
#else
  scene->intersect(segment.ray, intersec);
#endif

  if (intersec.primId != NullPrimitive) {
    if (scene->dev_materials[intersec.matId].type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
      if (glm::dot(intersec.norm, segment.ray.direction) > 0) {
        intersec.primId = NullPrimitive;
      } else
#endif
          if (depth != 0) {
        // If not first ray, preserve previous sampling information for
        // MIS calculation
        intersec.prevPos = segment.ray.origin;
        intersec.prev = segment.prev;
      }
    } else {
      intersec.wo = -segment.ray.direction;
    }
    if (sortMaterial) {
      materialKeys[pathIdx] = intersec.matId;
    }
  } else if (sortMaterial) {
    materialKeys[pathIdx] = -1;
  }
  intersections[pathIdx] = intersec;
}

__global__ void computeTerminatedRays(int depth, PathSegment *segments,
                                      Intersection *intersections,
                                      DevScene *scene, int num_paths) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_paths) {
    return;
  }
}

__global__ void pathIntegSampleSurface(int iter, int depth,
                                       PathSegment *segments,
                                       Intersection *intersections,
                                       DevScene *scene, int num_paths) {
  const int SamplesConsumedOneIter = 10;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_paths) {
    return;
  }

  Intersection intersec = intersections[idx];
  if (intersec.primId == NullPrimitive) {
    // TODO
    // Environment map

    if (Math::luminance(segments[idx].radiance) < 1e-4f) {
      segments[idx].pixelIndex = PixelIdxForTerminated;
    } else {
      segments[idx].remainingBounces = 0;
    }
    return;
  }

#if BVH_DEBUG_VISUALIZATION
  float logDepth = 0.f;
  int size = scene->BVHSize;
  while (size) {
    logDepth += 1.f;
    size >>= 1;
  }
  segment.radiance = glm::vec3(float(intersec.primId) / logDepth * 0.1f);

  segment.remainingBounces = 0;
  return;
#endif

  PathSegment &segment = segments[idx];
  thrust::default_random_engine rng =
      makeSeededRandomEngine(iter, idx, 4 + depth * SamplesConsumedOneIter);

  Material material = scene->dev_materials[intersec.matId];
  glm::vec3 accRadiance(0.f);

  if (material.type == Material::Type::Light) {
    PrevBSDFSampleInfo prev = intersec.prev;

    glm::vec3 radiance = material.baseColor * material.emittance;
    if (depth == 0) {
      accRadiance += radiance;
    } else if (prev.deltaSample) {
      accRadiance += radiance * segment.throughput;
    } else {
      float lightPdf = Math::pdfAreaToSolidAngle(
          Math::luminance(radiance) * scene->sumLightPowerInv, intersec.prevPos,
          intersec.pos, intersec.norm);
      float BSDFPdf = prev.BSDFPdf;

      accRadiance += radiance * segment.throughput *
                     Math::powerHeuristic(BSDFPdf, lightPdf);
    }
    segment.remainingBounces = 0;
  } else {
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
        float BSDFPdf = material.pdf(intersec.norm, intersect.wo, wi);
        accRadiance += segment.throughput *
                       material.BSDF(intersec.norm, intersec.wo, wi) *
                       radiance * Math::satDot(intersec.norm, wi) / lightPdf *
                       Math::powerHeuristic(lightPdf, BSDFPdf);
      }
    }

    BSDFSample sample;
    material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);

    if (sample.type == BSDFSampleType::Invalid) {
      // Terminate path if sampling fails
      segment.remainingBounces = 0;
    } else {
      bool deltaSample = (sample.type & BSDFSampleType::Specular);
      segment.throughput *=
          sample.bsdf / sample.pdf *
          (deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));
      segment.ray = makeOffsetedRay(intersec.position, sample.dir);
      segment.prev = {sample.pdf, deltaSample};

      segment.remainingBounces--;
    }
  }
  segment.radiance += accRadiance;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image,
                            PathSegment *iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths) {
    PathSegment iterationPath = iterationPaths[index];
    if (iterationPath.pixelIndex >= 0 && iterationPath.remainingBounces <= 0) {
      glm::vec3 r = iterationPath.radiance;
      if (isnan(r.x) || isnan(r.y) || isnan(r.z) || isinf(r.x) || isinf(r.y) ||
          isinf(r.z)) {
        return
      }
      image[iterationPath.pixelIndex] +=
          glm::clamp(r, glm::vec3(0.f), glm::vec3(FLT_MAX / 10.0f));
    }
  }
}

struct CompactTerminatedPaths {
  __host__ __device__ bool operator()(const PathSegment &path) {
    return !(path.pixelIndex >= 0 && path.remainingBounces <= 0);
  }
};

struct RemoveInvalidPaths {
  __host__ __device__ bool operator()(const PathSegment &path) {
    return path.pixelIndex < 0 || path.remainingBounces <= 0;
  }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a
 * ton of memory management
 */
void pathTrace(uchar4 *pbo, int frame, int iter) {
  const int traceDepth = hst_scene->state.traceDepth;
  const Camera &cam = hst_scene->state.camera;
  const int pixelCount = cam.resolution.x * cam.resolution.y;

  // 2D block for generating ray from camera
  const dim3 blockSize2D(8, 8);
  const dim3 blocksPerGrid2D(
      (cam.resolution.x + blockSize2D.x - 1) / blockSize2D.x,
      (cam.resolution.y + blockSize2D.y - 1) / blockSize2D.y);

  // 1D block for path tracing
  const int blockSize1D = 128;

  // TODO: perform one iteration of path tracing

  generateRayFromCamera<<<blocksPerGrid2D, blockSize2D>>>(cam, iter, traceDepth,
                                                          dev_paths);
  checkCUDAError("PT::generateRayFromCamera");
  cudaDeviceSynchronize();

  int depth = 0;
  int num_paths = pixelCount;

  auto dev_terminated_thrust = dev_terminated_paths_thrust;

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
  while (!iterationComplete) {
    // clean shading chunks
    cudaMemset(dev_intersections, 0, pixelCount * sizeof(Intersection));

    // tracing
    dim3 numBlocksPathSegmentTracing =
        (num_paths + blockSize1D - 1) / blockSize1D;
    computeIntersections<<<numBlocksPathSegmentTracing, blockSize1D>>>(
        depth, num_paths, dev_paths, hstScene->devScene, dev_intersections,
        devIntersecMatKeys, Settings::sortMaterial);
    checkCUDAError("PT::computeInteractions");
    cudaDeviceSynchronize();

    if (Settings::sortMaterial) {
      cudaMemcpyDevToHost(devSegmentMatKeys, devIntersecMatKeys,
                          num_paths * sizeof(int));
      thrust::sort_by_key(devIntersecMatKeysThr,
                          devIntersecMatKeysThr + num_paths,
                          devIntersectionsThr);
      thrust::sort_by_key(devSegmentMatKeysThr,
                          devSegmentMatKeysThr + num_paths, dev_pathsThr);
    }

    pathIntegSampleSurface<<<numBlocksPathSegmentTracing, blockSize1D>>>(
        iter, depth, dev_paths, dev_intersections, hstScene->devScene,
        num_paths);
    checkCUDAError("PT::sampleSurface");
    cudaDeviceSynchronize();
    // Compact paths that are terminated but carry contribution into a
    // separate buffer
    dev_terminated_thrust =
        thrust::remove_copy_if(dev_path_thrust, dev_path_thrust + num_paths,
                               dev_terminated_thrust, CompactTerminatedPaths());
    // Only keep active paths
    auto end = thrust::remove_if(dev_path_thrust, dev_path_thrust + num_paths,
                                 RemoveInvalidPaths());
    num_paths = end - dev_path_thrust;
    // std::cout << "Remaining paths: " << numPaths << "\n";
    iterationComplete = num_paths == 0;
    depth++;

    if (guiData != nullptr) {
      guiData->TracedDepth = depth;
    }
  }

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelCount + blockSize1D - 1) / blockSize1D;

  int numContributing = dev_terminated_thrust.get() - dev_terminated_paths;
  finalGather<<<numBlocksPixels, blockSize1D>>>(numContributing, dev_image,
                                                dev_terminated_paths);

  ///////////////////////////////////////////////////////////////////////////

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2D, blockSize2D>>>(
      pbo, cam.resolution, iter, dev_image, Settings::toneMapping);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathTrace");
}
