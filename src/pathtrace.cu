#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

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

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter,
                               glm::vec3 *Image) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    glm::vec3 color = Image[index] / (float)iter;
    glm::vec3 mapped = Math::ACES(color);
    mapped = color;
    mapped = Math::gammaCorrection(mapped);
    glm::ivec3 intColor = glm::ivec3(glm::clamp(mapped * 255.f, 0.f, 255.f));

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
// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> dev_path_thrust;
static thrust::device_ptr<PathSegment> dev_terminated_paths_thrust;

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

  // TODO: initialize any extra device memeory you need

  checkCUDAError("pathTraceInit");
}

void pathTraceFree() {
  cudaFree(dev_image); // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_terminated_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  cudaFree(dev_intersections);
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
                                     Intersection *intersections) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index < num_paths) {
#if BVH_DEBUG_VISUALIZATION
    scene->visualizedIntersect(pathSegments[path_index].ray,
                               intersections[path_index]);
#else
    scene->intersect(pathSegments[path_index].ray, intersections[path_index]);
#endif
  }
}

__global__ void pathIntegSampleSurface(int iter, PathSegment *segments,
                                       Intersection *intersections,
                                       DevScene *scene, int num_paths) {
  const int SamplesConsumedOneIter = 10;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_paths) {
    return;
  }

  Intersection intersec = intersections[idx];
  if (intersec.primitive == NullPrimitive) {
    // TODO
    // Environment map

    segments[idx].pixelIndex = PixelIdxForTerminated;
    return;
  }

  PathSegment &segment = segments[idx];
  thrust::default_random_engine rng =
      makeSeededRandomEngine(iter, idx, 4 + iter * SamplesConsumedOneIter);
  Material material = scene->dev_materials[intersec.materialId];

  // TODO
  // perform light area sampling and MIS
  bool deltaBSDF = material.type == Material::Type::Dielectric;

  if (!deltaBSDF) {
    glm::vec3 radiance;
    glm::vec3 wi;
    float lightPdf = scene->sampleDirectLight(intersec.position, sample4D(rng),
                                              radiance, wi);
    float BSDFPdf =
        material.pdf(intersec.surfaceNormal, intersec.incomingDir, wi);
    segment.radiance +=
        segment.throughput *
        material.BSDF(intersec.surfaceNormal, intersec.incomingDir, wi) *
        radiance * glm::dot(intersec.surfaceNormal, wi) *
        Math::powerHeuristic(lightPdf, BSDFPdf) / lightPdf;
  }
#if BVH_DEBUG_VISUALIZATION
  float logDepth = 0.f;
  int size = scene->BVHSize;
  while (size) {
    logDepth += 1.f;
    size >>= 1;
  }
  segment.radiance = glm::vec3(float(intersec.primitive) / logDepth * 0.1f);

  segment.remainingBounces = 0;
  return;
#endif

  if (material.type == Material::Type::Light) {
    // TODO
    // MIS

    segment.radiance +=
        segment.throughput * material.emittance * material.baseColor;
    segment.remainingBounces = 0;
  } else {
    if (material.type != Material::Type::Dielectric &&
        glm::dot(intersec.surfaceNormal, intersec.incomingDir) < 0.f) {
      intersec.surfaceNormal = -intersec.surfaceNormal;
    }

    BSDFSample sample;
    material.sample(intersec.surfaceNormal, intersec.incomingDir, material,
                    sample3D(rng), sample);
    if (sample.type == BSDFSampleType::Invalid) {
      // Terminate path if sampling fails
      segment.remainingBounces = 0;
    } else {
      bool isSampleDelta = (sample.type & BSDFSampleType::Specular);
      segment.throughput *=
          sample.bsdf / sample.pdf *
          (isSampleDelta ? 1.f
                         : Math::absDot(intersec.surfaceNormal, sample.dir));
      segment.ray = makeOffsetedRay(intersec.position, sample.dir);
      segment.remainingBounces--;
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *Image,
                            PathSegment *iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < nPaths) {
    PathSegment iterationPath = iterationPaths[index];
    if (iterationPath.pixelIndex >= 0 && iterationPath.remainingBounces == 0) {
      Image[iterationPath.pixelIndex] += iterationPath.radiance;
    }
  }
}

struct CompactTerminatedPaths {
  __host__ __device__ bool operator()(const PathSegment &path) {
    return !(path.pixelIndex >= 0 && path.remainingBounces == 0);
  }
};

struct RemoveInvalidPaths {
  __host__ __device__ bool operator()(const PathSegment &path) {
    return path.pixelIndex < 0 || path.remainingBounces == 0;
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
        depth, num_paths, dev_paths, hstScene->devScene, dev_intersections);
    checkCUDAError("PT::computeInteractions");
    cudaDeviceSynchronize();
    depth++;
    // TODO:
    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays
    // by evaluating the BSDF.
    // Start off with just a big kernel that
    // handles all the different materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and
    // shading path segments that have been reshuffled to be contiguous
    // in memory.

    pathIntegSampleSurface<<<numBlocksPathSegmentTracing, blockSize1D>>>(
        iter, dev_paths, dev_intersections, hstScene->devScene, num_paths);
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
  sendImageToPBO<<<blocksPerGrid2D, blockSize2D>>>(pbo, cam.resolution, iter,
                                                   dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathTrace");
}
