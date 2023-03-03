#include <cmath>
#include <cstdio>
#include <cuda.h>
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
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#define ERRORCHECK 1

#define FILENAME                                                               \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err) {
    return;
  }

  fprintf(stderr, "CUDA error");
  if (file) {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
  getchar();
#endif
  exit(EXIT_FAILURE);
#endif
}

__host__ __device__ thrust::default_random_engine
makeSeededRandomEngine(int iter, int index, int depth) {
  int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
  return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter,
                               glm::vec3 *image) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    glm::vec3 color = image[index] / (float)iter;
    glm::vec3 mapped = Math::gammaCorrect(Math::ACES(color));
    glm::ivec3 intColor = glm::ivec3(glm::clamp(mapped * 255.f, 0.f, 255.f));

    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 0;
    pbo[index].x = intColor.x;
    pbo[index].y = intColor.y;
    pbo[index].z = intColor.z;
  }
}

static Scene *hst_scene = NULL;
static GuiDataContainer *guiData = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Material *dev_materials = NULL;
static PathSegment *dev_paths = NULL;
static ShadeableIntersection *dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer *imGuiData) { guiData = imGuiData; }

void pathtraceInit(Scene *scene) {
  hst_scene = scene;

  const Camera &cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(),
             scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

  // TODO: initialize any extra device memeory you need

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
  cudaFree(dev_image); // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  cudaFree(dev_intersections);
  // TODO: clean up any extra device memory you created

  checkCUDAError("pathtraceFree");
}

/**
 * @brief Map a pair of evenly distributed random numbers to a point on a unit
 * disk.
 *
 */

__host__ __device__ glm::vec2 ConcentricSampleDisk(glm::vec2 c) {
  float r = glm::sqrt(c.x);
  float theta = 2.0f * PI * c.y;
  return glm::vec2(r * glm::cos(theta), r * glm::sin(theta));
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
    thrust::uniform_real_distribution<float> u(0.f, 1.f);
    // add antialiasing jitter
    float rx = u(rng);
    float ry = u(rng);
    float rz = u(rng);
    float rw = u(rng);

    PathSegment &segment = pathSegments[index];

    // TODO: implement antialiasing by jittering the ray
    float aspect = float(cam.resolution.x) / float(cam.resolution.y);
    float tanFovY = glm::tan(glm::radians(cam.fov.y));
    glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
    glm::vec2 screenPos = glm::vec2(x, y) * pixelSize;
    glm::vec2 ruv = screenPos + pixelSize * glm::vec2(rx, ry);
    // it is important to flip the y axis here
    ruv = 1.f - ruv * 2.f;

    glm::vec3 pLens =
        glm::vec3(ConcentricSampleDisk(rz, rw) * cam.lensRadius, 0.f);
    glm::vec3 pFocus =
        glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY * cam.focalDistance,
                  cam.focalDistance);
    glm::vec3 dir = pFocus - pLens;
    dir = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);
    // the result is in world space by using the camera matrix

    segment.ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
    segment.ray.direction = dir;

    segment.throughput = glm::vec3(1.0f);

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
  }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int num_paths,
                                     PathSegment *pathSegments, Geom *geoms,
                                     int geoms_size,
                                     ShadeableIntersection *intersections) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index < num_paths) {
    PathSegment pathSegment = pathSegments[path_index];

    float dist;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++) {
      Geom &geom = geoms[i];

      // TODO: add more intersection tests here... triangle? metaball? CSG?
      dist = intersectGeom(geom, pathSegment.ray, tmp_intersect, tmp_normal,
                           outside);

      // Compute the minimum t from the intersection tests to determine what
      // scene geometry object was hit first.
      if (dist > 0.0f && t_min > dist) {
        t_min = dist;
        hit_geom_index = i;
        intersect_point = tmp_intersect;
        normal = tmp_normal;
      }
    }

    if (hit_geom_index == -1) {
      intersections[path_index].dist = -1.0f;
    } else {
      // The ray hits something
      intersections[path_index].dist = t_min;
      intersections[path_index].materialId = geoms[hit_geom_index].materialId;
      intersections[path_index].surfaceNormal = normal;
    }
  }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(int iter, int num_paths,
                                  ShadeableIntersection *shadeableIntersections,
                                  PathSegment *pathSegments,
                                  Material *materials) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.dist > 0.0f) { // if the intersection exists...
                                    // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].throughput *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a
      // one-liner
      else {
        float lightTerm =
            glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].throughput *=
            (materialColor * lightTerm) * 0.3f +
            ((1.0f - intersection.dist * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].throughput *=
            u01(rng); // apply some noise because why not
      }
      // If there was no intersection, color the ray black.
      // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
      // used for opacity, in which case they can indicate "no opacity".
      // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}

__global__ void pathIntegSampleSurface(int iter, PathSegment *segments,
                                       ShadeableIntersection *intersections,
                                       Material *materials, int num_paths) {
  // Add the current iteration's output to the overall image
  __global__ void finalGather(int nPaths, glm::vec3 *image,
                              PathSegment *iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= nPaths) {
      return;
    }
    ShadeableIntersection intersection = intersections[index];
    if (intersection.dist < 0.0f) {
      return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    Material material = materials[intersection.materialId];
  }

  /**
   * Wrapper for the __global__ call that sets up the kernel calls and does a
   * ton of memory management
   */
  void pathtrace(uchar4 * pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric
    //     distance, t, or a "distance along the ray." t = -1.0 indicates no
    //     intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment *dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {

      // clean shading chunks
      cudaMemset(dev_intersections, 0,
                 pixelcount * sizeof(ShadeableIntersection));

      // tracing
      dim3 numblocksPathSegmentTracing =
          (num_paths + blockSize1d - 1) / blockSize1d;
      computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
          depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(),
          dev_intersections);
      checkCUDAError("trace one bounce");
      cudaDeviceSynchronize();
      depth++;

      // TODO:
      // --- Shading Stage ---
      // Shade path segments based on intersections and generate new rays by
      // evaluating the BSDF.
      // Start off with just a big kernel that handles all the different
      // materials you have in the scenefile.
      // TODO: compare between directly shading the path segments and shading
      // path segments that have been reshuffled to be contiguous in memory.

      shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
          iter, num_paths, dev_intersections, dev_paths, dev_materials);
      iterationComplete =
          true; // TODO: should be based off stream compaction results.

      if (guiData != NULL) {
        guiData->TracedDepth = depth;
      }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image,
                                                  dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter,
                                                     dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
  }
