#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda.h>
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

static Scene            *hstScene           = nullptr;
static GuiDataContainer *guiData            = nullptr;
static PathSegment      *devPaths           = nullptr;
static PathSegment      *devTerminatedPaths = nullptr;
static Intersection     *devIntersections   = nullptr;

// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> devPathsThr;
static thrust::device_ptr<PathSegment> devTerminatedPathsThr;

static thrust::device_ptr<Intersection> devIntersectionsThr;

static int looper = 0;

void InitDataContainer(GuiDataContainer *imGuiData) {
    guiData = imGuiData;
}

void pathTraceInit(Scene *scene) {
    hstScene = scene;

    const Camera &cam        = hstScene->camera;
    const int     pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&devPaths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&devTerminatedPaths, pixelcount * sizeof(PathSegment));
    devPathsThr           = thrust::device_ptr<PathSegment>(devPaths);
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
                               int height, int toneMapping) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }
    int idx = x + (y * width);

    glm::vec3 color = image[idx];

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
    int px  = image[idx] % width;
    int py  = image[idx] / width;

    glm::vec3 color = glm::vec3(glm::vec2(px, py) / glm::vec2(width, height), 0.f);

    color = Math::gammaCorrection(color);

    glm::ivec3 icolor =
        glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
    pbo[idx] = make_uchar4(icolor.x, icolor.y, icolor.z, 0);
}

void copyImageToPBO(uchar4 *devPBO, glm::vec3 *devImage, int width, int height,
                    int toneMapping) {
    const int BlockSize = 32;
    dim3      blockSize(BlockSize, BlockSize);
    dim3      blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
    sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height,
                                            toneMapping);
}

void copyImageToPBO(uchar4 *devPBO, float *devImage, int width, int height) {
    const int BlockSize = 32;
    dim3      blockSize(BlockSize, BlockSize);
    dim3      blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
    sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height);
}

void copyImageToPBO(uchar4 *devPBO, int *devImage, int width, int height) {
    const int BlockSize = 32;
    dim3      blockSize(BlockSize, BlockSize);
    dim3      blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
    sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height);
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generateRayFromCamera(DevScene *scene, Camera cam, int iter,
                                      int          traceDepth,
                                      PathSegment *pathSegments) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        PathSegment &segment = pathSegments[index];

        Sampler rng =
            makeSeededRandomEngine(iter, index, traceDepth, scene->sampleSequence);

        // Sample4D used for antialiasing
        segment.ray = cam.sample(x, y, sample4D(rng));

        segment.throughput  = glm::vec3(1.f);
        segment.directIllum = glm::vec3(0.f);

        segment.pixelIndex       = index;
        segment.remainingBounces = traceDepth;
    }
}
__global__ void previewGBuffer(int iter, DevScene *scene, Camera cam,
                               glm::vec3 *image, int kind) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }
    int     index = y * cam.resolution.x + x;
    Sampler rng   = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);

    Ray          ray = cam.sample(x, y, sample4D(rng));
    Intersection intersec;
    scene->intersect(ray, intersec);

    if (kind == 0) {
        image[index] += intersec.pos;
    } else if (kind == 1) {
        if (intersec.primId != NullPrimitive) {
            Material m = scene->getTexturedMaterialAndSurface(intersec);
        }
        image[index] += (intersec.norm + 1.f) * .5f;
    } else if (kind == 2) {
        image[index] += glm::vec3(intersec.uv, 1.f);
    }
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int numPaths,
                                     PathSegment *pathSegments, DevScene *scene,
                                     Intersection *intersections) {
    int pathIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pathIdx >= numPaths) {
        return;
    }

    Intersection intersec;
    PathSegment  segment = pathSegments[pathIdx];

    scene->intersect(segment.ray, intersec);

    if (intersec.primId != NullPrimitive) {
        if (scene->materials[intersec.matId].type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
            if (glm::dot(intersec.norm, segment.ray.direction) < 0.f) {
                intersec.primId = NullPrimitive;
            } else
#endif
                if (depth != 0) {
                // If not first ray, preserve previous sampling information for
                // MIS calculation
                intersec.prevPos = segment.ray.origin;
            }
        } else {
            intersec.wo = -segment.ray.direction;
        }
    }
    intersections[pathIdx] = intersec;
}

__global__ void pathIntegSampleSurface(int iter, int depth,
                                       PathSegment  *segments,
                                       Intersection *intersections,
                                       DevScene *scene, int numPaths) {
    const int SamplesConsumedOneIter = 7;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= numPaths) {
        return;
    }

    Intersection intersec = intersections[idx];
    PathSegment &segment  = segments[idx];

    if (intersec.primId == NullPrimitive) {
        // Environment map
        if (scene->envMap != nullptr) {
            if (scene->envMap != nullptr) {
                glm::vec3 w = segment.ray.direction;
                glm::vec3 radiance =
                    scene->envMap->linearSample(Math::toPlane(w)) * segment.throughput;

                if (depth == 0) {
                    segment.directIllum += radiance * segment.throughput;
                } else {
                    float weight = segment.prev.deltaSample ? 1.f : Math::powerHeuristic(segment.prev.BSDFPdf, scene->enviromentMapPdf(w));
                    segment.directIllum += radiance * weight;
                }
            }
        }
        segment.remainingBounces = 0;

        if (Math::luminance(segment.directIllum) < 1e-4f) {
            segment.pixelIndex = PixelIdxForTerminated;
        }
        return;
    }

    Sampler rng = makeSeededRandomEngine(
        iter, idx, 4 + depth * SamplesConsumedOneIter, scene->sampleSequence);

    Material material = scene->getTexturedMaterialAndSurface(intersec);

    glm::vec3 accRadiance(0.f);

    if (material.type == Material::Type::Light) {
        PrevBSDFSampleInfo prev = segment.prev;

        glm::vec3 radiance = material.baseColor;
        if (depth == 0) {
            accRadiance += radiance;
        } else if (prev.deltaSample) {
            accRadiance += radiance * segment.throughput;
        } else {
            float lightPdf = Math::pdfAreaToSolidAngle(
                Math::luminance(radiance) * scene->sumLightPowerInv * scene->getPrimitiveArea(intersec.primId),
                intersec.prevPos, intersec.pos, intersec.norm);
            float BSDFPdf = prev.BSDFPdf;

            accRadiance += radiance * segment.throughput * Math::powerHeuristic(BSDFPdf, lightPdf);
        }
        segment.remainingBounces = 0;
    } else {
        //  delta BSDFs, which have a probability density function of zero and
        //  require special handling.
        bool deltaBSDF = (material.type == Material::Type::Dielectric);
        if (material.type != Material::Type::Dielectric && glm::dot(intersec.norm, intersec.wo) < 0.f) {
            intersec.norm = -intersec.norm;
        }

        if (!deltaBSDF) {
            glm::vec3 radiance;
            glm::vec3 wi;
            float     lightPdf =
                scene->sampleDirectLight(intersec.pos, sample4D(rng), radiance, wi);

            if (lightPdf > 0.f) {
                float BSDFPdf = material.pdf(intersec.norm, intersec.wo, wi);
                accRadiance += segment.throughput * material.BSDF(intersec.norm, intersec.wo, wi) * radiance * Math::satDot(intersec.norm, wi) / lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
            }
        }

        BSDFSample sample;
        material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);

        if (sample.type == BSDFSampleType::Invalid) {
            // Terminate path if sampling fails
            segment.remainingBounces = 0;
        } else if (sample.pdf < 1e-8f) {
            segment.remainingBounces = 0;
        } else {
            bool deltaSample = (sample.type & BSDFSampleType::Specular);
            segment.throughput *=
                sample.bsdf / sample.pdf * (deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));
            segment.ray  = makeOffsetedRay(intersec.pos, sample.dir);
            segment.prev = {sample.pdf, deltaSample};

            segment.remainingBounces--;
        }
    }
    segment.directIllum += accRadiance;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image,
                            PathSegment *iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths) {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.pixelIndex >= 0 && iterationPath.remainingBounces <= 0) {
            glm::vec3 r = iterationPath.directIllum;
            if (isnan(r.x) || isnan(r.y) || isnan(r.z) || isinf(r.x) || isinf(r.y) || isinf(r.z)) {
                return;
            }
            image[iterationPath.pixelIndex] +=
                glm::clamp(r, glm::vec3(0.f), glm::vec3(FLT_MAX / 10.f));
        }
    }
}

__global__ void singleKernelPT(int iter, int maxDepth, DevScene *scene,
                               Camera cam, glm::vec3 *directIllum,
                               glm::vec3 *indirectIllum) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }
    glm::vec3 direct(0.f);
    glm::vec3 indirect(0.f);

    int     index = y * cam.resolution.x + x;
    Sampler rng   = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);

    Ray ray = cam.sample(x, y, sample4D(rng));

    Intersection intersec;
    scene->intersect(ray, intersec);

    if (intersec.primId == NullPrimitive) {
        if (scene->envMap != nullptr) {
            glm::vec2 uv = Math::toPlane(ray.direction);
            direct += scene->envMap->linearSample(uv);
        }
        goto WriteRadiance;
    }

    Material material = scene->getTexturedMaterialAndSurface(intersec);
#if DENOISER_DEMODULATE
    glm::vec3 albedo   = material.baseColor;
    material.baseColor = glm::vec3(1.f);
#endif
    if (material.type == Material::Type::Light) {
        if (glm::dot(intersec.norm, ray.direction) > 0.f) {
            direct = material.baseColor;
        }
        goto WriteRadiance;
    }

    glm::vec3 throughput(1.f);
    intersec.wo = -ray.direction;

    for (int depth = 1; depth <= maxDepth; depth++) {
        bool deltaBSDF = (material.type == Material::Type::Dielectric);

        if (material.type != Material::Type::Dielectric && glm::dot(intersec.norm, intersec.wo) < 0.f) {
            intersec.norm = -intersec.norm;
        }

        if (!deltaBSDF) {
            glm::vec3 radiance;
            glm::vec3 wi;
            float     lightPdf =
                scene->sampleDirectLight(intersec.pos, sample4D(rng), radiance, wi);

            if (lightPdf > 0.f) {
                float BSDFPdf = material.pdf(intersec.norm, intersec.wo, wi);
                (depth == 1 ? direct : indirect) +=
                    throughput * material.BSDF(intersec.norm, intersec.wo, wi) * radiance * Math::satDot(intersec.norm, wi) / lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
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

        throughput *= sample.bsdf / sample.pdf * (deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));

        ray = makeOffsetedRay(intersec.pos, sample.dir);

        glm::vec3 curPos = intersec.pos;
        scene->intersect(ray, intersec);

        intersec.wo = -ray.direction;

        if (intersec.primId == NullPrimitive) {
            if (scene->envMap != nullptr) {
                glm::vec3 radiance =
                    scene->envMap->linearSample(Math::toPlane(ray.direction)) * throughput;

                float weight =
                    deltaSample ? 1.f : Math::powerHeuristic(sample.pdf, scene->enviromentMapPdf(ray.direction));

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

            float lightPdf = Math::pdfAreaToSolidAngle(
                Math::luminance(radiance) * 2.f * PI * scene->getPrimitiveArea(intersec.primId) * scene->sumLightPowerInv,
                curPos, intersec.pos, intersec.norm);

            float weight =
                deltaSample ? 1.f : Math::powerHeuristic(sample.pdf, lightPdf);
            indirect += throughput * radiance * weight;

            break;
        }
    }
WriteRadiance:
#if DENOISER_DEMODULATE
    // direct /= albedo + DEMODULATE_EPS;
    // indirect /= albedo + DEMODULATE_EPS;
#endif

    if (!isnan(direct.x) && !isnan(direct.y) && !isnan(direct.z) && !isinf(direct.x) && !isinf(direct.y) && !isinf(direct.z)) {
        directIllum[index] = direct / (direct + 1.f);
    }

    if (!isnan(indirect.x) && !isnan(indirect.y) && !isnan(indirect.z) && !isinf(indirect.x) && !isinf(indirect.y) && !isinf(indirect.z)) {
        indirectIllum[index] = indirect / (indirect + 1.f);
    }
}

__global__ void BVHVisualize(int iter, DevScene *scene, Camera cam,
                             glm::vec3 *image) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }
    int index = y * cam.resolution.x + x;

    Sampler rng = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);
    Ray     ray = cam.sample(x, y, sample4D(rng));

    Intersection intersec;
    scene->visualizedIntersect(ray, intersec);

    float logDepth = 0.f;
    int   size     = scene->BVHSize;
    while (size) {
        logDepth += 1.f;
        size >>= 1;
    }

    image[index] += glm::vec3(float(intersec.primId) / logDepth * .06f);
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
void pathTrace(glm::vec3 *DirectIllum, glm::vec3 *IndirectIllum) {
    const Camera &cam = hstScene->camera;

    const int BlockSizeSinglePTX = 8;
    const int BlockSizeSinglePTY = 8;

    int blockNumSinglePTX = ceilDiv(cam.resolution.x, BlockSizeSinglePTX);
    int blockNumSinglePTY = ceilDiv(cam.resolution.y, BlockSizeSinglePTY);

    dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
    dim3 singlePTBlockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);

    singleKernelPT<<<singlePTBlockNum, singlePTBlockSize>>>(
        looper, Settings::traceDepth, hstScene->devScene, cam, DirectIllum,
        IndirectIllum);

    checkCUDAError("pathTrace");
    looper = (looper + 1) % SobolSampleNum;

    /*
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2D(8, 8);
    const dim3 blocksPerGrid2D(
        (cam.resolution.x + blockSize2D.x - 1) / blockSize2D.x,
        (cam.resolution.y + blockSize2D.y - 1) / blockSize2D.y);

    int depth = 0;
    int numPaths = pixelcount;

    auto devTerminatedThr = devTerminatedPathsThr;

    if (Settings::tracer == Tracer::Streamed) {
      generateRayFromCamera<<<blocksPerGrid2D, blockSize2D>>>(
          hstScene->devScene, cam, iter, Settings::traceDepth, devPaths);
      checkCUDAError("PT::generateRayFromCamera");
      cudaDeviceSynchronize();

      bool iterationComplete = false;
      while (!iterationComplete) {
        // clean shading chunks
        cudaMemset(devIntersections, 0, pixelcount * sizeof(Intersection));

        // tracing
        const int BlockSizeIntersec = 128;
        int blockNumIntersec =
            (numPaths + BlockSizeIntersec - 1) / BlockSizeIntersec;
        computeIntersections<<<blockNumIntersec, BlockSizeIntersec>>>(
            depth, numPaths, devPaths, hstScene->devScene, devIntersections);

        checkCUDAError("PT::computeInteractions");
        cudaDeviceSynchronize();

        const int BlockSizeSample = 64;
        int blockNumSample = (numPaths + BlockSizeSample - 1) / BlockSizeSample;

        pathIntegSampleSurface<<<blockNumSample, BlockSizeSample>>>(
            iter, depth, devPaths, devIntersections, hstScene->devScene,
            numPaths);

        checkCUDAError("PT::sampleSurface");
        cudaDeviceSynchronize();
        // Compact paths that are terminated but carry contribution into a
        // separate buffer
        devTerminatedThr =
            thrust::remove_copy_if(devPathsThr, devPathsThr + numPaths,
                                   devTerminatedThr, CompactTerminatedPaths());
        // Only keep active paths
        auto end = thrust::remove_if(devPathsThr, devPathsThr + numPaths,
                                     RemoveInvalidPaths());
        numPaths = end - devPathsThr;
        // std::cout << "Remaining paths: " << numPaths << "\n";
        iterationComplete = (numPaths == 0);
        depth++;

        if (guiData != nullptr) {
          guiData->TracedDepth = depth;
        }
      }

      // Assemble this iteration and apply it to the image
      const int BlockSizeGather = 128;
      dim3 numBlocksPixels = (pixelcount + BlockSizeGather - 1) / BlockSizeGather;
      int numContributing = devTerminatedThr.get() - devTerminatedPaths;
      finalGather<<<numBlocksPixels, BlockSizeGather>>>(numContributing, devImage,
                                                        devTerminatedPaths);
    } else {
      const int BlockSizeSinglePTX = 8;
      const int BlockSizeSinglePTY = 8;
      int blockNumSinglePTX =
          (cam.resolution.x + BlockSizeSinglePTX - 1) / BlockSizeSinglePTX;
      int blockNumSinglePTY =
          (cam.resolution.y + BlockSizeSinglePTY - 1) / BlockSizeSinglePTY;

      dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
      dim3 singlePTBlockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);

      if (Settings::tracer == Tracer::SingleKernel) {
        singleKernelPT<<<singlePTBlockNum, singlePTBlockSize>>>(
            iter, Settings::traceDepth, hstScene->devScene, cam, devImage);
      } else if (Settings::tracer == Tracer::BVHVisualize) {
        BVHVisualize<<<singlePTBlockNum, singlePTBlockSize>>>(
            iter, hstScene->devScene, cam, devImage);
      } else {
        previewGBuffer<<<singlePTBlockNum, singlePTBlockSize>>>(
            iter, hstScene->devScene, cam, devImage, Settings::GBufferPreviewOpt);
      }

      if (guiData != nullptr) {
        guiData->TracedDepth = Settings::traceDepth;
      }
    }

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2D, blockSize2D>>>(
        pbo, cam.resolution, iter, devImage, Settings::toneMapping);

    // Retrieve image from GPU
    // ince the path tracing algorithm is executed on the GPU, the image data is
    // stored in the device memory. In order to display the image on the screen,
    // we need to copy the image data from the device memory to the host memory,
    // where it can be accessed by the display system.
    cudaMemcpy(hstScene->state.image.data(), devImage,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathTrace");
    */
}