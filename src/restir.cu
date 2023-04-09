#include "glm/geometric.hpp"
#include "restir.h"

static DirectReservoir *directReservoir = nullptr;
static DirectReservoir *lastDirectReservoir = nullptr;
static DirectReservoir *directTemp = nullptr;
static bool ReSTIR_FirstFrame = true;

__device__ DirectReservoir mergeReservoir(const DirectReservoir &a,
                                          const DirectReservoir &b,
                                          glm::vec2 r) {
  DirectReservoir reservoir;
  reservoir.update(a.sample, a.weight, r.x);
  reservoir.update(b.sample, b.weight, r.y);
  reservoir.numSamples = a.numSamples + b.numSamples;
  return reservoir;
}

template <typename T>
__device__ T findTemporalNeighbor(T *reservoir, int idx,
                                  const GBuffer &gBuffer) {
  int primId = gBuffer.getPrimId()[idx];
  int lastIdx = gBuffer.motion[idx];
  bool diff = false;

  if (lastIdx < 0) {
    diff = true;
  } else if (primId != NullPrimitive) {
    diff = true;
  } else if (gBuffer.lastPrimId()[lastIdx] != primId) {
    diff = true;
  } else {
    glm::vec3 norm = DECODE_NORM(gBuffer.getNormal()[idx]);
    glm::vec3 lastNorm = DECODE_NORM(gBuffer.lastNormal()[lastIdx]);
    if (glm::abs(glm::dot(norm, lastNorm)) < .1f) {
      diff = true;
    }
  }
  return diff ? T() : reservoir[lastIdx].invalid() ? T() : reservoir[lastIdx];
}

template <typename T>
__device__ T findSpatialNeighborDisk(T *reservoir, int x, int y,
                                     const GBuffer &gBuffer, glm::vec2 rand) {
  const float radius = 30.f;

  int idx = y * gBuffer.width + x;

  glm::vec2 p = Math::concentricSampleDisk(rand.x, rand.y) * radius;
  int px = x + p.x;
  int py = y + p.y;
  int pIdx = py * gBuffer.width + px;

  bool diff = false;

  if (px < 0 || px >= gBuffer.width || py < 0 || py >= gBuffer.height) {
    diff = true;
  } else if (gBuffer.getPrimId()[pIdx] != gBuffer.getPrimId()[idx]) {
    diff = true;
  } else {
    glm::vec3 norm = DECODE_NORM(gBuffer.getNormal()[idx]);
    glm::vec3 pNorm = DECODE_NORM(gBuffer.getNormal()[pIdx]);
    if (Math::absDot(norm, pNorm) < .1f) {
      diff = true;
    }

    glm::vec3 pos = gBuffer.getPos()[idx];
    glm::vec3 pPos = gBuffer.getPos()[pIdx];
    if (glm::distance(pos, pPos) > .5f) {
      diff = true;
    }
  }
  return diff ? T() : reservoir[pIdx].invalid() ? T() : reservoir[pIdx];
}

__device__ DirectReservoir
mergeSpatialNeighborDirect(DirectReservoir *reservoir, int x, int y,
                           const GBuffer &gBuffer, Sampler &rng) {
  DirectReservoir resvr;
#pragma unroll
  for (int i = 0; i < 5; i++) {
    reservoir->merge(
        findSpatialNeighborDisk(reservoir, x, y, gBuffer, sample2D(rng)),
        sample1D(rng));
  }
  resvr.checkValidity();
  return resvr;
}

__global__ void ReSTIRDirectKernel(int looper, int iter, DevScene *scene,
                                   Camera cam, glm::vec3 *directIllum,
                                   DirectReservoir *reservoirOut,
                                   DirectReservoir *reservoirIn,
                                   DirectReservoir *reservoirTemp,
                                   GBuffer gBuffer, bool firstFrame) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
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

  DirectReservoir reservoir;
  for (int i = 0; i < RESERVOIR_SIZE; ++i) {
    glm::vec3 Li;
    glm::vec3 wi;
    float dist;

    float lightPdf = scene->sampleDirectLightNoVisibility(
        intersec.pos, sample4D(rng), Li, wi, dist);

    glm::vec3 bsdf = Li * material.BSDF(intersec.norm, intersec.wo, wi) *
                     Math::satDot(intersec.norm, wi);
    float weight = DirectReservoir::toScalar(bsdf / lightPdf);

    if (Math::isNanOrInf(weight) || lightPdf <= 0.f) {
      weight = 0.f;
    }
    reservoir.update({Li, wi, dist}, weight, sample1D(rng));
  }

  LightLiSample sample = reservoir.sample;

  if (scene->testOcclusion(intersec.pos,
                           intersec.pos + sample.wi * sample.dist)) {
    reservoir.weight = 0.f;
  }

  if (!firstFrame) {
    reservoir.preClampedMerge<20>(
        findTemporalNeighbor(reservoirIn, idx, gBuffer), sample1D(rng));
  }

  sample = reservoir.sample;
  if (reservoir.invalid()) {
    reservoir.clear();
  } else {
    direct = sample.Li * material.BSDF(intersec.norm, intersec.wo, sample.wi) *
             Math::satDot(intersec.norm, sample.wi) *
             reservoir.W(intersec, material);
  }
  if (Math::hasNanOrInf(direct)) {
    direct = glm::vec3(0.f);
  }
  reservoirOut[idx] = reservoir;

WriteRadiance:
  directIllum[idx] =
      (directIllum[idx] * float(iter) + direct) / float(iter + 1);
}

__global__ void spatialReuseDirect(int looper, int iter,
                                   DirectReservoir *reservoirOut,
                                   DirectReservoir *reservoirIn,
                                   DevScene *scene, GBuffer gBuffer) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= gBuffer.width || y >= gBuffer.height) {
    return;
  }
  int index = y * gBuffer.width + x;
  Sampler rng = makeSeededRandomEngine(looper, index, 5 * RESERVOIR_SIZE + 1,
                                       scene->sampleSequence);
  DirectReservoir reservoir = reservoirIn[index];

  if (reservoir.numSamples == 0) {
    reservoirOut[index].clear();
    return;
  }

#pragma unroll
  for (int i = 0; i < 5; i++) {
    DirectReservoir neighbor =
        findSpatialNeighborDisk(reservoirIn, x, y, gBuffer, sample2D(rng));
    if (!neighbor.invalid()) {
      reservoir.merge(neighbor, sample1D(rng));
    }
  }
  reservoirOut[index] = reservoir;
}
void ReSTIRDirect(glm::vec3 *directIllum, int iter, const GBuffer &gBuffer) {
  const Camera &cam = State::scene->camera;

  const int blockSizeSinglePTX = 8;
  const int blockSizeSinglePTY = 8;
  int blockNumSinglePTX = ceilDiv(cam.resolution.x, blockSizeSinglePTX);
  int blockNumSinglePTY = ceilDiv(cam.resolution.y, blockSizeSinglePTY);

  dim3 blockNum(blockNumSinglePTX, blockNumSinglePTY);
  dim3 blockSize(blockSizeSinglePTX, blockSizeSinglePTY);

  ReSTIRDirectKernel<<<blockNum, blockSize>>>(
      State::looper, iter, State::scene->devScene, cam, directIllum,
      directReservoir, lastDirectReservoir, directTemp, gBuffer,
      ReSTIR_FirstFrame);

  std::swap(directReservoir, lastDirectReservoir);

  if (ReSTIR_FirstFrame) {
    ReSTIR_FirstFrame = false;
  }

  checkCUDAError("ReSTIRDirect");
#if SAMPLER_USE_SOBOL
  State::looper = (State::looper + 1) % SobolSampleNum;
#else
  State::looper++;
#endif
}

void ReSTIRInit() {
  const Camera &cam = State::scene->camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  directReservoir = cudaMalloc<DirectReservoir>(pixelcount);
  cudaMemset(directReservoir, 0, sizeof(DirectReservoir) * pixelcount);
  lastDirectReservoir = cudaMalloc<DirectReservoir>(pixelcount);
  cudaMemset(lastDirectReservoir, 0, sizeof(DirectReservoir) * pixelcount);
  directTemp = cudaMalloc<DirectReservoir>(pixelcount);
  cudaMemset(directTemp, 0, sizeof(DirectReservoir) * pixelcount);
}

void ReSTIRFree() {
  cudaSafeFree(directReservoir);
  cudaSafeFree(lastDirectReservoir);
  cudaSafeFree(directTemp);
}