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
  } else if (primId <= NullPrimitive) {
    diff = true;
  } else if (gBuffer.lastPrimId()[lastIdx] != primId) {
    diff = true;
  } else {
    glm::vec3 norm = DECODE_NORM(gBuffer.getNormal()[idx]);
    glm::vec3 lastNorm = DECODE_NORM(gBuffer.lastNormal()[lastIdx]);
    if (Math::absDot(norm, lastNorm) < .1f) {
      diff = true;
    }
  }
  return diff ? T() : reservoir[lastIdx];
}

template <typename T>
__device__ T findSpatialNeighborDisk(T *reservoir, int x, int y,
                                     const GBuffer &gBuffer, glm::vec2 rand) {
  const float radius = 5.f;

  int idx = y * gBuffer.width + x;

  glm::vec2 p = Math::concentricSampleDisk(rand.x, rand.y) * radius;
  int px = x + .5f + p.x;
  int py = y + .5f + p.y;
  int pIdx = py * gBuffer.width + px;

  bool diff = false;

  if (px < 0 || px >= gBuffer.width || py < 0 || py >= gBuffer.height ||
      (px == x && py == y)) {
    diff = true;
  } else if (gBuffer.getPrimId()[pIdx] != gBuffer.getPrimId()[idx]) {
    diff = true;
  } else {
    glm::vec3 norm = DECODE_NORM(gBuffer.getNormal()[idx]);
    glm::vec3 pNorm = DECODE_NORM(gBuffer.getNormal()[pIdx]);
    if (glm::dot(norm, pNorm) < .1f) {
      diff = true;
    }
#if DENOISER_ENCODE_POSITION
    float depth = gBuffer.getDepth()[idx];
    float pDepth = gBuffer.getDepth()[pIdx];
    if (glm::abs(depth - pDepth) > depth * .1f) {
#else
    glm::vec3 pos = gBuffer.getPos()[idx];
    glm::vec3 pPos = gBuffer.getPos()[pIdx];
    if (glm::distance(pos, pPos) > .1f) {
#endif
      diff = true;
    }
  }
  return diff ? T() : reservoir[pIdx];
}

__device__ DirectReservoir
mergeSpatialNeighborDirect(DirectReservoir *reservoir, int x, int y,
                           const GBuffer &gBuffer, Sampler &rng) {
  DirectReservoir resvr;
#pragma unroll
  for (int i = 0; i < 5; i++) {
    DirectReservoir spatial =
        findSpatialNeighborDisk(reservoir, x, y, gBuffer, sample2D(rng));
    if (!spatial.invalid()) {
      resvr.merge(spatial, sample1D(rng));
    }
  }
  return resvr;
}

__global__ void
ReSTIRDirectKernel(int looper, int iter, DevScene *scene, Camera cam,
                   glm::vec3 *directIllum, DirectReservoir *reservoirOut,
                   DirectReservoir *reservoirIn, DirectReservoir *reservoirTemp,
                   GBuffer gBuffer, bool firstFrame, int reuseState) {
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
  material.baseColor = glm::vec3(1.f);

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

  if (!firstFrame && (reuseState & ReservoirReuse::Temporal)) {
    DirectReservoir temporal = findTemporalNeighbor(reservoirIn, idx, gBuffer);
    if (!temporal.invalid()) {
      reservoir.preClampedMerge<20>(temporal, sample1D(rng));
    }
  }

  sample = reservoir.sample;
  DirectReservoir tempReservoir = reservoir;

  if (reuseState & ReservoirReuse::Spatial) {
    reservoir.checkValidity();
    reservoirTemp[idx] = reservoir;
    __syncthreads();

    DirectReservoir spatial =
        mergeSpatialNeighborDirect(reservoirTemp, x, y, gBuffer, rng);
    if (!spatial.invalid() && !reservoir.invalid()) {
      reservoir.merge(spatial, sample1D(rng));
    }
  }
  tempReservoir.checkValidity();
  reservoirOut[idx] = tempReservoir;

  sample = reservoir.sample;
  if (!reservoir.invalid()) {
    direct = sample.Li * material.BSDF(intersec.norm, intersec.wo, sample.wi) *
             Math::satDot(intersec.norm, sample.wi) *
             reservoir.W(intersec, material);
  }

  if (Math::hasNanOrInf(direct)) {
    direct = glm::vec3(0.f);
  }
WriteRadiance:
  direct *= gBuffer.albedo[idx];
  directIllum[idx] =
      (directIllum[idx] * float(iter) + direct) / float(iter + 1);
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
      ReSTIR_FirstFrame, Settings::reservoirReuse);

  std::swap(directReservoir, lastDirectReservoir);

  if (ReSTIR_FirstFrame) {
    ReSTIR_FirstFrame = false;
  }

  checkCUDAError("ReSTIR Direct");
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