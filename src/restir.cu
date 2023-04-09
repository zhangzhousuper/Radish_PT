#include "restir.h"

static DirectReservoir *directReservoir = nullptr;
static DirectReservoir *lastDirectReservoir = nullptr;
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
__device__ T findTemporalReservoir(const T *lastReservoir, int idx,
                                   const GBuffer &gBuffer, bool fistFrame) {
  int primId = gBuffer.getPrimId()[idx];
  int lastIdx = gBuffer.motion[idx];
  bool diff = fistFrame;

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
  return diff ? T() : lastReservoir[idx];
}

template <typename T>
__device__ T findSpatialReservoir(T *lastReservoir, int idx,
                                  const GBuffer &gBuffer, T *spatialReservoir) {
}

__global__ void ReSTIRDirectKernel(int looper, int iter, DevScene *scene,
                                   Camera cam, glm::vec3 *directIllum,
                                   DirectReservoir *reservoirOut,
                                   DirectReservoir *reservoirIn,
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

  if (reservoir.invalid() ||
      scene->testOcclusion(intersec.pos,
                           intersec.pos + sample.wi * sample.dist)) {
    reservoir.clear();
  }

  DirectReservoir temporalReservoir =
      findTemporalReservoir(reservoirIn, idx, gBuffer, firstFrame);
  if (!temporalReservoir.invalid()) {
    if (temporalReservoir.numSamples > 19 * reservoir.numSamples) {
      temporalReservoir.weight *=
          19.f * reservoir.numSamples / temporalReservoir.numSamples;
      temporalReservoir.numSamples = 19 * reservoir.numSamples;
    }
    reservoir.merge(temporalReservoir, sample1D(rng));
  }

  if (reservoir.invalid()) {
    reservoir.clear();
  } else {
    LightLiSample sample = reservoir.sample;
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
      directReservoir, lastDirectReservoir, gBuffer, ReSTIR_FirstFrame);

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
}

void ReSTIRFree() {
  cudaSafeFree(directReservoir);
  cudaSafeFree(lastDirectReservoir);
}