#include "denoiser.h"
#include "glm/fwd.hpp"
#include <cuda_device_runtime_api.h>

__device__ constexpr float Gaussian3x3[3][3] = {
    {.075f, .124f, .075f}, {.124f, .204f, .124f}, {.075f, .124f, .075f}};

__device__ constexpr float Gaussian5x5[5][5] = {
    {.0030f, .0133f, .0219f, .0133f, .0030f},
    {.0133f, .0596f, .0983f, .0596f, .0133f},
    {.0219f, .0983f, .1621f, .0983f, .0219f},
    {.0133f, .0596f, .0983f, .0596f, .0133f},
    {.0030f, .0133f, .0219f, .0133f, .0030f}};
#if DENOISER_ENCODE_NORMAL
#define ENCODE_NORM(x) Math::encodeNormalHemiOct32(x)
#define DECODE_NORM(x) Math::decodeNormalHemiOct32(x)
#else
#define ENCODE_NORM(x) x
#define DECODE_NORM(x) x
#endif

__global__ void renderGBuffer(DevScene *scene, Camera cam, GBuffer gBuffer) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam.resolution.x || y >= cam.resolution.y) {
    return;
  }
  int idx = x + y * cam.resolution.x;

  float aspect = cam.resolution.x / (float)cam.resolution.y;
  float tanFovY = glm::tan(glm::radians(cam.fov.y));
  glm::vec2 pixelsize = 1.f / glm::vec2(cam.resolution);
  glm::vec2 scr = glm::vec2(x, y) * pixelsize;
  glm::vec2 ruv = scr + pixelsize * glm::vec2(0.5f);
  ruv = 1.f - ruv * 2.f;

  glm::vec3 pLens(0.f);
  glm::vec3 pFocus =
      glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
  glm::vec3 dir = pFocus - pLens;

  Ray ray;
  ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
  ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);

  Intersection intersect;
  scene->intersect(ray, intersect);

  if (intersect.primId != NullPrimitive) {
    bool isLight =
        scene->materials[intersect.matId].type == Material::Type::Light;
    int matId = intersect.matId;
    if (isLight) {
      matId = NullPrimitive - 1;
#if SCENE_LIGHT_SINGLE_SIDED
      if (glm::dot(intersect.norm, ray.direction) < 0.f) {
        intersect.primId = NullPrimitive;
      }
#endif
    }
    Material material = scene->getTexturedMaterialAndSurface(intersect);

    gBuffer.albedo[idx] = material.baseColor;
    gBuffer.getNormal()[idx] = ENCODE_NORM(intersect.norm);
    gBuffer.getPrimId()[idx] = matId;
#if DENOISER_ENCODE_POSITION
    gBuffer.getDepth()[idx] = glm::distance(intersect.pos, ray.origin);
#else
    gBuffer.getPos()[idx] = intersect.pos;
#endif
    glm::ivec2 lastPos = gBuffer.lastCam.getRasterCoord(intersect.pos);
    if (lastPos.x >= 0 && lastPos.x < gBuffer.width && lastPos.y >= 0 &&
        lastPos.y < gBuffer.height) {
      gBuffer.motion[idx] = lastPos.y * cam.resolution.x + lastPos.x;
    } else {
      gBuffer.motion[idx] = -1;
    }
  } else {
    glm::vec3 albedo(0.f);
    if (scene->envMap != nullptr) {
      glm::vec2 uv = Math::toPlane(ray.direction);
      albedo = scene->envMap->linearSample(uv);
    }
    gBuffer.albedo[idx] = albedo;
    gBuffer.getNormal()[idx] = GBuffer::NormT(0.f);
    gBuffer.getPrimId()[idx] = NullPrimitive;
#if DENOISER_ENCODE_POSITION
    gBuffer.getDepth()[idx] = 1.f;
#else
    gBuffer.getPos()[idx] = glm::vec3(0.f);
#endif
    gBuffer.motion[idx] = 0;
  }
}

__global__ void waveletFilter(glm::vec3 *colorOut, glm::vec3 *colorIn,
                              GBuffer gBuffer, float sigDepth, float sigNormal,
                              float sigLuminance, Camera cam, int level) {
  int step = 1 << level;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cam.resolution.x || y >= cam.resolution.y) {
    return;
  }
  int idxP = x + y * cam.resolution.x;
  int primIdP = gBuffer.getPrimId()[idxP];

  if (primIdP <= NullPrimitive) {
    colorOut[idxP] = colorIn[idxP];
    return;
  }

  glm::vec3 colorP = colorIn[idxP];
  glm::vec3 normalP = DECODE_NORM(gBuffer.getNormal()[idxP]);
  glm::vec3 posP =
#if DENOISER_ENCODE_POSITION
      cam.getPosition(x, y, gBuffer.getDepth()[idxP]);
#else
      gBuffer.getPos()[idxP];
#endif
  glm::vec3 sum = glm::vec3(0.f);
  float weightSum = 0.f;

#pragma unroll
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      int qx = x + j * step;
      int qy = y + i * step;
      int idxQ = qx + qy * cam.resolution.x;

      if (qx >= cam.resolution.x || qy >= cam.resolution.y || qx < 0 ||
          qy < 0) {
        continue;
      }
      if (gBuffer.getPrimId()[idxQ] != primIdP) {
        continue;
      }
      glm::vec3 normalQ = DECODE_NORM(gBuffer.getNormal()[idxQ]);
      glm::vec3 posQ =
#if DENOISER_ENCODE_POSITION
          cam.getPosition(qx, qy, gBuffer.getDepth()[idxQ]);
#else
          gBuffer.getPos()[idxQ];
#endif
      glm::vec3 colorQ = colorIn[idxQ];

      float distColor2 = glm::dot(colorP - colorQ, colorP - colorQ);
      float wColor = glm::min(1.f, glm::exp(-distColor2 / sigLuminance));

      float distNormal2 = glm::dot(normalP - normalQ, normalP - normalQ);
      float wNormal = glm::min(1.f, glm::exp(-distNormal2 / sigNormal));

      float distPos2 = glm::dot(posP - posQ, posP - posQ);
      float wPos = glm::min(1.f, glm::exp(-distPos2 / sigDepth));

      float weight = wColor * wNormal * wPos * Gaussian5x5[i + 2][j + 2];
      sum += colorQ * weight;
      weightSum += weight;
    }
  }
  colorOut[idxP] = (weightSum == 0.f) ? colorIn[idxP] : sum / weightSum;
}

/*
 * SVGF version, filtering variance at the same time
 * Variance is stored as the last component of vec4
 */

__global__ void waveletFilter(glm::vec3 *colorOut, glm::vec3 *colorIn,
                              float *varianceOut, float *varianceIn,
                              float *varFiltered, GBuffer gBuffer,
                              float sigDepth, float sigNormal,
                              float sigLuminance, Camera cam, int level) {
  int step = 1 << level;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cam.resolution.x || y >= cam.resolution.y) {
    return;
  }
  int idxP = x + y * cam.resolution.x;
  int primIdP = gBuffer.getPrimId()[idxP];

  if (primIdP <= NullPrimitive) {
    colorOut[idxP] = colorIn[idxP];
    varianceOut[idxP] = varianceIn[idxP];
    return;
  }

  glm::vec3 colorP = colorIn[idxP];
  glm::vec3 normalP = DECODE_NORM(gBuffer.getNormal()[idxP]);
  glm::vec3 posP =
#if DENOISER_ENCODE_POSITION
      cam.getPosition(x, y, gBuffer.getDepth()[idxP]);
#else
      gBuffer.getPos()[idxP];
#endif
  glm::vec3 colorSum = glm::vec3(0.f);
  float varianceSum = 0.f;
  float weightSum = 0.f;
  float weight2Sum = 0.f;

#pragma unroll
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      int qx = x + j * step;
      int qy = y + i * step;
      int idxQ = qx + qy * cam.resolution.x;

      if (qx >= cam.resolution.x || qy >= cam.resolution.y || qx < 0 ||
          qy < 0) {
        continue;
      }
      glm::vec3 normalQ = DECODE_NORM(gBuffer.getNormal()[idxQ]);
      glm::vec3 posQ =
#if DENOISER_ENCODE_POSITION
          cam.getPosition(qx, qy, gBuffer.getDepth()[idxQ]);
#else
          gBuffer.getPos()[idxQ];
#endif
      float varQ = varianceIn[idxQ];
      glm::vec3 colorQ = colorIn[idxQ];

      float distPos2 = glm::dot(posP - posQ, posP - posQ);
      float wPos = glm::exp(-distPos2 / sigDepth) + 1e-4f;

      float wNormal =
          glm::pow(Math::satDot(normalP, normalQ), sigNormal) + 1e-4f;
      float denom =
          sigLuminance * glm::sqrt(glm::max(varFiltered[idxP], 0.f)) + 1e-4f;
      float wColor = glm::exp(-glm::abs(Math::luminance(colorP) -
                                        Math::luminance(colorQ)) /
                              denom) +
                     1e-4f;

      float weight = wColor * wNormal * wPos * Gaussian5x5[i + 2][j + 2];
      float weight2 = weight * weight;

      colorSum += colorQ * weight;
      varianceSum += varQ * weight2;
      weightSum += weight;
      weight2Sum += weight2;
    }
  }
  colorOut[idxP] =
      (weightSum < FLT_EPSILON) ? colorIn[idxP] : colorSum / weightSum;
  varianceOut[idxP] =
      (weight2Sum < FLT_EPSILON) ? varianceIn[idxP] : varianceSum / weight2Sum;
}

__global__ void modulate(glm::vec3 *devImage, GBuffer gBuffer, int width,
                         int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = x + y * width;
    glm::vec3 color = devImage[idx];
    color = Math::LDRToHDR(color);
    devImage[idx] = color * glm::max(gBuffer.albedo[idx], glm::vec3(0.f));
  }
}

__global__ void add(glm::vec3 *devImage, glm::vec3 *in, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * width + x;
    devImage[idx] += in[idx];
  }
}

__global__ void add(glm::vec3 *out, glm::vec3 *in1, glm::vec3 *in2, int width,
                    int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * width + x;
    out[idx] = in1[idx] + in2[idx];
  }
}

__global__ void temporalAccumulate(glm::vec3 *colorAccumOut,
                                   glm::vec3 *colorAccumIn,
                                   glm::vec3 *momentAccumOut,
                                   glm::vec3 *momentAccumIn, glm::vec3 *colorIn,
                                   GBuffer gBuffer, bool first) {
  const float alpha = 0.2f;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= gBuffer.width || y >= gBuffer.height) {
    return;
  }
  int idx = x + y * gBuffer.width;

  int primId = gBuffer.getPrimId()[idx];
  int lastIdx = gBuffer.motion[idx];

  bool diff = first;

  if (lastIdx < 0) {
    diff = true;
  } else if (primId <= NullPrimitive) {
    diff = true;
  } else if (gBuffer.lastPrimId()[lastIdx] != primId) {
    diff = true;
  } else {
    glm::vec3 normal = DECODE_NORM(gBuffer.getNormal()[idx]);
    glm::vec3 lastNormal = DECODE_NORM(gBuffer.lastNormal()[lastIdx]);
    if (glm::abs(glm::dot(normal, lastNormal)) < .1f) {
      diff = true;
    }
  }

  glm::vec3 color = colorIn[idx];
  glm::vec3 lastColor = colorAccumIn[lastIdx];
  glm::vec3 lastMoment = momentAccumIn[lastIdx];
  float lum = Math::luminance(color);

  glm::vec3 colorAccum;
  glm::vec3 momentAccum;

  if (diff) {
    colorAccum = color;
    momentAccum = glm::vec3(lum, lum * lum, 0.f);
  } else {
    colorAccum = glm::mix(lastColor, color, alpha);
    momentAccum = glm::vec3(
        glm::mix(glm::vec2(lastMoment), glm::vec2(lum, lum * lum), alpha),
        lastMoment.b + 1.f);
  }

  colorAccumOut[idx] = colorAccum;
  momentAccumOut[idx] = momentAccum;
}

__global__ void estimateVariance(float *variance, glm::vec3 *moment, int width,
                                 int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }
  int idx = x + y * width;

  glm::vec3 m = moment[idx];
  if (m.z > 3.5f) {
    // temporal variance
    variance[idx] = m.y - m.x * m.x;
  } else {
    // spatial variance
    glm::vec2 momentSum(0.f);
    int pixelCount = 0;

    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        int qx = x + j;
        int qy = y + i;

        if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
          continue;
        }
        int idxQ = qx + qy * width;
        momentSum += glm::vec2(moment[idxQ]);
        pixelCount++;
      }
    }
    momentSum /= pixelCount;
    variance[idx] = momentSum.y - momentSum.x * momentSum.x;
  }
}

__global__ void filterVariance(float *varianceOut, float *varianceIn, int width,
                               int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }
  int idx = x + y * width;
  float sum = 0.f;
  float weightSum = 0.f;
#pragma unroll
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int qx = x + i;
      int qy = y + j;

      if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
        continue;
      }
      int idxQ = qx + qy * width;
      float weight = Gaussian3x3[i + 1][j + 1];
      sum += varianceIn[idxQ] * weight;
      weightSum += weight;
    }
  }
  varianceOut[idx] = sum / weightSum;
}
void GBuffer::create(int width, int height) {
  this->width = width;
  this->height = height;
  int numPixels = width * height;
  albedo = cudaMalloc<glm::vec3>(numPixels);
  motion = cudaMalloc<int>(numPixels);

  for (int i = 0; i < 2; i++) {
    normal[i] = cudaMalloc<NormT>(numPixels);
    primId[i] = cudaMalloc<int>(numPixels);
#if DENOISER_ENCODE_POSITION
    depth[i] = cudaMalloc<float>(numPixels);
#else
    position[i] = cudaMalloc<glm::vec3>(numPixels);
#endif
  }
}

void GBuffer::destroy() {
  cudaSafeFree(albedo);
  cudaSafeFree(motion);
  for (int i = 0; i < 2; i++) {
    cudaSafeFree(normal[i]);
    cudaSafeFree(primId[i]);
#if DENOISER_ENCODE_POSITION
    cudaSafeFree(depth[i]);
#else
    cudaSafeFree(position[i]);
#endif
  }
}

void GBuffer::update(const Camera &cam) {
  lastCam = cam;
  frameIdx ^= 1;
}

void GBuffer::render(DevScene *scene, const Camera &cam) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int BlockSize = 8;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(cam.resolution.x, BlockSize),
                ceilDiv(cam.resolution.y, BlockSize));
  cudaEventRecord(start, 0);
  renderGBuffer<<<blockNum, blockSize>>>(scene, cam, *this);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GBuffer runtime%.3f ms\n", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCUDAError("renderGBuffer");
}

void modulateAlbedo(glm::vec3 *devImage, const GBuffer &gBuffer) {
  constexpr int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(gBuffer.width, BlockSize),
                ceilDiv(gBuffer.height, BlockSize));
  modulate<<<blockNum, blockSize>>>(devImage, gBuffer, gBuffer.width,
                                    gBuffer.height);
  checkCUDAError("modulate");
}

void addImage(glm::vec3 *devImage, glm::vec3 *in, int width, int height) {
  constexpr int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
  add<<<blockNum, blockSize>>>(devImage, in, width, height);
}

void addImage(glm::vec3 *out, glm::vec3 *in1, glm::vec3 *in2, int width,
              int height) {
  constexpr int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
  add<<<blockNum, blockSize>>>(out, in1, in2, width, height);
}

void EAWaveletFilter::filter(glm::vec3 *colorOut, glm::vec3 *colorIn,
                             const GBuffer &gBuffer, const Camera &cam,
                             int level) {
  constexpr int BlockSize = 8;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
  waveletFilter<<<blockNum, blockSize>>>(colorOut, colorIn, gBuffer, sigDepth,
                                         sigNormal, sigLumin, cam, level);
  checkCUDAError("EAW Filter");
}

void EAWaveletFilter::filter(glm::vec3 *colorOut, glm::vec3 *colorIn,
                             float *varianceOut, float *varianceIn,
                             float *filteredVar, const GBuffer &gBuffer,
                             const Camera &cam, int level) {
  constexpr int BlockSize = 16;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
  waveletFilter<<<blockNum, blockSize>>>(
      colorOut, colorIn, varianceOut, varianceIn, filteredVar, gBuffer,
      sigDepth, sigNormal, sigLumin, cam, level);
}

void LeveledEAWFilter::create(int width, int height, int level) {
  this->level = level;
  waveletFilter = EAWaveletFilter(width, height, 64.f, .2f, 1.f);
  tmpImg = cudaMalloc<glm::vec3>(width * height);
}

void LeveledEAWFilter::destroy() { cudaSafeFree(tmpImg); }

void LeveledEAWFilter::filter(glm::vec3 *&colorOut, glm::vec3 *colorIn,
                              const GBuffer &gBuffer, const Camera &cam) {
  waveletFilter.filter(colorOut, colorIn, gBuffer, cam, 0);

  waveletFilter.filter(tmpImg, colorOut, gBuffer, cam, 1);
  std::swap(colorOut, tmpImg);

  waveletFilter.filter(tmpImg, colorOut, gBuffer, cam, 2);
  std::swap(colorOut, tmpImg);

  waveletFilter.filter(tmpImg, colorOut, gBuffer, cam, 3);
  std::swap(colorOut, tmpImg);

  waveletFilter.filter(tmpImg, colorOut, gBuffer, cam, 4);
  std::swap(colorOut, tmpImg);
}
void SpatioTemporalFilter::create(int width, int height, int level) {
  this->level = level;
  for (int i = 0; i < 2; i++) {
    accumColor[i] = cudaMalloc<glm::vec3>(width * height);
    accumMoment[i] = cudaMalloc<glm::vec3>(width * height);
  }
  variance = cudaMalloc<float>(width * height);
  waveletFilter = EAWaveletFilter(width, height, 4.f, 128.f, 1.f);

  tmpColor = cudaMalloc<glm::vec3>(width * height);
  tmpVar = cudaMalloc<float>(width * height);
  filteredVar = cudaMalloc<float>(width * height);
}

void SpatioTemporalFilter::destroy() {
  for (int i = 0; i < 2; i++) {
    cudaSafeFree(accumColor[i]);
    cudaSafeFree(accumMoment[i]);
  }
  cudaSafeFree(variance);
  cudaSafeFree(tmpColor);
  cudaSafeFree(tmpVar);
  cudaSafeFree(filteredVar);
}

void SpatioTemporalFilter::temporalAccumulate(glm::vec3 *colorIn,
                                              const GBuffer &gBuffer) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(gBuffer.width, BlockSize),
                ceilDiv(gBuffer.height, BlockSize));
  cudaEventRecord(start, 0);
  ::temporalAccumulate<<<blockNum, blockSize>>>(
      accumColor[frameIdx], accumColor[frameIdx ^ 1], accumMoment[frameIdx],
      accumMoment[frameIdx ^ 1], colorIn, gBuffer, firstTime);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("temporalAccumulate runtime%.3f ms\n", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  firstTime = false;
  checkCUDAError("SpatioTemporalFilter::temporalAccumulate");
}

void SpatioTemporalFilter::estimateVariance() {
  constexpr int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(waveletFilter.width, BlockSize),
                ceilDiv(waveletFilter.height, BlockSize));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  ::estimateVariance<<<blockNum, blockSize>>>(variance, accumMoment[frameIdx],
                                              waveletFilter.width,
                                              waveletFilter.height);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("estimateVariance runtime%.3f ms\n", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCUDAError("SpatioTemporalFilter::estimateVariance");
}

void SpatioTemporalFilter::filterVariance() {
  constexpr int BlockSize = 32;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(waveletFilter.width, BlockSize),
                ceilDiv(waveletFilter.height, BlockSize));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  ::filterVariance<<<blockNum, blockSize>>>(
      filteredVar, variance, waveletFilter.width, waveletFilter.height);
  checkCUDAError("SpatioTemporalFilter::filterVariance");
}

void SpatioTemporalFilter::filter(glm::vec3 *&colorOut, glm::vec3 *colorIn,
                                  const GBuffer &gBuffer, const Camera &cam) {
  temporalAccumulate(colorIn, gBuffer);
  estimateVariance();

  filterVariance();
  waveletFilter.filter(colorOut, accumColor[frameIdx], tmpVar, variance,
                       filteredVar, gBuffer, cam, 0);
  std::swap(colorOut, accumColor[frameIdx]);
  std::swap(tmpVar, variance);

  filterVariance();
  waveletFilter.filter(colorOut, accumColor[frameIdx], tmpVar, variance,
                       filteredVar, gBuffer, cam, 1);
  std::swap(tmpVar, variance);

  filterVariance();
  waveletFilter.filter(tmpColor, colorOut, tmpVar, variance, filteredVar,
                       gBuffer, cam, 2);
  std::swap(tmpColor, colorOut);
  std::swap(tmpVar, variance);

  filterVariance();
  waveletFilter.filter(tmpColor, colorOut, tmpVar, variance, filteredVar,
                       gBuffer, cam, 3);
  std::swap(tmpColor, colorOut);
  std::swap(tmpVar, variance);

  filterVariance();
  waveletFilter.filter(tmpColor, colorOut, tmpVar, variance, filteredVar,
                       gBuffer, cam, 4);
  std::swap(tmpColor, colorOut);
  std::swap(tmpVar, variance);
}

void SpatioTemporalFilter::nextFrame() { frameIdx ^= 1; }