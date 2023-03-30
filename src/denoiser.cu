#include "denoiser.h"
#include "glm/fwd.hpp"
#include <cuda_device_runtime_api.h>

__device__ constexpr float Gaussian5x5[] = {.0625f, .25f, .375f, .25f, .0625f};

__global__ void renderGBuffer(DevScene *scene, Camera cam,
                              GBuffer gBuffer) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }
    int idx = x + y * cam.resolution.x;

    float     aspect    = cam.resolution.x / (float) cam.resolution.y;
    float     tanFovY   = glm::tan(glm::radians(cam.fov.y));
    glm::vec2 pixelsize = 1.f / glm::vec2(cam.resolution);
    glm::vec2 scr       = glm::vec2(x, y) * pixelsize;
    glm::vec2 ruv       = scr + pixelsize * glm::vec2(0.5f);
    ruv                 = 1.f - ruv * 2.f;

    glm::vec3 pLens(0.f);
    glm::vec3 pFocus =
        glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
    glm::vec3 dir = pFocus - pLens;

    Ray ray;
    ray.origin    = cam.position + cam.right * pLens.x + cam.up * pLens.y;
    ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);

    Intersection intersect;
    scene->intersect(ray, intersect);

    if (intersect.primId != NullPrimitive) {
        bool isLight = scene->materials[intersect.matId].type == Material::Type::Light;
        int  matId   = intersect.matId;
        if (isLight) {
            matId = NullPrimitive - 1;
#if SCENE_LIGHT_SINGLE_SIDED
            if (glm::dot(intersect.norm, ray.direction) < 0.f) {
                intersect.primId = NullPrimitive;
            }
#endif
        }
        Material material = scene->getTexturedMaterialAndSurface(intersect);

        gBuffer.albedo[idx]      = isLight ? glm::vec3(1.f) : material.baseColor;
        gBuffer.getNormal()[idx] = intersect.norm;
        gBuffer.getPrimId()[idx] = matId;
        gBuffer.getDepth()[idx]  = glm::distance(intersect.pos, ray.origin);

        glm::ivec2 lastPos  = gBuffer.lastCam.getRasterCoord(intersect.pos);
        gBuffer.motion[idx] = lastPos.y * cam.resolution.x + lastPos.x;
    } else {
        gBuffer.albedo[idx]      = glm::vec3(0.f);
        gBuffer.getNormal()[idx] = glm::vec3(0.f);
        gBuffer.getPrimId()[idx] = NullPrimitive;
        gBuffer.getDepth()[idx]  = 0.f;
    }
}

__device__ float weightLuminance(glm::vec3 *color, int p, int q) {
    return 0.f;
}

__device__ float weightLuminance(glm::vec4 *colorVar, int p, int q) {
    return 0.f;
}

__device__ float weightNormal(const GBuffer &gBuffer, int p, int q) {
}

__global__ void waveletFilter(glm::vec3 *colorOut,
                              glm::vec3 *colorIn, GBuffer gBuffer, float sigDepth,
                              float sigNormal, float sigLuminance, Camera cam,
                              int level) {
    int step = 1 << level;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }
    int idxP    = x + y * cam.resolution.x;
    int primIdP = gBuffer.getPrimId()[idxP];

    if (primIdP <= NullPrimitive) {
        colorOut[idxP] = colorIn[idxP];
        return;
    }

    glm::vec3 colorP  = colorIn[idxP];
    glm::vec3 normalP = gBuffer.getNormal()[idxP];
    glm::vec3 posP    = cam.getPosition(x, y, gBuffer.getDepth()[idxP]);

    glm::vec3 sum       = glm::vec3(0.f);
    float     weightSum = 0.f;

#pragma unroll
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int qx   = x + i * step;
            int qy   = y + j * step;
            int idxQ = qx + qy * cam.resolution.x;

            if (qx >= cam.resolution.x || qy >= cam.resolution.y) {
                continue;
            }
            if (gBuffer.getPrimId()[idxQ] != primIdP) {
                continue;
            }
            glm::vec3 normalQ = gBuffer.getNormal()[idxQ];
            glm::vec3 posQ    = cam.getPosition(qx, qy, gBuffer.getDepth()[idxQ]);
            glm::vec3 colorQ  = colorIn[idxQ];

            float distColor2 = glm::dot(colorP - colorQ, colorP - colorQ);
            float wColor     = glm::min(1.f, glm::exp(-distColor2 / sigLuminance));

            float distNormal2 = glm::dot(normalP - normalQ, normalP - normalQ);
            float wNormal     = glm::min(1.f, glm::exp(-distNormal2 / sigNormal));

            float distPos2 = glm::dot(posP - posQ, posP - posQ);
            float wPos     = glm::min(1.f, glm::exp(-distPos2 / sigDepth));

            float weight =
                wColor * wNormal * wPos * Gaussian5x5[i + 2] * Gaussian5x5[j + 2];
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

__global__ void waveletFilter(glm::vec3 *colorOut,
                              glm::vec3 *colorIn, float *varianceOut, float *varianceIn, GBuffer gBuffer, float sigDepth,
                              float sigNormal, float sigLuminance, Camera cam,
                              int level) {
    int step = 1 << level;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }

#pragma unroll
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= -2; j++) {
        }
    }
}

__global__ void modulate(glm::vec3 *devImage, GBuffer gBuffer, int width,
                         int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int       idx   = x + y * width;
        glm::vec3 color = devImage[idx];
        color           = color / (1.f - color);
        color *= DENOSIE_COMPRESS;
        devImage[idx] *=
            glm::max(gBuffer.albedo[idx] - DEMODULATE_EPS, glm::vec3(0.f));
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

__global__ void temporalAccumulate(glm::vec3 *colorAccum, glm::vec2 *momentAccum, glm::vec3 *colorIn, GBuffer gBuffer, bool first) {
    const float alpha = 0.2f;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= gBuffer.width || y >= gBuffer.height) {
        return;
    }
    int idx = x + y * gBuffer.width;

    int primId  = gBuffer.getPrimId()[idx];
    int lastIdx = gBuffer.motion[idx];

    if (primId <= NullPrimitive || first || gBuffer.getPrimId()[lastIdx] != primId) {
        glm::vec3 color  = colorIn[idx];
        float     lum    = Math::luminance(color);
        colorAccum[idx]  = color;
        momentAccum[idx] = glm::vec2(lum, lum * lum);
        return;
    }

    glm::vec3 lastColor = colorIn[lastIdx];
    float     lastLum   = Math::luminance(lastColor);
    colorAccum[idx]     = glm::mix(colorAccum[idx], lastColor, alpha);
    momentAccum[idx]    = glm::mix(momentAccum[idx], glm::vec2(lastLum, lastLum * lastLum), alpha);
}

void GBuffer::create(int width, int height) {
    this->width   = width;
    this->height  = height;
    int numPixels = width * height;
    albedo        = cudaMalloc<glm::vec3>(numPixels);
    motion        = cudaMalloc<int>(numPixels);
    for (int i = 0; i < 2; i++) {
        normal[i] = cudaMalloc<glm::vec3>(numPixels);
        primId[i] = cudaMalloc<int>(numPixels);
        depth[i]  = cudaMalloc<float>(numPixels);
    }
}

void GBuffer::destroy() {
    cudaSafeFree(albedo);
    cudaSafeFree(motion);
    for (int i = 0; i < 2; i++) {
        cudaSafeFree(normal[i]);
        cudaSafeFree(primId[i]);
        cudaSafeFree(depth[i]);
    }
}

void GBuffer::update(const Camera &cam) {
    lastCam = cam;
    frame ^= 1;
}

void GBuffer::render(DevScene *scene, const Camera &cam) {
    constexpr int BlockSize = 8;
    dim3          blockSize(BlockSize, BlockSize);
    dim3          blockNum(ceilDiv(cam.resolution.x, BlockSize),
                           ceilDiv(cam.resolution.y, BlockSize));
    renderGBuffer<<<blockNum, blockSize>>>(scene, cam, *this);
    checkCUDAError("renderGBuffer");
}

void modulateAlbedo(glm::vec3 *devImage, GBuffer gBuffer, int width,
                    int height) {
    constexpr int BlockSize = 32;
    dim3          blockSize(BlockSize, BlockSize);
    dim3          blockNum(ceilDiv(gBuffer.width, BlockSize), ceilDiv(gBuffer.height, BlockSize));
    modulate<<<blockNum, blockSize>>>(devImage, gBuffer, gBuffer.width, gBuffer.height);
    checkCUDAError("modulate");
}

void addImage(glm::vec3 *devImage, glm::vec3 *in, int width, int height) {
    constexpr int BlockSize = 32;
    dim3          blockSize(BlockSize, BlockSize);
    dim3          blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
    add<<<blockNum, blockSize>>>(devImage, in, width, height);
}

void EAWaveletFilter::filter(glm::vec3 *colorOut, glm::vec3 *colorIn,
                             const GBuffer &gBuffer, const Camera &cam,
                             int level) {
    constexpr int BlockSize = 8;
    dim3          blockSize(BlockSize, BlockSize);
    dim3          blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
    waveletFilter<<<blockNum, blockSize>>>(colorOut, colorIn, gBuffer, sigDepth,
                                           sigNormal, sigLumin, cam, level);
    checkCUDAError("EAW Filter");
}

void EAWaveletFilter::filter(glm::vec3 *colorOut, glm::vec3 *colorIn,
                             float *varianceOut, float *varianceIn,
                             const GBuffer &gBuffer, const Camera &cam,
                             int level) {
    constexpr int BlockSize = 32;
    dim3          blockSize(BlockSize, BlockSize);
    dim3          blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
    waveletFilter<<<blockNum, blockSize>>>(
        colorOut, colorIn, varianceOut, varianceIn, gBuffer, sigDepth, sigNormal, sigLumin, cam, level);
}

void LeveledEAWFilter::create(int width, int height, int level) {
    this->level   = level;
    waveletFilter = EAWaveletFilter(width, height);
    tmpImg        = cudaMalloc<glm::vec3>(width * height);
}

void LeveledEAWFilter::destroy() {
    cudaSafeFree(tmpImg);
}

void LeveledEAWFilter::filter(glm::vec3 *&devColorIn, const GBuffer &gBuffer, const Camera &cam) {
    for (int i = 0; i < level; i++) {
        waveletFilter.filter(tmpImg, devColorIn, gBuffer, cam, i);
        std::swap(devColorIn, tmpImg);
    }
}

void SpatioTemporalFilter::create(int width, int height, int level) {
    this->level   = level;
    accumColor    = cudaMalloc<glm::vec3>(width * height);
    accumMoment   = cudaMalloc<glm::vec2>(width * height);
    waveletFilter = EAWaveletFilter(width, height);
}

void SpatioTemporalFilter::destroy() {
    cudaSafeFree(accumColor);
    cudaSafeFree(accumMoment);
}

void SpatioTemporalFilter::temporalAccumulate(glm::vec3 *colorIn, const GBuffer &gBuffer) {
    constexpr int BlockSize = 32;
    dim3          blockSize(BlockSize, BlockSize);
    dim3          blockNum(ceilDiv(gBuffer.width, BlockSize), ceilDiv(gBuffer.height, BlockSize));
    ::temporalAccumulate<<<blockNum, blockSize>>>(accumColor, accumMoment, colorIn, gBuffer, firstTime);
    checkCUDAError("Temporal Accumulate");
    firstTime = false;
}

void denoiserInit(int width, int height) {
}

void denoiserFree() {
}