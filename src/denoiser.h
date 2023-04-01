#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

#include "common.h"
#include "pathtrace.h"
#include "scene.h"

struct GBuffer {
#if DENOISER_ENCODE_NORMAL
    using NormT = glm::vec2;
#else
    using NormT = glm::vec3;
#endif

    GBuffer() = default;

    void create(int width, int height);
    void destroy();
    void render(DevScene *scene, const Camera &cam);
    void update(const Camera &cam);

    __device__ NormT *getNormal() {
        return normal[frameIdx];
    }
    __device__ NormT *lastNormal() {
        return normal[frameIdx ^ 1];
    }
    __device__ int *getPrimId() {
        return primId[frameIdx];
    }
    __device__ int *lastPrimId() {
        return primId[frameIdx ^ 1];
    }

#if DENOISER_ENCODE_POSITION
    __device__ float *getDepth() {
        return depth[frameIdx];
    }
    __device__ float *lastDepth() {
        return depth[frameIdx ^ 1];
    }
#else
    __device__ glm::vec3 *getPos() {
        return position[frameIdx];
    }
    __device__ glm::vec3 *lastPos() {
        return position[frameIdx ^ 1];
    }
#endif

    glm::vec3 *albedo    = nullptr;
    NormT     *normal[2] = {nullptr};

    int *motion = nullptr;

#if DENOISER_ENCODE_POSITION
    float *depth[2] = {nullptr};
#else
    glm::vec3 *position[2] = {nullptr};
#endif
    int *primId[2] = {nullptr};
    int  frameIdx  = 0;

    Camera lastCam;
    int    width;
    int    height;
};

struct EAWaveletFilter {
    EAWaveletFilter() = default;

    EAWaveletFilter(int width, int height, float sigLumin, float sigNormal, float sigDepth) :
        width(width), height(height), sigLumin(sigLumin), sigNormal(sigNormal), sigDepth(sigDepth) {
    }

    void filter(glm::vec3 *colorOut, glm::vec3 *colorIn, const GBuffer &gBuffer,
                const Camera &cam, int level);
    void filter(glm::vec3 *colorOut, glm::vec3 *colorIn, float *varianceOut, float *varianceIn,
                float *filteredVar, const GBuffer &gBuffer, const Camera &cam, int level);

    float sigLumin;
    float sigNormal;
    float sigDepth;

    int width  = 0;
    int height = 0;
};

struct LeveledEAWFilter {
    LeveledEAWFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void filter(glm::vec3 *&colorOut, glm::vec3 *colorIn, const GBuffer &gBuffer, const Camera &cam);

    EAWaveletFilter waveletFilter;
    int             level  = 0;
    glm::vec3      *tmpImg = nullptr;
};

struct SpatioTemporalFilter {
    SpatioTemporalFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void temporalAccumulate(glm::vec3 *colorIn, const GBuffer &gBuffer);
    void estimateVariance();
    void filterVariance();

    void filter(glm::vec3 *&colorOut, glm::vec3 *colorIn, const GBuffer &gBuffer, const Camera &cam);

    void nextFrame();

    EAWaveletFilter waveletFilter;
    int             level = 0;

    glm::vec3 *accumColor[2]  = {nullptr};
    glm::vec3 *accumMoment[2] = {nullptr};
    float     *variance       = nullptr;
    bool       firstTime      = true;

    glm::vec3 *tmpColor    = nullptr;
    float     *tmpVar      = nullptr;
    float     *filteredVar = nullptr;
    int        frameIdx    = 0;
};

void modulateAlbedo(glm::vec3 *devImage, const GBuffer &gBuffer);
void addImage(glm::vec3 *devImage, glm::vec3 *in, int width, int height);
void addImage(glm::vec3 *out, glm::vec3 *in1, glm::vec3 *in2, int width, int height);