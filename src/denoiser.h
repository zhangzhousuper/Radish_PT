#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

#include "common.h"
#include "glm/fwd.hpp"
#include "pathtrace.h"
#include "scene.h"

struct GBuffer {
    GBuffer() = default;

    void create(int width, int height);
    void destroy();
    void render(DevScene *scene, const Camera &cam);
    void update(const Camera &cam);

    __host__ __device__ glm::vec3 *getNormal() {
        return normal[frame];
    }
    __host__ __device__ int *getPrimId() {
        return primId[frame];
    }
    __host__ __device__ float *getDepth() {
        return depth[frame];
    }

    glm::vec3 *albedo    = nullptr;
    glm::vec3 *normal[2] = {nullptr};
    int       *motion    = nullptr;
    float     *depth[2]  = {nullptr};
    int       *primId[2] = {nullptr};
    int        frame     = 0;

    Camera lastCam;
    int    width;
    int    height;
};

struct EAWaveletFilter {
    EAWaveletFilter() = default;

    EAWaveletFilter(int width, int height) :
        width(width), height(height) {
    }

    void filter(glm::vec3 *colorOut, glm::vec3 *colorIn, const GBuffer &gBuffer,
                const Camera &cam, int level);
    void filter(glm::vec3 *colorOut, glm::vec3 *colorIn, float *varianceOut, float *varianceIn,
                const GBuffer &gBuffer, const Camera &cam, int level);

    float sigLumin  = 64.f;
    float sigNormal = .2f;
    float sigDepth  = 1.f;

    int width  = 0;
    int height = 0;
};

struct LeveledEAWFilter {
    LeveledEAWFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void LeveledEAWFilter::filter(glm::vec3 *&colorIn, const GBuffer &gBuffer, const Camera &cam);

    EAWaveletFilter waveletFilter;
    int             level  = 0;
    glm::vec3      *tmpImg = nullptr;
};

struct SpatioTemporalFilter {
    SpatioTemporalFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void temporalAccumulate(glm::vec3 *colorIn, const GBuffer &gBuffer);

    EAWaveletFilter waveletFilter;
    int             level = 0;

    glm::vec3 *accumColor  = nullptr;
    glm::vec2 *accumMoment = nullptr;
    bool       firstTime   = true;
};

void denoiserInit(int width, int height);
void denoiserFree();

void modulateAlbedo(glm::vec3 *devImage, const GBuffer &gBuffer);
void addImage(glm::vec3 *devImage, glm::vec3 *in, int width, int height);
void addImage(glm::vec3 *out, glm::vec3 *in1, glm::vec3 *in2, int width, int height);