#pragma once

#include <cuda_runtime.h>
#include <iostream>

#include <glm/glm.hpp>
#include <vcruntime.h>

class Image {
  public:
    Image(int width, int height);
    Image(const std::string &filename);
    ~Image();

    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveJPG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);

    int width() const {
        return mWidth;
    }

    int height() const {
        return mHeight;
    }

    size_t byteSize() const {
        return mWidth * mHeight * sizeof(glm::vec3);
    }

    glm::vec3 *data() const {
        return mPixels;
    }

  private:
    int        mWidth;
    int        mHeight;
    glm::vec3 *mPixels = nullptr;
};

template <typename T>
__device__ T linearSample(T *data, glm::vec2 uv, int width, int height) {
    const float eps = FLT_MIN;
    uv              = glm::fract(uv);

    // float part
    float fx = uv.x * (width - eps) + 0.5f;
    float fy = uv.y * (height - eps) + 0.5f;

    // integer part
    int ix = glm::fract(fx) > 0.5f ? fx : fx - 1;
    int iy = glm::fract(fy) > 0.5f ? fy : fy - 1;
    if (ix < 0)
        ix += width;
    if (iy < 0)
        iy += height;

    int ux = ix + 1;
    int uy = iy + 1;
    if (ux >= width)
        ux -= width;
    if (uy >= height)
        uy -= height;

    float lx = glm::fract(fx + 0.5f);
    float ly = glm::fract(fy + 0.5f);

    // bilinear interpolation
    T c1 = glm::mix(data[iy * width + ix], data[iy * width + ux], lx);
    T c2 = glm::mix(data[uy * width + ix], data[uy * width + ux], lx);
    return glm::mix(c1, c2, ly);
}

struct DevTextureObj {
    DevTextureObj() = default;

    DevTextureObj(Image *img, glm::vec3 *devData) :
        width(img->width()), height(img->height()), devData(devData) {
    }

    __device__ glm::vec3 fetchTexel(int x, int y) {
        return devData[y * width + x];
    }

    __device__ glm::vec3 linearSample(glm::vec2 uv) {
        return ::linearSample(devData, uv, width, height);
    }
    int        width;
    int        height;
    glm::vec3 *devData;
};
