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
  void saveHDR(const std::string &baseFilename);

  int width() const { return mWidth; }

  int height() const { return mHeight; }

  size_t byteSize() const { return mWidth * mHeight * sizeof(glm::vec3); }

  glm::vec3 *data() const { return mPixels; }

private:
  int mWidth;
  int mHeight;
  glm::vec3 *mPixels = nullptr;
};

struct DevTextureObj {
  DevTextureObj() = default;

  DevTextureObj(Image *img, glm::vec3 *devData)
      : width(img->width()), height(img->height()), devData(devData) {}

  __device__ glm::vec3 fetchTexel(int x, int y) {
    return devData[y * width + x];
  }

  __device__ glm::vec3 linearSample(glm::vec2 uv) {
    const float eps = FLT_MIN * 2.f;
    uv = glm::fract(uv);

    float fx = uv.x * (width - eps);
    float fy = uv.y * (height - eps);

    int ix = glm::fract(fx) < 0.5f ? int(fx) : int(fx) - 1;
    int iy = glm::fract(fy) < 0.5f ? int(fy) : int(fy) - 1;
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
    glm::vec3 c00 = fetchTexel(ix, iy);
    glm::vec3 c01 = fetchTexel(ix, uy);
    glm::vec3 c10 = fetchTexel(ux, iy);
    glm::vec3 c11 = fetchTexel(ux, uy);

    return glm::mix(glm::mix(c00, c10, lx), glm::mix(c01, c11, lx), ly);
  }

  int width;
  int height;
  glm::vec3 *devData;
};
