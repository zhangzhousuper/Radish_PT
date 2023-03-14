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

  size_t size() const { return sizeof(glm::vec3) * mWidth * mHeight; }

  int mWidth;
  int mHeight;
  glm::vec3 *mPixels = nullptr;
};

struct DevTexture {
  int width;
  int height;
  glm::vec3 *data;
};

__device__ inline glm::vec3 texelFetch(const DevTexture &texture, int x,
                                       int y) {
  return texture.data[y * texture.width + x];
}
