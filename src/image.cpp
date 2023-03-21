#include <iostream>

#include <string>

#include <stb_image.h>
#include <stb_image_write.h>

#include "image.h"

Image::Image(int width, int height)
    : mWidth(width), mHeight(height), mPixels(new glm::vec3[width * height]) {}

Image::Image(const std::string &filename) {
  int channels;

  // The textures should be in the linear space
  stbi_ldr_to_hdr_gamma(1.f);
  float *data = stbi_loadf(filename.c_str(), &mWidth, &mHeight, &channels, 3);
  if (data) {
    std::cout << "Loaded " << filename << " (" << mWidth << "x" << mHeight
              << "x" << channels << ")" << std::endl;
  }
  if (!data) {
    throw std::runtime_error("Failed to load image " + filename);
  }

  mPixels = new glm::vec3[mWidth * mHeight];
  memcpy(mPixels, data, mWidth * mHeight * sizeof(glm::vec3));

  if (data) {
    stbi_image_free(data);
  }
}

Image::~Image() {
  if (mPixels) {
    delete[] mPixels;
  }
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel) {
  assert(expression : x >= 0 && x < mWidth && y >= 0 && y < mHeight);
  mPixels[y * mWidth + x] = pixel;
}

void Image::savePNG(const std::string &baseFilename) {
  unsigned char *bytes = new unsigned char[mWidth * mHeight * 3];
  for (int y = 0; y < mHeight; y++) {
    for (int x = 0; x < mWidth; x++) {
      int idx = y * mWidth + x;
      glm::vec3 pixel =
          glm::clamp(mPixels[idx], glm::vec3(0.0f), glm::vec3(1.0f)) * 255.0f;
      bytes[idx * 3 + 0] = (unsigned char)pixel.r;
      bytes[idx * 3 + 1] = (unsigned char)pixel.g;
      bytes[idx * 3 + 2] = (unsigned char)pixel.b;
    }
  }

  std::string filename = baseFilename + ".png";
  stbi_write_png(filename.c_str(), mWidth, mHeight, 3, bytes, mWidth * 3);

  std::cout << "Saved image " << filename << std::endl;

  delete[] bytes;
}

void Image::saveHDR(const std::string &baseFilename) {
  std::string filename = baseFilename + ".hdr";
  stbi_write_hdr(filename.c_str(), mWidth, mHeight, 3, (const float *)mPixels);

  std::cout << "Saved image " << filename << std::endl;
}
