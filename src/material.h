#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

#include "glm/fwd.hpp"
#include "mathUtil.h"

#define INVALID_PDF -1.f;

struct Material {
  enum Type { Lambertian = 0, Metallic = 1, Dielectric = 2, Light = 3 };
  int type;
  glm::vec3 baseColor;
  struct {
    float exponent;
    glm::vec3 color;
  } specular;
  float metallic;
  float roughness;
  float ior;
  float emittance;
};

enum BSDFSampleType {
  Diffuse = 1 << 0,
  Glossy = 1 << 1,
  Specular = 1 << 2,

  Reflection = 1 << 4,
  Transmission = 1 << 5,
};

struct BSDFSample {
  glm::vec3 wi;
  glm::vec3 bsdf;
  float pdf;
  int type;
};

__device__ inline float fresnelApprox(float cosTheta) {
  return Math::pow5(1.f - cosTheta);
}

__device__ static glm::vec3 fresnelShlick(float lDotH, glm::vec3 f0) {
  return glm::mix(f0, glm::vec3(1.f), Math::pow5(1.f - lDotH));
}

__device__ static float fresnel(float cosIn, float ior) {
  if (cosIn < 0.f) {
    ior = 1.f / ior;
    cosIn = -cosIn;
  }
  float sinIn = glm::sqrt(1.f - cosIn * cosIn);
  float sinTr = sinIn / ior;
  if (sinTr >= 1.f) {
    return 1.f;
  }

  float cosTr = glm::sqrt(1.f - sinTr * sinTr);
  float rPar = (cosIn - ior * cosTr) / (cosIn + ior * cosTr);
  float rPer = (ior * cosIn - cosTr) / (ior * cosIn + cosTr);
  return (rPar * rPar + rPer * rPer) / 2.f;
}

__device__ static glm::vec3 lambertianBSDF(glm::vec3 n, glm::vec3 wo,
                                           glm::vec3 wi,
                                           const Material &material) {
  return material.baseColor * Math::satDot(n, wi) * INV_PI;
  // satDot makes sure that the dot product is positive
}

__device__ static float lambertianPDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
  return glm::dot(n, wi) * INV_PI;
}

__device__ static void lambertianSample(glm::vec3 n, glm::vec3 wo,
                                        const Material &material, glm::vec3 r,
                                        BSDFSample &sample) {
  sample.wi = Math::cosineSampleHemisphere(n, r.x, r.y);
  sample.bsdf = material.baseColor * INV_PI;
  sample.pdf = glm::dot(n, sample.wi) * INV_PI;
  sample.type = BSDFSampleType::Diffuse | BSDFSampleType::Reflection;
}

__device__ static glm::vec3 dielectricBSDF(glm::vec3 n, glm::vec3 wi,
                                           const Material &material) {
  return glm::vec3(0.f);
}

__device__ static float dielectricPDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi,
                                      const Material &material) {
  return 0.f;
}

__device__ static void dielectricSample(glm::vec3 n, glm::vec3 wo,
                                        const Material &material, glm::vec3 r,
                                        BSDFSample &sample) {
  float ior = material.ior;
  float pdfReflection = fresnel(glm::dot(n, wo), ior);
  float pdfTransmission = 1.f - pdfReflection;

  sample.pdf = 1.f;
  sample.bsdf = material.baseColor;

  if (r.z < pdfReflection) {
    sample.type = BSDFSampleType::Specular | BSDFSampleType::Reflection;
    sample.wi = glm::reflect(-wo, n);
  } else {
    if (!Math::refract(n, wo, ior, sample.wi)) {
      sample.pdf = INVALID_PDF;
      return;
    } else {
      sample.type = BSDFSampleType::Specular | BSDFSampleType::Transmission;
    }
  }
}