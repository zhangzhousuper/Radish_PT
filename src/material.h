#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

#include "glm/fwd.hpp"
#include "mathUtil.h"

#define INVALID_PDF -1.f;
#define MATERIAL_DIELETRIC_USE_SCHLICK_APPROX false

struct Material {
  enum Type { Lambertian, Metallic, Dielectric, Disney, Light };
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

  int textureId;
};

enum BSDFSampleType {
  Diffuse = 1 << 0,
  Glossy = 1 << 1,
  Specular = 1 << 2,

  Reflection = 1 << 4,
  Transmission = 1 << 5,
  Invalid = 1 << 15,
};

struct BSDFSample {
  glm::vec3 dir;
  glm::vec3 bsdf;
  float pdf;
  uint32_t type;
};

__device__ inline glm::vec3 fresnelSchlick(float lDotH, glm::vec3 f0) {
  return glm::mix(f0, glm::vec3(1.f), Math::pow5(1.f - lDotH));
}

__device__ static float fresnel(float cosIn, float ior) {
#if MATERIAL_DIELECTRIC_USE_SCHLICK_APPROX
  return fresnelSchlick(cosIn, ior);
#else
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
#endif
}

__device__ static glm::vec3 lambertianBSDF(glm::vec3 n, glm::vec3 wo,
                                           glm::vec3 wi,
                                           const Material &material) {
  return material.baseColor * Math::satDot(n, wi) * INV_PI;
  // satDot makes sure that the dot product is positive
}

__device__ static float lambertianPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi,
                                      const Material &m) {
  return glm::dot(n, wi) * INV_PI;
}

__device__ static void lambertianSample(glm::vec3 n, glm::vec3 wo,
                                        const Material &material, glm::vec3 r,
                                        BSDFSample &sample) {
  sample.dir = Math::cosineSampleHemisphere(n, r.x, r.y);
  sample.bsdf = material.baseColor * INV_PI;
  sample.pdf = glm::dot(n, sample.dir) * INV_PI;
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
  pdfReflection = 1.f;

  sample.bsdf = material.baseColor;

  if (r.z < pdfReflection) {
    sample.type = BSDFSampleType::Specular | BSDFSampleType::Reflection;
    sample.dir = glm::reflect(-wo, n);
    sample.pdf = 1.f;
  } else {
    bool result = Math::refract(n, wo, ior, sample.dir);
    if (!result) {
      sample.type = BSDFSampleType::Invalid;
      return;
    }
    if (glm::dot(n, wo) < 0) {
      ior = 1.f / ior;
    }
    sample.type = BSDFSampleType::Specular | BSDFSampleType::Transmission;
    sample.pdf = 1.f;
    sample.bsdf /= ior * ior;
  }
}

// cosTheta is the cosine of the angle between the normal and the direction
// alpha is the roughness
__device__ static float schlickG(float cosTheta, float alpha) {
  float alpha2 = alpha * alpha;
  return cosTheta / (cosTheta * (1.f - alpha2) + alpha2);
}

__device__ inline float smithG(float cosWo, float cosWi, float alpha) {
  return schlickG(glm::abs(cosWo), alpha) * schlickG(glm::abs(cosWi), alpha);
}
__device__ static float ggxDistribution(float cosTheta, float alpha) {
  float alpha2 = alpha * alpha;
  float nom = alpha2;
  float denom = (cosTheta * cosTheta) * (alpha2 - 1.f) + 1.f;
  denom = denom * denom * PI;
  return nom / denom;
}

// m is the microfacet normal
// n is the surface normal
// https://cseweb.ucsd.edu/~tzli/cse272/wi2022/lectures/04_uber_bsdf.pdf
__device__ static float ggxPdf(glm::vec3 n, glm::vec3 m, glm::vec3 wo,
                               float alpha) {
  return ggxDistribution(glm::dot(n, m), alpha) *
         schlickG(glm::dot(n, wo), alpha) * Math::absDot(m, wo) /
         Math::absDot(n, wo);
}

/**
 * Sample GGX microfacet distribution, but only visible normals.
 * This reduces invalid samples and make pdf values at grazing angles more
 * stable See [Sampling the GGX Distribution of Visible Normals, Eric Heitz,
 * JCGT 2018]: https://jcgt.org/published/0007/04/01/
 */

__device__ static glm::vec3 ggxSample(glm::vec3 n, glm::vec3 wo, float alpha,
                                      glm::vec2 r) {
  glm::mat3 transMat = Math::localRefMatrix(n);
  glm::mat3 transInv = glm::inverse(transMat);

  glm::vec3 vh = glm::normalize(transInv * wo) * glm::vec3(alpha, alpha, 1.f);

  float lenSq = vh.x * vh.x + vh.y * vh.y;
  glm::vec3 t = lenSq > 0.f ? glm::vec3(-vh.y, vh.x, 0.f) / sqrt(lenSq)
                            : glm::vec3(1.f, 0.f, 0.f);
  glm::vec3 b = glm::cross(vh, t);

  glm::vec2 p = Math::concentricSampleDisk(r.x, r.y);
  float s = 0.5f * (vh.z + 1.f);
  p.y = (1.f - s) * glm::sqrt(1.f - p.x * p.x) + s * p.y;

  glm::vec3 h =
      p.x + b * p.y + vh * glm::sqrt(glm::max(0.f, 1.f - glm::dot(p, p)));
  h = glm::normalize(glm::vec3(h.x * alpha, h.y * alpha, glm::max(0.f, h.z)));
  return transMat * h;
}

__device__ static glm::vec3 metallicBSDF(glm::vec3 n, glm::vec3 wo,
                                         glm::vec3 wi,
                                         const Material &material) {
  float alpha = material.roughness * material.roughness;
  glm::vec3 h = glm::normalize(wo + wi);

  float cosThetaO = glm::dot(n, wo);
  float cosThetaI = glm::dot(n, wi);
  if (cosThetaI * cosThetaO <= 1e-7f) {
    return glm::vec3(0.f);
  }

  glm::vec3 f =
      fresnelSchlick(glm::dot(h, wo), material.baseColor * material.metallic);
  float d = ggxDistribution(glm::dot(n, h), alpha);
  float g = smithG(cosThetaO, cosThetaI, alpha);

  return glm::mix(material.baseColor * (1.f - material.metallic),
                  glm::vec3(g * d / (4.f * cosThetaO * cosThetaI)), f);
}

__device__ static float metallicPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi,
                                    const Material &material) {
  glm::vec3 h = glm::normalize(wo + wi);
  return glm::mix(Math::satDot(n, wi) * INV_PI,
                  ggxPdf(n, h, wo, material.roughness * material.roughness) /
                      (4.f * Math::absDot(h, wo)),
                  1.f / (2.f - material.metallic));
}

__device__ static void metallicSample(glm::vec3 n, glm::vec3 wo,
                                      const Material &material, glm::vec3 r,
                                      BSDFSample &sample) {
  float alpha = material.roughness * material.roughness;

  if (r.z > (1.f / 2.f - material.metallic)) {
    sample.dir = Math::cosineSampleHemisphere(n, r.x, r.y);
  } else {
    glm::vec3 h = ggxSample(n, wo, alpha, glm::vec2(r.x, r.y));
    sample.dir = -glm::reflect(wo, h);
  }

  if (glm::dot(n, sample.dir) < 0.f) {
    sample.type = Invalid;
  } else {
    sample.type = BSDFSampleType::Reflection | BSDFSampleType::Glossy;
    sample.pdf = metallicPdf(n, wo, sample.dir, material);
    sample.bsdf = metallicBSDF(n, wo, sample.dir, material);
  }
}

__device__ static glm::vec3 materialBSDF(glm::vec3 n, glm::vec3 wo,
                                         glm::vec3 wi,
                                         const Material &material) {
  switch (material.type) {
  case Material::Lambertian:
    return lambertianBSDF(n, wo, wi, material);
  case Material::Type::Metallic:
    return metallicBSDF(n, wo, wi, material);
  case Material::Dielectric:
    return dielectricBSDF(n, wi, material);
  default:
    return glm::vec3(0.f);
  }
}

__device__ static float materialPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi,
                                    const Material &material) {
  switch (material.type) {
  case Material::Lambertian:
    return lambertianPdf(n, wo, wi, material);
  case Material::Type::Metallic:
    return metallicPdf(n, wo, wi, material);
  case Material::Dielectric:
    return dielectricPDF(n, wo, wi, material);
  default:
    return 0.f;
  }
}

__device__ static void materialSample(glm::vec3 n, glm::vec3 wo,
                                      const Material &material, glm::vec3 r,
                                      BSDFSample &sample) {
  switch (material.type) {
  case Material::Lambertian:
    lambertianSample(n, wo, material, r, sample);
    break;
  case Material::Metallic:
    metallicSample(n, wo, material, r, sample);
    break;
  case Material::Dielectric:
    dielectricSample(n, wo, material, r, sample);
    break;
  default:
    sample.type = Invalid;
    break;
  }
}