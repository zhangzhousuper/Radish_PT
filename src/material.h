#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/intersect.hpp>

#include "glm/fwd.hpp"
#include "mathUtil.h"

#define MATERIAL_DIELETRIC_USE_SCHLICK_APPROX false

#define NullTextureId -1
#define ProcTextureId -2
#define ProceduralTexId -2
#define INVALID_PDF -1.f

enum BSDFSampleType {
    Diffuse  = 1 << 0,
    Glossy   = 1 << 1,
    Specular = 1 << 2,

    Reflection   = 1 << 4,
    Transmission = 1 << 5,
    Invalid      = 1 << 15,
};

struct BSDFSample {
    glm::vec3 dir;
    glm::vec3 bsdf;
    float     pdf;
    uint32_t  type;
};

__device__ inline float fresnelSchlick(float lDotH, float ior) {
    float f0 = (1.f - ior) / (1.f + ior);
    return glm::mix(f0, 1.f, Math::pow5(1.f - lDotH));
}

__device__ inline glm::vec3 fresnelSchlick(float lDotH, glm::vec3 f0) {
    return glm::mix(f0, glm::vec3(1.f), Math::pow5(1.f - lDotH));
}

__device__ static float fresnel(float cosIn, float ior) {
#if MATERIAL_DIELECTRIC_USE_SCHLICK_APPROX
    return fresnelSchlick(cosIn, ior);
#else
    if (cosIn < 0.f) {
        ior   = 1.f / ior;
        cosIn = -cosIn;
    }
    float sinIn = glm::sqrt(1.f - cosIn * cosIn);
    float sinTr = sinIn / ior;
    if (sinTr >= 1.f) {
        return 1.f;
    }

    float cosTr = glm::sqrt(1.f - sinTr * sinTr);
    float rPar  = (cosIn - ior * cosTr) / (cosIn + ior * cosTr);
    float rPer  = (ior * cosIn - cosTr) / (ior * cosIn + cosTr);
    return (rPar * rPar + rPer * rPer) * .5f;
#endif
}

// cosTheta is the cosine of the angle between the normal and the direction
// alpha is the roughness
__device__ static float schlickG(float cosTheta, float alpha) {
    float a = alpha * .5f;
    return cosTheta / (cosTheta * (1.f - a) + a);
}

__device__ inline float smithG(float cosWo, float cosWi, float alpha) {
    return schlickG(glm::abs(cosWo), alpha) * schlickG(glm::abs(cosWi), alpha);
}
__device__ static float ggxDistribution(float cosTheta, float alpha) {
    if (cosTheta < 1e-6f) {
        return 0.f;
    }
    float alpha2 = alpha * alpha;
    float nom    = alpha2;
    float denom  = (cosTheta * cosTheta) * (alpha2 - 1.f) + 1.f;
    denom        = denom * denom * PI;
    return nom / denom;
}

// m is the microfacet normal
// n is the surface normal
// https://cseweb.ucsd.edu/~tzli/cse272/wi2022/lectures/04_uber_bsdf.pdf
__device__ static float ggxPdf(glm::vec3 n, glm::vec3 m, glm::vec3 wo,
                               float alpha) {
    return ggxDistribution(glm::dot(n, m), alpha) * schlickG(glm::dot(n, wo), alpha) * Math::absDot(m, wo) / Math::absDot(n, wo);
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

    glm::vec3 vh = glm::normalize((transInv * wo) * glm::vec3(alpha, alpha, 1.f));

    float     lenSq = vh.x * vh.x + vh.y * vh.y;
    glm::vec3 t     = lenSq > 0.f ? glm::vec3(-vh.y, vh.x, 0.f) / sqrt(lenSq) : glm::vec3(1.f, 0.f, 0.f);
    glm::vec3 b     = glm::cross(vh, t);

    glm::vec2 p = Math::concentricSampleDisk(r.x, r.y);
    float     s = 0.5f * (vh.z + 1.f);
    p.y         = (1.f - s) * glm::sqrt(1.f - p.x * p.x) + s * p.y;

    glm::vec3 h =
        t * p.x + b * p.y + vh * glm::sqrt(glm::max(0.f, 1.f - glm::dot(p, p)));
    h = glm::vec3(h.x * alpha, h.y * alpha, glm::max(0.f, h.z));
    return glm::normalize(transMat * h);
}

struct Material {
    enum Type { Lambertian,
                MetallicWorkflow,
                Dielectric,
                Disney,
                Light };

    __device__ glm::vec3 lambertianBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        return baseColor * INV_PI; // 1/pi for energy conservation
    }

    __device__ float lambertianPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        return Math::satDot(n, wi) * INV_PI;
    }

    __device__ void lambertianSample(glm::vec3 n, glm::vec3 wo, glm::vec3 r,
                                     BSDFSample &sample) {
        sample.dir  = Math::cosineSampleHemisphere(n, r.x, r.y);
        sample.bsdf = baseColor * INV_PI;
        sample.pdf  = Math::satDot(n, sample.dir) * INV_PI;
        sample.type = Diffuse | Reflection;
    }

    __device__ glm::vec3 dielectricBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        return glm::vec3(0.f);
    }

    __device__ float dielectricPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        return 0.f;
    }

    __device__ void dielectricSample(glm::vec3 n, glm::vec3 wo, glm::vec3 r,
                                     BSDFSample &sample) {
        float pdfRefl = fresnel(glm::dot(n, wo), ior);

        sample.bsdf = baseColor;

        if (r.z < pdfRefl) {
            sample.dir  = glm::reflect(-wo, n);
            sample.type = Specular | Reflection;
            sample.pdf  = 1.f;
        } else {
            bool result = Math::refract(n, wo, ior, sample.dir);
            if (!result) {
                sample.type = Invalid;
                return;
            }
            if (glm::dot(n, wo) < 0) {
                ior = 1.f / ior;
            }
            sample.type = Specular | Transmission;
            sample.pdf  = 1.f;
            sample.bsdf /= ior * ior;
        }
    }

    __device__ glm::vec3 metallicBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        float     alpha = roughness * roughness;
        glm::vec3 h     = glm::normalize(wo + wi);

        float cosO = glm::dot(n, wo);
        float cosI = glm::dot(n, wi);
        if (cosI * cosO < 1e-7f) {
            return glm::vec3(0.f);
        }

        glm::vec3 f = fresnelSchlick(
            glm::dot(h, wo), glm::mix(glm::vec3(.08f), baseColor, metallic));
        float d = ggxDistribution(glm::dot(n, h), alpha);
        float g = smithG(cosO, cosI, alpha);

        return glm::mix(baseColor * INV_PI * (1.f - metallic),
                        glm::vec3(g * d / (4.f * cosI * cosO)), f);
    }

    __device__ float metallicPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        glm::vec3 h = glm::normalize(wo + wi);
        return glm::mix(Math::satDot(n, wi) * INV_PI,
                        ggxPdf(n, h, wo, roughness * roughness) / (4.f * Math::absDot(h, wo)),
                        1.f / (2.f - metallic));
    }

    __device__ void metallicSample(glm::vec3 n, glm::vec3 wo, glm::vec3 r,
                                   BSDFSample &sample) {
        float alpha = roughness * roughness;

        if (r.z > (1.f / (2.f - metallic))) {
            sample.dir = Math::cosineSampleHemisphere(n, r.x, r.y);
        } else {
            glm::vec3 h = ggxSample(n, wo, alpha, glm::vec2(r));
            sample.dir  = -glm::reflect(wo, h);
        }

        if (glm::dot(n, sample.dir) < 0.f) {
            sample.type = Invalid;
        } else {
            sample.type = Glossy | Reflection;
            sample.pdf  = metallicPdf(n, wo, sample.dir);
            sample.bsdf = metallicBSDF(n, wo, sample.dir);
        }
    }

    __device__ glm::vec3 BSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        switch (type) {
        case Material::Type::Lambertian:
            return lambertianBSDF(n, wo, wi);
        case Material::Type::MetallicWorkflow:
            return metallicBSDF(n, wo, wi);
        case Material::Type::Dielectric:
            return dielectricBSDF(n, wo, wi);
        }

        return glm::vec3(0.f);
    }

    __device__ float pdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        switch (type) {
        case Material::Type::Lambertian:
            return lambertianPdf(n, wo, wi);
        case Material::Type::MetallicWorkflow:
            return metallicPdf(n, wo, wi);
        case Material::Type::Dielectric:
            return dielectricPdf(n, wo, wi);
        }
        return 0.f;
    }

    __device__ void sample(glm::vec3 n, glm::vec3 wo, glm::vec3 r,
                           BSDFSample &sample) {
        switch (type) {
        case Material::Type::Lambertian:
            lambertianSample(n, wo, r, sample);
            break;
        case Material::Type::MetallicWorkflow:
            metallicSample(n, wo, r, sample);
            break;
        case Material::Type::Dielectric:
            dielectricSample(n, wo, r, sample);
            break;
        default:
            sample.type = Invalid;
        }
    }
    int       type      = Type::Lambertian;
    glm::vec3 baseColor = glm::vec3(.9f);

    float metallic  = 0.f;
    float roughness = 1.f;
    float ior       = 1.5f;

    int baseColorMapId = NullTextureId;
    int normalMapId    = NullTextureId;
    int metallicMapId  = NullTextureId;
    int roughnessMapId = NullTextureId;
};