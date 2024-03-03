#pragma once

#include<limits>
#include<array>
#include<vector>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "sceneStructs.h"
#include "image.h"

enum BxDFFlags {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    Diffuse = 1 << 2,
    Glossy = 1 << 3,
    Specular = 1 << 4,
};

class scatter_record
{
public:
    glm::vec3 bsdf;
    float pdf = 0;
    bool delta;
    glm::vec3 dir;
};

class Material
{
public:
    enum Type {
        Lambertian,
        MetallicWorkflow,
        Dielectric,
        Microfacet,
        Light
    };
    
    __host__ __device__ color lambertianBSDF(const glm::vec3 &n, const glm::vec3 &wo, const glm::vec3 &wi, const glm::vec2 &uv)
    {
        return albedoSampler.linearSample(uv) / PI;
    }

    __host__ __device__ float lambertianPDF(const glm::vec3 &n, const glm::vec3 &wo, const glm::vec3 &wi)
    {
        return glm::dot(wi, n) / PI;
    }

    __host__ __device__ glm::vec3 lambertianScatter(const glm::vec3 &n, const glm::vec3 &wo, Sampler &sampler)
    {
        glm::vec2 r = sample2D(sampler);
        return math::sampleHemisphereCosine(n, r);
    }

    __host__ __device__ void lambertianScatterSample(const glm::vec3 &n, const glm::vec3 &wo, scatter_record &srec, Sampler &sampler, const glm::vec2& uv)
    {
        srec.bsdf = albedoSampler.linearSample(uv) / PI;
        srec.dir = math::sampleHemisphereCosine(n, sample2D(sampler));
        srec.pdf = glm::dot(srec.dir, n) / PI;
        srec.delta = false;
    }

    /// <summary>
    /// compute dielectric bsdf, pdf and scatter direction
    /// always assume one medium is air(ior = 1)
    /// </summary>
    /// <param name="n"></param>
    /// <param name="wo"></param>
    /// <param name="srec"></param>
    /// <param name="sampler"></param>
    /// <returns></returns>
    __host__ __device__ void dielectricScatterSample(const glm::vec3& n, const glm::vec3& wo, scatter_record& srec, Sampler& sampler, const glm::vec2& uv)
    {
        float ior1, ior2;

        if (glm::dot(wo, n) < 0)
        {
            (ior1 = 1, ior2 = ior);
        }
        else
        {
            (ior2 = 1, ior1 = ior);
        }

        float FresnelRefl = math::FresnelMaxwell(abs(glm::dot(wo, n)), ior1, ior2);

        srec.delta = true;
        //reflect
        if (sample1D(sampler) < FresnelRefl)
        {
            srec.pdf = 1.f;
            srec.dir = math::getReflectDir(n, wo);
            srec.bsdf = albedoSampler.linearSample(uv);
        }

        //refract
        else
        {
            srec.pdf = 1.f;
            srec.dir = math::getRefractDir(n, wo, ior1, ior2);
            srec.bsdf = albedoSampler.linearSample(uv) * (ior2 * ior2) / (ior1 * ior1);
        }

        srec.bsdf /= glm::abs(glm::dot(srec.dir, n));
    }

    __host__ __device__ glm::vec3 microfacetBSDF(const glm::vec3& n, const glm::vec3& wo, const glm::vec3 &wi, const glm::vec3 &sampleAlbedo, float sampleRoughness)
    {
        float a2 = sampleRoughness * sampleRoughness;
        float cosO = glm::dot(n, wo), cosI = glm::dot(n, wi);
        glm::vec3 wm = glm::normalize(wo + wi);
        float D = math::normalDistribGGX(glm::dot(wm, n), a2);
        float G2 = math::SmithG2(sampleRoughness, cosO, cosI);
        glm::vec3 F = math::FresnelSchilick(sampleAlbedo, glm::dot(wo, wm));
        glm::vec3 ret = F * D * G2 / glm::max(4 * cosO * cosI, 1e-8f);
        return F * D * G2 / glm::max(4 * cosO * cosI, 1e-8f);
    }

    __host__ __device__ float microfacetPDF(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& wi, float sampleRoughness)
    {
        float a2 = sampleRoughness * sampleRoughness;
        float cosO = glm::dot(n, wo), cosI = glm::dot(n, wi);
        glm::vec3 wm = glm::normalize(wo + wi);
        float D = math::normalDistribGGX(glm::dot(wm, n), a2);
        float G1 = math::SmithG1(sampleRoughness, cosO);
        return G1 * D / glm::max(4 * glm::dot(wo, n), 1e-8f);
    }

    __host__ __device__ void microfacetScatterSample(const glm::vec3& n, const glm::vec3& wo, scatter_record& srec, Sampler& sampler, const glm::vec2& uv)
    {
        glm::vec3 wm = math::sampleNormalGGX(n, -1.f * wo, roughness, sample2D(sampler));
        srec.dir = glm::reflect(wo, wm);
        if (glm::dot(srec.dir, n) * glm::dot(-1.f * wo, n) < 0)
        {
            srec.bsdf = glm::vec3(0);
            srec.pdf = 0;
            return;
        }
        float sampleRoughness = roughnessSampler.linearSample(uv).x;
        glm::vec3 sampleAlbedo = albedoSampler.linearSample(uv);
        srec.bsdf = microfacetBSDF(n, -1.f * wo, srec.dir, sampleAlbedo, sampleRoughness);
        srec.pdf = microfacetPDF(n, -1.f * wo, srec.dir, sampleRoughness);
    }

    __host__ __device__ void metallicScatterSample(const glm::vec3& n, const glm::vec3& wo, scatter_record& srec, Sampler& sampler, const glm::vec2& uv)
    {
        glm::vec3 wm = math::sampleNormalGGX(n, -1.f * wo, roughness, sample2D(sampler));
        srec.dir = glm::reflect(wo, wm);
        if (glm::dot(srec.dir, n) * glm::dot(-1.f * wo, n) < 0)
        {
            srec.bsdf = glm::vec3(0);
            srec.pdf = 0;
            return;
        }
        float sampleRoughness = roughnessSampler.linearSample(uv).x;
        float sampleMetallic = metallicSampler.linearSample(uv).x;
        glm::vec3 sampleAlbedo = albedoSampler.linearSample(uv);
        srec.bsdf = microfacetBSDF(n, -1.f * wo, srec.dir, sampleAlbedo, sampleRoughness);
        srec.pdf = microfacetPDF(n, -1.f * wo, srec.dir, sampleRoughness);
    }

    __host__ __device__ bool scatterSample(const glm::vec3 &n, const glm::vec3 &wo, scatter_record &srec, Sampler &sampler, const glm::vec2 &uv = glm::vec2(0.f))
    {
        switch (type)
        {
        case Material::Lambertian:
            lambertianScatterSample(n, wo, srec, sampler, uv);
            return true;
            break;
        case Material::MetallicWorkflow:
            metallicScatterSample(n, wo, srec, sampler, uv);
            break;
        case Material::Dielectric:
            dielectricScatterSample(n, wo, srec, sampler, uv);
            return true;
            break;
        case Material::Microfacet:
            microfacetScatterSample(n, wo, srec, sampler, uv);
            return true;
            break;
        case Material::Light:
            srec.bsdf = albedo;
            srec.pdf = 1.f;
            break;
        default:
            break;
        }

        return false;
    }

    // parameters of bsdf
    Material::Type type;
    color albedo    =   color(1.f);
    float roughness =   .0f;
    float metallic  =   .0f;
    float ior       =   1.5f;       // index of refraction

    int albedoMapID = -1;
    int metallicMapID = -1;
    int roughnessMapID = -1;
    int normalMapID = -1;

    devTexSampler albedoSampler;
    devTexSampler roughnessSampler;
    devTexSampler metallicSampler;
    devTexSampler normalSampler;
};