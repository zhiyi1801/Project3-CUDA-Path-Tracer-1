#pragma once

#include<limits>
#include<array>
#include<vector>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "sceneStructs.h"

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
    
    __host__ __device__ color lambertianBSDF(const glm::vec3 &n, const glm::vec3 &wo, const glm::vec3 &wi)
    {
        return albedo / PI;
    }

    __host__ __device__ float lambertianPDF(const glm::vec3 &n, const glm::vec3 &wo, const glm::vec3 &wi)
    {
        return glm::dot(wi, n) / PI;
    }

    __host__ __device__ glm::vec3 lambertianScatter(const glm::vec3 &n, const glm::vec3 &wo, Sampler &sampler)
    {
        return math::sampleHemisphereCosine(n, sampler);
    }

    __host__ __device__ void lambertianScatterSample(const glm::vec3 &n, const glm::vec3 &wo, scatter_record &srec, Sampler &sampler)
    {
        srec.bsdf = albedo / PI;
        srec.dir = math::sampleHemisphereCosine(n, sampler);
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
    __host__ __device__ void dielectricScatterSample(const glm::vec3& n, const glm::vec3& wo, scatter_record& srec, Sampler& sampler)
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
            srec.bsdf = albedo;
        }

        //refract
        else
        {
            srec.pdf = 1.f;
            srec.dir = math::getRefractDir(n, wo, ior1, ior2);
            srec.bsdf = albedo * (ior2 * ior2) / (ior1 * ior1);
        }

        srec.bsdf /= glm::abs(glm::dot(srec.dir, n));
    }

    __host__ __device__ glm::vec3 microfacetBSDF(const glm::vec3& n, const glm::vec3& wo, const glm::vec3 &wi)
    {
        float a2 = roughness * roughness;
        float cosO = glm::dot(n, wo), cosI = glm::dot(n, wi);
        glm::vec3 wm = glm::normalize(wo + wi);
        float D = math::normalDistribGGX(glm::dot(wm, n), a2);
        float G2 = math::SmithG2(roughness, cosO, cosI);
        glm::vec3 F = math::FresnelSchilick(albedo, glm::dot(wo, wm));
        glm::vec3 ret = F * D * G2 / glm::max(4 * cosO * cosI, 1e-8f);
        volatile float r1 = ret.r, r2 = ret.g, r3 = ret.b;
        return F * D * G2 / glm::max(4 * cosO * cosI, 1e-8f);
    }

    __host__ __device__ float microfacetPDF(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& wi)
    {
        float a2 = roughness * roughness;
        float cosO = glm::dot(n, wo), cosI = glm::dot(n, wi);
        glm::vec3 wm = glm::normalize(wo + wi);
        float D = math::normalDistribGGX(glm::dot(wm, n), a2);
        float G1 = math::SmithG1(roughness, cosO);
        return G1 * D / glm::max(4 * glm::dot(wo, n), 1e-8f);
    }

    __host__ __device__ void microfacetScatterSample(const glm::vec3& n, const glm::vec3& wo, scatter_record& srec, Sampler& sampler)
    {
        volatile float n1 = n.x, n2 = n.y, n3 = n.z;
        glm::vec3 wm = math::sampleNormalGGX(n, -1.f * wo, roughness, sample2D(sampler));
        volatile float newN1 = wm.x, newN2 = wm.y, newN3 = wm.z;
        srec.dir = glm::reflect(wo, wm);
        if (glm::dot(srec.dir, n) * glm::dot(-1.f * wo, n) < 0)
        {
            srec.bsdf = glm::vec3(0);
            srec.pdf = 0;
            return;
        }
        srec.bsdf = microfacetBSDF(n, -1.f * wo, srec.dir);
        volatile float c1 = srec.bsdf.r, c2 = srec.bsdf.g, c3 = srec.bsdf.b;
        srec.pdf = microfacetPDF(n, -1.f * wo, srec.dir);
        volatile float pdf1 = srec.pdf;
    }

    __host__ __device__ bool scatterSample(const glm::vec3 &n, const glm::vec3 &wo, scatter_record &srec, Sampler &sampler)
    {
        switch (type)
        {
        case Material::Lambertian:
            lambertianScatterSample(n, wo, srec, sampler);
            return true;
            break;
        case Material::MetallicWorkflow:
            break;
        case Material::Dielectric:
            dielectricScatterSample(n, wo, srec, sampler);
            return true;
            break;
        case Material::Microfacet:
            microfacetScatterSample(n, wo, srec, sampler);
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
};