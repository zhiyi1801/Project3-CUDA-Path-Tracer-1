#pragma once

#include<limits>
#include<array>
#include<vector>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "sceneStructs.h"

class scatter_record
{
public:
    glm::vec3 bsdf;
    float pdf;
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
        Disney,
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
        glm::dot(wo, n) < 0 ? (ior1 = 1, ior2 = ior) : (ior2 = 1, ior1 = ior);

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
        case Material::Disney:
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