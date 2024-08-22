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

    __host__ __device__ glm::vec3 dielectricBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        return glm::vec3(0.f);
    }

    __host__ __device__ float dielectricPDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi) {
        return 0.f;
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

        float FresnelRefl = math::FresnelMaxwell(glm::abs(glm::dot(wo, n)), ior1, ior2);

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

    /*From CIS 561*/
    __host__ __device__ float AbsDot(glm::vec3 a, glm::vec3 b) {
        return glm::abs(dot(a, b));
    }

    __host__ __device__ float CosTheta(glm::vec3 w) { return w.z; }
    __host__ __device__ float Cos2Theta(glm::vec3 w) { return w.z * w.z; }
    __host__ __device__ float AbsCosTheta(glm::vec3 w) { return glm::abs(w.z); }
    __host__ __device__ float Sin2Theta(glm::vec3 w) {
        return max(0.f, 1.f - Cos2Theta(w));
    }
    __host__ __device__ float SinTheta(glm::vec3 w) { return glm::sqrt(Sin2Theta(w)); }
    __host__ __device__ float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }

    __host__ __device__  float Tan2Theta(glm::vec3 w) {
        return Sin2Theta(w) / Cos2Theta(w);
    }

    __host__ __device__  float CosPhi(glm::vec3 w) {
        float sinTheta = SinTheta(w);
        return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
    }
    __host__ __device__  float SinPhi(glm::vec3 w) {
        float sinTheta = SinTheta(w);
        return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
    }
    __host__ __device__  float Cos2Phi(glm::vec3 w) { return CosPhi(w) * CosPhi(w); }
    __host__ __device__  float Sin2Phi(glm::vec3 w) { return SinPhi(w) * SinPhi(w); }

    __host__ __device__ bool SameHemisphere(glm::vec3 w, glm::vec3 wp) {
        return w.z * wp.z > 0;
    }

    __host__ __device__ glm::vec3 Sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness) {
        glm::vec3 wh;

        float cosTheta = 0;
        float phi = TWO_PI * xi[1];
        // We'll only handle isotropic microfacet materials
        float tanTheta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
        cosTheta = 1 / sqrt(1 + tanTheta2);

        float sinTheta =
            sqrt(max(0.f, 1.f - cosTheta * cosTheta));

        wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        if (!SameHemisphere(wo, wh)) wh = -wh;

        return wh;
    }

    __host__ __device__ float TrowbridgeReitzD(const glm::vec3 &wh, float roughness) {
        float tan2Theta = Tan2Theta(wh);
        if (isinf(tan2Theta)) return 0.f;

        float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

        float e =
            (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) / (roughness * roughness)) *
            tan2Theta;
        return 1 / (PI * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
    }

    __host__ __device__ float Lambda(glm::vec3 w, float roughness) {
        float absTanTheta = glm::abs(TanTheta(w));
        if (isinf(absTanTheta)) return 0.;

        // Compute alpha for direction w
        float alpha =
            sqrt(Cos2Phi(w) * roughness * roughness + Sin2Phi(w) * roughness * roughness);
        float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
        return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
    }

    __host__ __device__ float TrowbridgeReitzG(glm::vec3 wo, glm::vec3 wi, float roughness) {
        return 1 / (1 + Lambda(wo, roughness) + Lambda(wi, roughness));
    }

    __host__ __device__ float TrowbridgeReitzPdf(const glm::vec3& wo, const glm::vec3& wh, float roughness) {
        return TrowbridgeReitzD(wh, roughness) * AbsCosTheta(wh);
    }

    __host__ __device__ glm::vec3 f_microfacet_refl(glm::vec3 albedo, glm::vec3 wo, glm::vec3 wi, float roughness) {
        float cosThetaO = AbsCosTheta(wo);
        float cosThetaI = AbsCosTheta(wi);
        glm::vec3 wh = wi + wo;
        // Handle degenerate cases for microfacet reflection
        if (cosThetaI == 0 || cosThetaO == 0) return glm::vec3(0.f);
        if (wh.x == 0 && wh.y == 0 && wh.z == 0) return glm::vec3(0.f);
        wh = normalize(wh);
        // TODO: Handle different Fresnel coefficients
        glm::vec3 F = glm::vec3(1.);//fresnel->Evaluate(glm::dot(wi, wh));
        float D = TrowbridgeReitzD(wh, roughness);
        float G = TrowbridgeReitzG(wo, wi, roughness);
        return albedo * D * G * F /
            (4 * cosThetaI * cosThetaO);
    }

    __host__ __device__ void Sample_f_microfacet_refl(const glm::vec3& n, const glm::vec3& woW, scatter_record& srec, Sampler& sampler, const glm::vec2& uv) {
        glm::mat3 local2world = math::localRefMatrix_Pixar(n);
        glm::mat3 world2local = glm::transpose(local2world);
        float sampleRoughness = glm::clamp(roughnessSampler.linearSample(uv).x, static_cast<float>(ROUGHNESS_MIN), ROUGHNESS_MAX);
        glm::vec3 sampleAlbedo = albedoSampler.linearSample(uv);
        glm::vec3 wo = world2local * woW;
        if (wo.z == 0)
        {
            srec.pdf = 0;
            return;
        }
        wo = -wo;
        volatile float wo1 = wo.x, wo2 = wo.y, wo3 = wo.z;

        glm::vec3 wh = Sample_wh(wo, sample2D(sampler), sampleRoughness);
        volatile float wh1 = wh.x, wh2 = wh.y, wh3 = wh.z;
        glm::vec3 wi = reflect(-wo, wh);
        volatile float wi1 = wi.x, wi2 = wi.y, wi3 = wi.z;
        srec.dir = local2world * wi;
        if (!SameHemisphere(wo, wi))
        {
            srec.pdf = 0;
            return;
        }

        // Compute PDF of _wi_ for microfacet reflection
        srec.pdf = TrowbridgeReitzPdf(wo, wh, sampleRoughness) / (4 * dot(wo, wh));
        srec.bsdf = f_microfacet_refl(sampleAlbedo, wo, wi, sampleRoughness);
        return;
    }
    /*From CIS 561*/

    __host__ __device__ glm::vec3 microfacetBSDF(const glm::vec3& n, const glm::vec3& wo, const glm::vec3 &wi, const glm::vec3 &sampleAlbedo, float sampleRoughness)
    {
        float a2 = sampleRoughness * sampleRoughness;
        float cosO = glm::dot(n, wo), cosI = glm::dot(n, wi);
        if (cosO * cosI < 1e-7)
        {
            return glm::vec3(0.f);
        }
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
        float sampleRoughness = glm::clamp(roughnessSampler.linearSample(uv).x, static_cast<float>(ROUGHNESS_MIN), ROUGHNESS_MAX);
        glm::vec3 sampleAlbedo = albedoSampler.linearSample(uv);
        volatile float c1 = sampleAlbedo.x, c2 = sampleAlbedo.y, c3 = sampleAlbedo.z;
        volatile float wm1 = 1, wm2 = 1, wm3 = 1;

        glm::vec3 wm = math::sampleNormalGGX(n, -1.f * wo, sampleRoughness, sample2D(sampler));
        srec.dir = glm::reflect(wo, wm);
        wm1 = wm.x, wm2 = wm.y, wm3 = wm.z;
        if (glm::dot(srec.dir, n) * glm::dot(-1.f * wo, n) < 0)
        {
            srec.bsdf = glm::vec3(0);
            srec.pdf = 0;
            return;
        }

        srec.bsdf = microfacetBSDF(n, -1.f * wo, srec.dir, sampleAlbedo, sampleRoughness);
        srec.pdf = microfacetPDF(n, -1.f * wo, srec.dir, sampleRoughness);
        volatile float b1 = srec.bsdf.x, b2 = srec.bsdf.y, b3 = srec.bsdf.z;
        volatile float p = srec.pdf;
        return;
    }

    __host__ __device__ glm::vec3 metallicBSDF(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& sampleAlbedo, float sampleRoughness, float sampleMetallic)
    {
        float a2 = sampleRoughness * sampleRoughness;
        float cosO = glm::dot(n, wo), cosI = glm::dot(n, wi);
        if (cosO * cosI < 1e-7)
        {
            return glm::vec3(0.f);
        }
        glm::vec3 wm = glm::normalize(wo + wi);
        float D = math::normalDistribGGX(glm::dot(wm, n), a2);
        float G2 = math::SmithG2(sampleRoughness, cosO, cosI);
        glm::vec3 F = math::FresnelSchilick(glm::mix(glm::vec3(0.08f), sampleAlbedo, sampleMetallic), glm::dot(wo, wm));

        return glm::mix((1.f - sampleMetallic) * sampleAlbedo * InvPI, glm::vec3(D * G2 / glm::max(4 * cosO * cosI, 1e-8f)), F);
    }
    
    __host__ __device__ float metallicPDF(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& wi, float sampleRoughness, float sampleMetallic)
    {
        float a2 = sampleRoughness * sampleRoughness;
        float cosO = glm::dot(n, wo), cosI = glm::dot(n, wi);
        glm::vec3 wm = glm::normalize(wo + wi);
        float D = math::normalDistribGGX(glm::dot(wm, n), a2);
        float G1 = math::SmithG1(sampleRoughness, cosO);
        return glm::mix(glm::dot(wi, n) * InvPI, G1 * D / glm::max(4 * glm::dot(wo, n), 1e-8f), 1.f / (2.f - sampleMetallic));
    }

    __host__ __device__ void metallicScatterSample(const glm::vec3& n, const glm::vec3& wo, scatter_record& srec, Sampler& sampler, const glm::vec2& uv)
    {
        float sampleRoughness = glm::clamp(roughnessSampler.linearSample(uv).x, static_cast<float>(ROUGHNESS_MIN), ROUGHNESS_MAX);
        float sampleMetallic = glm::clamp(metallicSampler.linearSample(uv).x, 0.f, 1.f);
        glm::vec3 sampleAlbedo = albedoSampler.linearSample(uv);
        float r = sample1D(sampler);
        volatile float c1 = sampleAlbedo.x, c2 = sampleAlbedo.y, c3 = sampleAlbedo.z;
        volatile float wm1 = 1, wm2 = 1, wm3 = 1;
        volatile float uv1 = uv.x, uv2 = uv.y;
        volatile float t = glm::dot(-1.f * wo, n);
        volatile float n1 = n.x, n2 = n.y, n3 = n.z;

        if (r < 1.f / (2.f - sampleMetallic))
        {
            glm::vec3 wm = math::sampleNormalGGX(n, -1.f * wo, sampleRoughness * sampleRoughness, sample2D(sampler));
            srec.dir = glm::reflect(wo, wm);
            wm1 = wm.z, wm1 = wm.x, wm2 = wm.y, wm3 = wm.z;
        }
        else
        {
            srec.dir = math::sampleHemisphereCosine(n, sample2D(sampler));
        }

        if (glm::dot(-1.f * wo, n) < 0 || glm::dot(srec.dir, n) < 0)
        {
            srec.bsdf = glm::vec3(0);
            srec.pdf = 0;
            return;
        }

        srec.bsdf = metallicBSDF(n, -1.f * wo, srec.dir, sampleAlbedo, sampleRoughness, sampleMetallic);
        srec.pdf = metallicPDF(n, -1.f * wo, srec.dir, sampleRoughness, sampleMetallic);
        volatile float b1 = srec.bsdf.x, b2 = srec.bsdf.y, b3 = srec.bsdf.z;
        volatile float p = srec.pdf;
        return;
    }

    __host__ __device__ static float schlickG(float cosTheta, float alpha) {
        float a = alpha * .5f;
        return cosTheta / (cosTheta * (1.f - a) + a);
    }

    __host__ __device__ inline float smithG(float cosWo, float cosWi, float alpha) {
        return schlickG(glm::abs(cosWo), alpha) * schlickG(glm::abs(cosWi), alpha);
    }

    __host__ __device__ static float GTR2Distrib(float cosTheta, float alpha) {
        if (cosTheta < 1e-6f) {
            return 0.f;
        }
        float aa = alpha * alpha;
        float nom = aa;
        float denom = cosTheta * cosTheta * (aa - 1.f) + 1.f;
        denom = denom * denom * PI;
        return nom / denom;
    }

    __host__ __device__ static float GTR2Pdf(glm::vec3 n, glm::vec3 m, glm::vec3 wo, float alpha) {
        return GTR2Distrib(glm::dot(n, m), alpha) * schlickG(glm::dot(n, wo), alpha) *
            glm::dot(m, wo) / glm::dot(n, wo);
    }

    __host__ __device__ glm::vec3 metallicBSDF2(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& sampleAlbedo, float sampleRoughness, float sampleMetallic)
    {
        float alpha = sampleRoughness * sampleRoughness;
        float cosO = glm::dot(n, wo), cosI = glm::dot(n, wi);
        glm::vec3 h = glm::normalize(wo + wi);
        if (cosO * cosI < 1e-7)
        {
            return glm::vec3(0.f);
        }
        volatile float c1 = sampleAlbedo.x, c2 = sampleAlbedo.y, c3 = sampleAlbedo.z;
        volatile float f1 = glm::mix(glm::vec3(.08f), sampleAlbedo, sampleMetallic).x, f2 = glm::mix(glm::vec3(.08f), sampleAlbedo, sampleMetallic).y, f3 = glm::mix(glm::vec3(.08f), sampleAlbedo, sampleMetallic).z;
        glm::vec3 f = math::FresnelSchilick(glm::mix(glm::vec3(.08f), sampleAlbedo, sampleMetallic), glm::dot(h, wo));
        float g = smithG(cosO, cosI, alpha);
        float d = GTR2Distrib(glm::dot(n, h), alpha);

        return glm::mix(sampleAlbedo * InvPI * (1.f - sampleMetallic), glm::vec3(g * d / (4.f * cosI * cosO)), f);
    }

    __host__ __device__ float metallicPDF2(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& wi, float sampleRoughness, float sampleMetallic)
    {
        glm::vec3 h = glm::normalize(wo + wi);
        return glm::mix(
            glm::max(glm::dot(n, wi), 0.f) * InvPI,
            GTR2Pdf(n, h, wo, sampleRoughness * sampleRoughness) / (4.f * glm::max(glm::dot(h, wo), 0.f)),
            1.f / (2.f - sampleMetallic)
        );
    }

    __host__ __device__ void metallicScatterSample2(const glm::vec3& n, const glm::vec3& wo, scatter_record& srec, Sampler& sampler, const glm::vec2& uv)
    {
        float sampleRoughness = glm::clamp(roughnessSampler.linearSample(uv).x, static_cast<float>(ROUGHNESS_MIN), ROUGHNESS_MAX);
        float sampleMetallic = glm::clamp(metallicSampler.linearSample(uv).x, 0.f, 1.f);
        glm::vec3 sampleAlbedo = albedoSampler.linearSample(uv);
        float r = sample1D(sampler);
        volatile float c1 = sampleAlbedo.x, c2 = sampleAlbedo.y, c3 = sampleAlbedo.z;
        volatile float wm1 = 1, wm2 = 1, wm3 = 1;
        volatile float uv1 = uv.x, uv2 = uv.y;
        volatile float t = glm::dot(-1.f * wo, n);
        volatile float n1 = n.x, n2 = n.y, n3 = n.z;
        volatile float wo1 = wo.x, wo2 = wo.y, wo3 = wo.z;

        if (r < 1.f / (2.f - sampleMetallic))
        {
            glm::vec3 wm = math::sampleNormalGGX2(n, -1.f * wo, sampleRoughness * sampleRoughness, sample2D(sampler));
            srec.dir = glm::reflect(wo, wm);
            wm1 = wm.z, wm1 = wm.x, wm2 = wm.y, wm3 = wm.z;
        }
        else
        {
            srec.dir = math::sampleHemisphereCosine2(n, sample2D(sampler));
        }

        if (glm::dot(-1.f * wo, n) < 0 || glm::dot(srec.dir, n) < 0)
        {
            srec.bsdf = glm::vec3(0);
            srec.pdf = 0;
            return;
        }

        srec.bsdf = metallicBSDF2(n, -1.f * wo, srec.dir, sampleAlbedo, sampleRoughness, sampleMetallic);
        srec.pdf = metallicPDF2(n, -1.f * wo, srec.dir, sampleRoughness, sampleMetallic);
        volatile float b1 = srec.bsdf.x, b2 = srec.bsdf.y, b3 = srec.bsdf.z;
        volatile float p = srec.pdf;
        return;
    }

    __host__ __device__ bool scatterSample(const ShadeableIntersection &intersection, const glm::vec3 &wo, scatter_record &srec, Sampler &sampler)
    {
        glm::vec2 uv = intersection.texCoords;
        glm::vec3 n = intersection.surfaceNormal;
        switch (type)
        {
        case Material::Lambertian:
            lambertianScatterSample(n, wo, srec, sampler, uv);
            return true;
            break;
        case Material::MetallicWorkflow:
            metallicScatterSample(n, wo, srec, sampler, uv);
            return true;
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

    __host__ __device__ glm::vec3 BSDF(const ShadeableIntersection& intersection, const glm::vec3& wo, const glm::vec3& wi) {
        glm::vec2 uv = intersection.texCoords;
        glm::vec3 n = intersection.surfaceNormal;
        float sampleRoughness = glm::clamp(roughnessSampler.linearSample(uv).x, static_cast<float>(ROUGHNESS_MIN), ROUGHNESS_MAX);
        float sampleMetallic = glm::clamp(metallicSampler.linearSample(uv).x, 0.f, 1.f);
        glm::vec3 sampleAlbedo = albedoSampler.linearSample(uv);

        switch (type) {
        case Material::Type::Lambertian:
            return lambertianBSDF(n, wo, wi, uv);
        case Material::Type::Microfacet:
            return microfacetBSDF(n, -1.f * wo, wi, sampleAlbedo, sampleRoughness);
        case Material::Type::MetallicWorkflow:
            return metallicBSDF(n, -1.f * wo, wi, sampleAlbedo, sampleRoughness, sampleMetallic);
        case Material::Type::Dielectric:
            return dielectricBSDF(n, wo, wi);
        }
        return glm::vec3(0.f);
    }

    __host__ __device__ float pdf(const ShadeableIntersection& intersection, const glm::vec3& wo, const glm::vec3& wi) {
        glm::vec2 uv = intersection.texCoords;
        glm::vec3 n = intersection.surfaceNormal;
        float sampleRoughness = glm::clamp(roughnessSampler.linearSample(uv).x, static_cast<float>(ROUGHNESS_MIN), ROUGHNESS_MAX);
        float sampleMetallic = glm::clamp(metallicSampler.linearSample(uv).x, 0.f, 1.f);
        glm::vec3 sampleAlbedo = albedoSampler.linearSample(uv);

        switch (type) {
        case Material::Type::Lambertian:
            return lambertianPDF(n, wo, wi);
        case Material::Type::Microfacet:
            return microfacetPDF(n, -1.f * wo, wi, sampleRoughness);
        case Material::Type::MetallicWorkflow:
            return metallicPDF(n, -1.f * wo, wi,sampleRoughness, sampleMetallic);
        case Material::Type::Dielectric:
            return dielectricPDF(n, wo, wi);
        }
        return 0;
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