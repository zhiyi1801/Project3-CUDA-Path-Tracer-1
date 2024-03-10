#pragma once
#include "glm/glm.hpp"
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.0001f
#define RAY_BIAS          0.001f
#define InvPI             1.f/PI

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define USE_BVH 1
#define USE_SAH 1
#define USE_MTBVH 1
#define TONEMAPPING 1
#define VERTEX_NORMAL 1
#define SHOW_NORMAL 0
#define ROUGHNESS_MIN 1e-3
#define ROUGHNESS_MAX 1.f

using Sampler = thrust::default_random_engine;
using color = glm::vec3;

//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
__host__ __device__ inline glm::vec3 ACESFilm(glm::vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), glm::vec3(0), glm::vec3(1));
}

__host__ __device__ inline glm::vec3 gammaCorrection(glm::vec3 x)
{
    return glm::pow(x, glm::vec3(1 / 2.2f));
}

__host__ __device__  inline static float sample1D(Sampler& sampler, float min = 0, float max = 1) {
    return thrust::uniform_real_distribution<float>(min, max)(sampler);
}

__host__ __device__  inline static glm::vec2 sample2D(Sampler& sampler, float min = 0, float max = 1) {
    return glm::vec2(sample1D(sampler, min, max), sample1D(sampler, min, max));
}

__host__ __device__  inline static glm::vec3 sample3D(Sampler& sampler, float min = 0, float max = 1) {
    return glm::vec3(sample2D(sampler, min, max), sample1D(sampler, min, max));
}

__host__ __device__  inline static glm::vec4 sample4D(Sampler& sampler, float min = 0, float max = 1) {
    return glm::vec4(sample3D(sampler, min, max), sample1D(sampler, min, max));
}

__host__ __device__  inline static glm::vec2 random2D(const glm::vec3 &w, int ite) {
    float r1 = glm::fract(glm::sin(glm::dot(w, glm::vec3(12.9898, 78.233, 45.645)) + ite * 25.345) * 43758.5453);
    float r2 = glm::fract(glm::sin(glm::dot(w, glm::vec3(45.432, 234.233, 99.99)) + ite * 42.345) * 219.23);
    return glm::vec2(r1, r2);
}

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}

namespace math {void localRefMatrix_Pixar(const glm::vec3& n, glm::vec3& xp, glm::vec3& yp); };

/// <summary>
/// An orthonormal basis (ONB)
/// </summary>
class ONB {
public:
    __host__ __device__
        ONB() {}

    __host__ __device__
        glm::vec3 operator[](int i) const { return axis[i]; }
    __host__ __device__
        glm::vec3& operator[](int i) { return axis[i]; }

    __host__ __device__
        glm::vec3 u() const { return axis[0]; }

    __host__ __device__
        glm::vec3 v() const { return axis[1]; }

    __host__ __device__
        glm::vec3 w() const { return axis[2]; }

    __host__ __device__
        glm::vec3 local(float a, float b, float c) const {
        return a * u() + b * v() + c * w();
    }

    __host__ __device__
        glm::vec3 local(const glm::vec3& a) const {
        return a.x * u() + a.y * v() + a.z * w();
    }

    __host__ __device__
        void build_from_w(const glm::vec3& w) {
        glm::vec3 unit_w = glm::normalize(w);
        glm::vec3 a = (fabs(unit_w.x) > 0.9) ? glm::vec3(0.f, 1.f, 0.f) : glm::vec3(1.f, 0.f, 0.f);
        glm::vec3 v = glm::normalize(cross(unit_w, a));
        glm::vec3 u = cross(v, unit_w);
        axis[0] = u;
        axis[1] = v;
        axis[2] = unit_w;
    }

    __host__ __device__
        void build_from_w_Pixar(const glm::vec3& w) {
        glm::vec3 u, v;
        glm::vec3 unit_w = glm::normalize(w);
        math::localRefMatrix_Pixar(w, u, v);

        volatile float u1 = u.x, u2 = u.y, u3 = u.z, v1 = v.x, v2 = v.y, v3 = v.z;

        axis[0] = u;
        axis[1] = v;
        axis[2] = unit_w;
    }

public:
    glm::vec3 axis[3];
};

namespace math
{
    template <typename T>
    __host__ __device__ inline T Sqr(T v) { return v * v; }

    __host__ __device__ inline float Lerp(float x, float a, float b) {
        return (1 - x) * a + x * b;
    }

    __host__ __device__ static glm::mat3 localRefMatrix2(glm::vec3 n) {
        glm::vec3 t = (glm::abs(n.y) > 0.9999f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        glm::vec3 b = glm::normalize(glm::cross(n, t));
        t = glm::cross(b, n);
        return glm::mat3(t, b, n);
    }

    __host__ __device__ static void localRefMatrix(const glm::vec3& n, glm::vec3& xp, glm::vec3& yp) {
        glm::vec3 t = (glm::abs(n.x) > 0.9f) ? glm::vec3(0.f, 1.f, 0.f) : glm::vec3(1.f, 0.f, 0.f);
        glm::vec3 b = glm::normalize(glm::cross(n, t));
        t = glm::cross(b, n);
        xp = t;
        yp = b;
    }

    __host__ __device__ static glm::mat3 localRefMatrix(const glm::vec3 &n) {
        glm::vec3 t, b;
        localRefMatrix(n, t, b);
        return glm::mat3(t, b, n);
    }

    //https://marc-b-reynolds.github.io/quaternions/2016/07/06/Orthonormal.html#fn:frisvad
    __host__ __device__ static void localRefMatrix_Pixar(const glm::vec3& n, glm::vec3& xp, glm::vec3& yp)
    {
        // -- snip start --
        float x = n.x, y = n.y, z = n.z;
        float sz = z >= 0 ? 1.f : -1.f;
        float a = 1.0f / (sz + z);
        // -- snip end --
        float sx = sz * x;     // shouldn't be needed but is for gcc & clang, saves 2 issues
        float b = x * y * a;

        xp = glm::vec3(sx * x * a - 1.f, sz * b, sx);  // {z+y/(z+1),   -xy/(z+1), -x}
        yp = glm::vec3(b, y * y * a - sz, y);  // {-xy/(z+1), 1-y^2/(z+1), -y}
    }

    __host__ __device__ static glm::mat3 localRefMatrix_Pixar(const glm::vec3& n)
    {
        glm::vec3 t, b;
        localRefMatrix_Pixar(n, t, b);
        return glm::mat3(t, b, n);
    }

    __host__ __device__ inline glm::vec2 sphere2Plane(const glm::vec3 &dir) {
        return glm::vec2(
            glm::fract(glm::atan(dir.z, dir.x) * InvPI * .5f + 1.f),
            glm::max(glm::atan(dir.y, glm::length(glm::vec2(dir.x, dir.z))) * InvPI + 0.5f, 0.f)
        );
    }

    //https://www.cs.princeton.edu/~funk/tog02.pdf
    //return bary centric coords glm::vec2(u, v)
    __host__ __device__ inline glm::vec2 sampleTriangleUniform(glm::vec2 r)
    {
        float t = sqrt(r.x);
        return glm::vec2(1.f - t, t * (1.f - r.y));
    }

    __host__ __device__
        inline glm::vec3 sampleHemisphereCosine(const glm::vec3& n, glm::vec2 r)
    {
        ONB uvw;
        uvw.build_from_w_Pixar(n);
        float r1 = r.x, r2 = r.y;
        float sinTheta = glm::sqrt(r1), cosTheta = sqrt(1 - r1);
        float phi = TWO_PI * r2;
        glm::vec3 xyz(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);
        volatile float f1 = xyz.x, f2 = xyz.y, f3 = xyz.z;
        return uvw.local(xyz);
    }

    __host__ __device__ static glm::vec2 toConcentricDisk(float x, float y) {
        float r = glm::sqrt(x);
        float theta = y * PI * 2.0f;
        return glm::vec2(glm::cos(theta), glm::sin(theta)) * r;
    }

    __host__ __device__
        inline glm::vec3 sampleHemisphereCosine2(const glm::vec3& n, glm::vec2 r)
    {
        glm::vec2 d = toConcentricDisk(r.x, r.y);
        float z = glm::sqrt(1.f - glm::dot(d, d));
        return localRefMatrix2(n) * glm::vec3(d, z);
    }

    __host__ __device__
        inline glm::vec3 sampleHemisphereUniform(const glm::vec3& n, Sampler& sampler)
    {
        ONB uvw;
        uvw.build_from_w_Pixar(n);
        float r1 = sample1D(sampler), r2 = sample1D(sampler);
        float cosTheta = 1 - r1, sinTheta = glm::sqrt(1 - cosTheta * cosTheta);
        float phi = TWO_PI * r2;
        glm::vec3 xyz(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);

        return uvw.local(xyz);
    }

    __host__ __device__  inline float sin_cos_convert(float t)
    {
        return sqrt(std::max(1 - t * t, 0.f));
    }

    __host__ __device__  inline float pow5(float x)
    {
        float x2 = x * x;
        return x2 * x2 * x;
    }

    __host__ __device__  inline float processNAN(float x)
    {
        return (x != x || isinf(x)) ? 0 : x;
    }

    __host__ __device__  inline glm::vec3 processNAN(const glm::vec3 &v)
    {
        return glm::vec3(processNAN(v.x), processNAN(v.y), processNAN(v.z));
    }

    __host__ __device__  inline float lengthSquared(const glm::vec3 &v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    __host__ __device__  inline glm::vec3 getReflectDir(const glm::vec3& n, const glm::vec3& wo)
    {
        glm::vec3 normal = glm::dot(wo, n) < 0 ? n : -n;
        return wo - 2.f * (normal * glm::dot(wo, normal));
    }

    __host__ __device__ inline  glm::vec2 sampleUniformDisc(glm::vec2 r)
    {
        float radius = glm::sqrt(r[0]);
        float theta = TWO_PI * r[1];
        return glm::vec2(radius * glm::cos(theta), radius * glm::sin(theta));
    }

    /// <summary>
    /// compute refract dir from ior1 to ior2
    /// </summary>
    /// <param name="n">surface normal</param>
    /// <param name="wo"></param>
    /// <param name="ior1"></param>
    /// <param name="ior2"></param>
    /// <returns></returns>
    __host__ __device__  inline glm::vec3 getRefractDir(const glm::vec3& n, const glm::vec3& wo, float ior1, float ior2)
    {
        glm::vec3 normal = glm::dot(wo, n) < 0 ? n : -n;

        glm::vec3 r_perp = (wo - glm::dot(wo, normal) * normal) * (ior1 / ior2);
        glm::vec3 r_para = -1 * sqrt(1 - lengthSquared(r_perp)) * normal;
        glm::vec3 r = r_para + r_perp;

        return r_perp + r_para;
    }

    __host__ __device__  inline float FresnelSchilick(float f0, float cosTheta)
    {
        return f0 + (1 - f0) * pow5(1 - cosTheta);
    }

    __host__ __device__  inline glm::vec3 FresnelSchilick(const glm::vec3 &f0, float cosTheta)
    {
        return f0 + (glm::vec3(1.f) - f0) * pow5(1 - cosTheta);
    }

    __host__ __device__  inline float FresnelMaxwell(float cosTheta1, float ior1, float ior2)
    {
        float sinTheta1 = glm::sqrt(glm::max(1 - cosTheta1 * cosTheta1, 0.f));
        float sinTheta2 = sinTheta1 * ior1 / ior2;
        if (sinTheta2 > 1) return 1;
        float cosTheta2 = glm::sqrt(glm::max(1 - sinTheta2 * sinTheta2, 0.f));
        float r_para = (ior1 * cosTheta2 - ior2 * cosTheta1) / (ior1 * cosTheta2 + ior2 * cosTheta1);
        float r_perp = (ior1 * cosTheta1 - ior2 * cosTheta2) / (ior1 * cosTheta1 + ior2 * cosTheta2);

        return (r_para * r_para + r_perp * r_perp) / 2.f;
    }

    /**
    * Sampling (Trowbridge-Reitz/GTR2/GGX) microfacet distribution, but only visible normals.
    * This reduces invalid samples and make pdf values at grazing angles more stable
    * See [Sampling the GGX Distribution of Visible Normals, Eric Heitz, JCGT 2018]:
    * https://jcgt.org/published/0007/04/01/
    * Note:
    */
    __host__ __device__  inline glm::vec3 sampleNormalGGX(const glm::vec3 &n, const glm::vec3 &wo, float alpha, const glm::vec2 &r)
    {
        volatile float r1 = r.x, r2 = r.y;
        glm::mat3 local2world = math::localRefMatrix_Pixar(n);
        glm::mat3 world2local = glm::transpose(local2world);

        glm::vec3 whL = glm::normalize(glm::vec3(alpha, alpha, 1.f) * (world2local * wo));
        if (whL.z < 0)
        {
            whL *= -1.f;
        }

        glm::vec3 T1 = (whL.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3 (0.f, 0.f, 1.f), whL))
            : glm::vec3(1.f, 0.f, 0.f);
        glm::vec3 T2 = glm::cross(whL, T1);

        glm::vec2 p = math::sampleUniformDisc(r);

        float h = glm::sqrt(1 - math::Sqr(p.x));
        p.y = math::Lerp((1 + whL.z) / 2.f, h, p.y);

        float pz = glm::sqrt(glm::max(0.f, 1.f - glm::dot(p, p)));
        glm::vec3 nh = p.x * T1 + p.y * T2 + pz * whL;

        return glm::normalize(local2world * glm::vec3(alpha * nh.x, alpha * nh.y,
                                        glm::max(1e-6f, nh.z)));
    }

    __host__ __device__ static glm::vec3 sampleNormalGGX2(glm::vec3 n, glm::vec3 wo, float alpha, glm::vec2 r) {
        volatile float r1 = r.x, r2 = r.y;
        glm::mat3 transMat = math::localRefMatrix2(n);
        glm::mat3 transInv = glm::inverse(transMat);

        glm::vec3 vh = glm::normalize((transInv * wo) * glm::vec3(alpha, alpha, 1.f));

        float lenSq = vh.x * vh.x + vh.y * vh.y;
        glm::vec3 t = lenSq > 0.f ? glm::vec3(-vh.y, vh.x, 0.f) / sqrt(lenSq) : glm::vec3(1.f, 0.f, 0.f);
        glm::vec3 b = glm::cross(vh, t);

        glm::vec2 p = math::sampleUniformDisc(r);
        float s = 0.5f * (vh.z + 1.f);
        p.y = (1.f - s) * glm::sqrt(1.f - p.x * p.x) + s * p.y;

        glm::vec3 h = t * p.x + b * p.y + vh * glm::sqrt(glm::max(0.f, 1.f - glm::dot(p, p)));
        h = glm::vec3(h.x * alpha, h.y * alpha, glm::max(0.f, h.z));
        return glm::normalize(transMat * h);
    }

    // Disney Appoximation of Smith term for GGX
    // [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
    // @a2: roughness^2 ? ((roughness + 1)/2)^2
    // @NoV: dot(normal, viewVector)
    // @NoL: dot(normal, lightVector)
    // Thanks to QianMo https://zhuanlan.zhihu.com/p/81708753
    // Disney's implementation: https://schuttejoe.github.io/post/ggximportancesamplingpart2/, https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2017/Presentations/Hammon_Earl_PBR_Diffuse_Lighting.pdf
    __host__ __device__  inline float SmithG2(float a2, float NoV, float NoL)
    {
        float denom = NoL * glm::sqrt(NoV * NoV * (1.f - a2) + a2) + NoV * glm::sqrt(NoL * NoL * (1.f - a2) + a2);
        float nom = 2.f * NoV * NoL;
        return nom / denom;
    }

    __host__ __device__  inline float SmithG1(float a2, float NoV)
    {
        float denom = glm::sqrt(NoV * NoV * (1 - a2) + a2) + NoV;
        float nom = 2 * NoV;
        return nom / denom;
    }

    // @cosTheta: dot(normal, wm), cosTheta of macroface normal and the sample normal
    __host__ __device__  inline float normalDistribGGX(float cosTheta, float a2)
    {
        if (cosTheta < 1e-6f) {
            return 0.f;
        }
        float nom = a2;
        float denom = cosTheta * cosTheta * (a2 - 1.f) + 1.f;
        denom = denom * denom * PI;
        return nom / denom;
    }

    __host__ __device__  inline float powerHeuristic(float fPdf, float gPdf)
    {
        float f = fPdf, g = gPdf;
        return (f * f) / (f * f + g * g);
    }

    __host__ __device__  inline float balanceHeuristic(float fPdf, float gPdf)
    {
        float f = fPdf, g = gPdf;
        return (f) / (f + g);
    }
}