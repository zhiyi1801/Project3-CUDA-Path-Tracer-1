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
#define InvPI             1.f/PI

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define USE_BVH 1
#define USE_SAH 1
#define USE_MTBVH 1

using Sampler = thrust::default_random_engine;
using color = glm::vec3;

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
        glm::vec3 a = (fabs(unit_w.x) > 0.9) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 v = glm::normalize(cross(unit_w, a));
        glm::vec3 u = cross(unit_w, v);
        axis[0] = u;
        axis[1] = v;
        axis[2] = unit_w;
    }

public:
    glm::vec3 axis[3];
};

namespace math
{
    /// <summary>
    /// get a cosine weighted sample in a hemisphere with normal n
    /// </summary>
    /// <param name="n"> the normal direction </param>
    /// <returns> a vec3 direction sample </returns>
    __host__ __device__ 
    inline glm::vec3 sampleHemisphereCosine(const glm::vec3 &n, Sampler &sampler)
    {
        ONB uvw;
        uvw.build_from_w(n);
        float r1 = sample1D(sampler), r2 = sample1D(sampler);
        float sinTheta = glm::sqrt(r1), cosTheta = sqrt(1 - r1);
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

    __host__ __device__  inline float lengthSquared(const glm::vec3 &v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    __host__ __device__  inline glm::vec3 getReflectDir(const glm::vec3& n, const glm::vec3& wo)
    {
        glm::vec3 normal = glm::dot(wo, n) < 0 ? n : -n;
        return wo - 2.f * (normal * glm::dot(wo, normal));
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

        volatile float n1 = n.x, n2 = n.y, n3 = n.z, wo1 = wo.x, wo2 = wo.y, wo3 = wo.z, wi1 = r.x, wi2 = r.y, wi3 = r.z;

        return r_perp + r_para;
    }

    __host__ __device__  inline float FresnelSchilick(float f0, float cosTheta)
    {
        return f0 + (1 - f0) * pow5(1 - cosTheta);
    }

    __host__ __device__  inline float FresnelMaxwell(float cosTheta1, float ior1, float ior2)
    {
        float sinTheta1 = sqrt(std::max(1 - cosTheta1 * cosTheta1, 0.f));
        float sinTheta2 = sinTheta1 * ior1 / ior2;
        if (sinTheta2 > 1) return 1;
        float cosTheta2 = sqrt(std::max(1 - sinTheta2 * sinTheta2, 0.f));
        float r_para = (ior1 * cosTheta2 - ior2 * cosTheta1) / (ior1 * cosTheta2 + ior2 * cosTheta1);
        float r_perp = (ior1 * cosTheta1 - ior2 * cosTheta2) / (ior1 * cosTheta1 + ior2 * cosTheta2);

        return (r_para * r_para + r_perp * r_perp) / 2.f;
    }
}