#pragma once

#include "intersections.h"

enum BSDFType {
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR |
    BSDF_REFLECTION | BSDF_TRANSMISSION,
};

class BSDFSampler
{
public:
    glm::vec3 dir;
    glm::vec3 bsdf;
    float pdf;
    BSDFType type;
};

__host__ __device__
void getTBN(const glm::vec3& N, glm::vec3& T, glm::vec3& B)
{
    glm::vec3 directionNotNormal;
    if (abs(T.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(T.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }
    T = glm::cross(N, directionNotNormal);
    B = glm::cross(N, T);
}

__host__ __device__
glm::vec3 sampleHemisphereCosWeighted(const glm::vec3& n, const glm::vec2& random01)
{
    float up = sqrt(random01.x); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = random01.y * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(n.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(n.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(n, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(n, perpendicularDirection1));

    return up * n
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 sampleHemisphereCosWeightedTBN(const glm::vec2& random01)
{
    float up = sqrt(random01.x); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = random01.y * TWO_PI;


    return glm::vec3(cos(around) * over, sin(around) * over, up);
}

__host__ __device__
glm::vec3 sampleHemisphereUniformTBN(const glm::vec2& random01)
{
    float up = random01.x; // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = random01.y * TWO_PI;


    return glm::vec3(cos(around) * over, sin(around) * over, up);
}

__host__ __device__
glm::vec2 sampleUnitDisk(glm::vec2 random01)
{
    float r = sqrt(random01.x);
    float phi = random01.y * TWO_PI;

    return glm::vec2(r * cos(phi), r * sin(phi));
}

__host__ __device__
glm::vec3 sampleHemisphereCosWeightedDiskTBN(const glm::vec2& random01)
{
    glm::vec2 xy = sampleUnitDisk(random01);
    float z = sqrt(1 - glm::dot(xy, xy));
    return glm::vec3(xy, z);
}

class BSDFMaterial
{
public:
    enum BSDFMaterialType
    {
        Lambertian
    };

    BSDFMaterialType type = Lambertian;
    glm::vec3 baseColor{ 1.f };

    //Lambertian l(wi, wo, p)L(wi)cos(theta_i) = reflectcolor_o
    glm::vec3 BSDFLambertian(glm::vec3 n, glm::vec3 wo, glm::vec3 wi)
    {
        return baseColor * InvPI;
    }

    float pdfLambertianCosWeighted(glm::vec3 n, glm::vec3 wo, glm::vec3 wi)
    {
        return std::max(glm::dot(wi, n) * InvPI, .0f);
    }

    float pdfLambertianUniform(glm::vec3 n, glm::vec3 wo, glm::vec3 wi)
    {
        return InvPI/2;
    }


};



// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 reflDir;
    const float reflProb = 0.5;
    thrust::uniform_real_distribution<float> u01(0, 1);
    //if (m.hasReflective)
    //{
    //    float sample01 = u01(rng);
    //    if(sample01 > reflProb)
    //    {
    //        reflDir = glm::reflect(pathSegment.ray.direction, normal);
    //        pathSegment.ray.direction = reflDir;
    //        pathSegment.color *= m.specular.color * glm::dot(normal, reflDir)/(1-reflProb);
    //        pathSegment.ray.origin = intersect + 0.00001f * reflDir;
    //    }
    //    else
    //    {
    //        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    //        pathSegment.color *= m.color/reflProb;
    //        pathSegment.ray.origin = intersect + 0.00001f * pathSegment.ray.direction;
    //    }
    //}
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.color *= m.albedo;
    pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
}