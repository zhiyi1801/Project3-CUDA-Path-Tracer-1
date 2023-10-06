#pragma once
#ifndef GEOM_H
#define GEOM_H
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
//Triangle class is the primitive of a scene
class Triangle
{
public:
    glm::vec3 v[3];
    glm::vec3 n[3];
    glm::vec2 tex[3];
    float area;
    __host__ __device__
        Triangle(const std::array<glm::vec3, 3>& _v, const std::array<glm::vec3, 3>& _n, const std::array<glm::vec2, 3>& _tex)
    {
        for (int i = 0; i < 3; ++i)
        {
            v[i] = _v[i];
            n[i] = _n[i];
            tex[i] = _tex[i];
        }
        glm::vec3 e1 = v[1] - v[0];
        glm::vec3 e2 = v[2] - v[0];
        area = glm::length(glm::cross(e1, e2)) * 0.5f;
    }

    __host__ __device__
        Triangle(const glm::vec3* _v, const glm::vec3* _n, const glm::vec2* _tex)
    {
        for (int i = 0; i < 3; ++i)
        {
            v[i] = _v[i];
            n[i] = _n[i];
            tex[i] = _tex[i];
        }
        glm::vec3 e1 = v[1] - v[0];
        glm::vec3 e2 = v[2] - v[0];
        area = glm::length(glm::cross(e1, e2)) * 0.5f;
    }

    __host__ __device__
        Triangle()
        : v(), n(), tex(), area(0)
    { }

    __host__ __device__
        Bounds3 getBound() { return Union(Bounds3(v[0], v[1]), v[2]); }

    __host__ __device__
        bool getInterSect(const Ray& ray, float& t, float& u, float& v)
    {
        glm::vec3 e1 = this->v[1] - this->v[0];
        glm::vec3 e2 = this->v[2] - this->v[0];
        glm::vec3 normal = glm::cross(e1, e2);
        normal = glm::normalize(normal);

        glm::vec3 pvec = glm::cross(ray.direction, e2);
        float det = glm::dot(e1, pvec);
        if (det == 0) return false;

        float invDet = 1 / det;

        glm::vec3 tvec = ray.origin - this->v[0];
        u = glm::dot(tvec, pvec);
        u *= invDet;

        glm::vec3 qvec = glm::cross(tvec, e1);
        v = glm::dot(ray.direction, qvec);
        v *= invDet;

        t = glm::dot(e2, qvec);
        t *= invDet;

        if (t < 0 || u < 0 || v < 0 || (1 - u - v) < 0) return false;

        return true;
    }
};
#endif // GEOM_H
