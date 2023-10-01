#pragma once

#ifndef SCENE_H
#define SCENE_H

#include "tiny_obj_loader.hpp"
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "Bounds3.hpp"
#include "cudautils.hpp"
#include "tiny_obj_loader.hpp"

using namespace std;

//Triangle class is the primitive of a scene
class Triangle
{
public:
    glm::vec3 v[3];
    glm::vec3 n[3];
    glm::vec2 tex[3];
    float area;
    __host__
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

//We have to transit the data of scene in cpu to the devscene in gpu
class Scene; 

class DevScene
{
public:
    void initiate(const Scene& scene);
    void destroy();

    int tri_num;
    Triangle* dev_triangles;
};

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(const string& filename);
    void setDevData();
    void clear();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    DevScene tempDevScene;
    DevScene* dev_scene;
    RenderState state;
};


class MeshData;

namespace Resource
{
    MeshData* loadObj(const string& filename);
    void clear();

    extern int meshCount;
    extern std::vector<MeshData*> meshDataPool;
    extern std::map<std::string, int> meshDataIdx;
}

//Meshdata contains data of an obj or other 3d model file, it is prototype of Geometry, area and bounding box is used to calculate BVH tree and intersections.
class MeshData
{
public:
    float area;
    Bounds3 boundingBox;
    std::vector<Triangle> triangles;
};

#endif // SCENE_H