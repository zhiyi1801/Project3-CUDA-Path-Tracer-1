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
#include "BVH.h";
#include "tiny_obj_loader.hpp"

using namespace std;

//We have to transit the data of scene in cpu to the devscene in gpu
class Scene; 

class DevScene
{
public:
    void initiate(const Scene& scene);
    void destroy();

    int tri_num;
    int bvh_size;
    Triangle* dev_triangles;
    GpuBVHNode* dev_gpuBVH;
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

    std::vector<GpuBVHNodeInfo> gpuBVHNodeInfos;
    std::vector<GpuBVHNode> gpuBVHNodes;
    RecursiveBVHNode* bvhRoot;

    BVHAccel bvhConstructor;
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
    //float area;
    //Bounds3 boundingBox;
    std::vector<Triangle> triangles;
};

#endif // SCENE_H