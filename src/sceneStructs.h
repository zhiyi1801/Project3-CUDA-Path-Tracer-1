#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"


class MeshData;
enum GeomType {
    SPHERE,
    CUBE,
    OBJ,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    __host__ __device__ glm::vec3 getPoint(float dist)
    {
        return origin + direction * dist;
    }
};

struct GPUGeom
{
    enum GeomType type;
    int materialid;
    glm::mat4 transform;
    GPUGeom(GeomType _type, int _materialid, const glm::mat4 _transform) :
        type(_type), materialid(_materialid), transform(_transform) {}
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    MeshData* mesh;
    //int protoId;
};

//struct Material {
//    glm::vec3 color;
//    struct {
//        float exponent;
//        glm::vec3 color;
//    } specular;
//    float hasReflective;
//    float hasRefractive;
//    float indexOfRefraction;
//    float emittance;
//};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 interPoint;
  glm::vec2 texCoords;
  int materialId;
};