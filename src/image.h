#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

using namespace std;

class image {
public:
    int width;
    int height;
    glm::vec3* pixels;


    image(int x, int y);
    image(std::string filePath, float gamma = 1.f);
    ~image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);

    size_t byteSize() const {
        return sizeof(glm::vec3) * width * height;
    }

    glm::vec3* data() const {
        return pixels;
    }

    glm::vec3 getValue(int x, int y) const
    {
        return pixels[y * width + x];
    }

    glm::vec3 linearSample(const glm::vec2& uv) const
    {
        float u = uv.x, v = uv.y;
        float x = u * (width - 1), y = v * (height - 1);
        int lx = x, ux = x + 1 >= width ? lx : lx + 1;
        int ly = y, uy = y + 1 >= height ? ly : ly + 1;

        float fx = glm::fract(x), fy = glm::fract(y);
        volatile float cll = getValue(lx, ly).x, cul = getValue(ux, ly).x, clu = getValue(lx, uy).x, cuu = getValue(ux, uy).x;
        glm::vec3 vll = getValue(lx, ly), vul = getValue(ux, ly), vlu = getValue(lx, uy), vuu = getValue(ux, uy);
        glm::vec3 p1 = glm::mix(getValue(lx, ly), getValue(ux, ly), fx);
        glm::vec3 p2 = glm::mix(getValue(lx, uy), getValue(ux, uy), fx);
        return glm::mix(p1, p2, fy);
    }
};

class devTexObj
{
public:
    int width, height;
    glm::vec3* data;

    devTexObj() = default;

    devTexObj(image* img, glm::vec3 *devData)
    {
        width = img->width;
        height = img->height;
        data = devData;
    }

    __host__ __device__ glm::vec3 getValue(int x, int y)
    {
        return data[y * width + x];
    }

    __host__ __device__ glm::vec3 linearSample(const glm::vec2 &uv)
    {
        float u = uv.x, v = uv.y;
        float x = u * (width - 1), y = v * (height - 1);
        int lx = x, ux = x + 1 >= width ? lx : lx + 1;
        int ly = y, uy = y + 1 >= height ? ly : ly + 1;

        float fx = glm::fract(x), fy = glm::fract(y);
        volatile float cll = getValue(lx, ly).x, cul = getValue(ux, ly).x, clu = getValue(lx, uy).x, cuu = getValue(ux, uy).x;
        glm::vec3 p1 = glm::mix(getValue(lx, ly), getValue(ux, ly), fx);
        glm::vec3 p2 = glm::mix(getValue(lx, uy), getValue(ux, uy), fx);
        return glm::mix(p1, p2, fy);
    }
};

class devTexSampler
{
public:
    devTexObj* tex;
    glm::vec3 fixVal;

    devTexSampler() :tex(nullptr), fixVal(0) {};

    devTexSampler(devTexObj *_tex) :tex(_tex) {}

    devTexSampler(const glm::vec3& val) :fixVal(val), tex(nullptr) {}

    devTexSampler(float val) :fixVal(glm::vec3(val)), tex(nullptr) {}

    __host__ __device__ glm::vec3 linearSample(const glm::vec2 &uv)
    {
        if (!this->tex)
        {
            return fixVal;
        }
        return tex->linearSample(uv);
    }
};