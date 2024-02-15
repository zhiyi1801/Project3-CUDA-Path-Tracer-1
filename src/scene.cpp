#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

const std::map<std::string, Material::Type> materialTypeMap{
    {"Lambertian", Material::Type::Lambertian},
    {"MetallicWorkflow", Material::Type::MetallicWorkflow},
    {"Dielectric", Material::Type::Dielectric},
    {"Microfacet", Material::Type::Microfacet},
    {"Light", Material::Type::Light}
};

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

namespace Resource
{
    int meshCount(0);
    std::vector<MeshData*> meshDataPool;
    std::vector<image*> textureDataPool;
    std::map<std::string, int> meshDataIdx;
    std::map<std::string, int> textureDataIdx;
}

Scene::Scene(const string& filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }

}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;
        newGeom.mesh = nullptr;
        //newGeom.protoId = -1;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (line.find(".obj") != line.npos)
            {
                cout << "---Loading obj file " << line << "---" << endl;
                newGeom.type = OBJ;
                newGeom.mesh = Resource::loadObj(line, geoms.size());
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState& state = this->state;
    Camera& camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

/// <summary>
/// material has TYPE, ALBEDO, METALLIC, ROUGHNESS, IOR(index of refraction)
/// </summary>
/// <param name="materialid"></param>
/// <returns></returns>
int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    }
    else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 5; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "TYPE") == 0) {
                auto it = materialTypeMap.find(tokens[1]);
                if (it != materialTypeMap.end())
                {
                    newMaterial.type = it->second;
                }
                else
                {
                    cout << "TYPE ERROR " << id  << endl;
                }
            }
            else if (strcmp(tokens[0].c_str(), "ALBEDO") == 0) {
                newMaterial.albedo = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "METALLIC") == 0) {
                newMaterial.metallic = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
                newMaterial.roughness = glm::max(atof(tokens[1].c_str()), 0.001);
            }
            else if (strcmp(tokens[0].c_str(), "IOR") == 0) {
                newMaterial.ior = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

void Scene::setDevData()
{
    for (const auto& g : geoms)
    {
        GPUGeom gpuGeom(g.type, g.materialid, g.transform);
        gpuGeoms.push_back(gpuGeom);
        if (g.type != OBJ) continue;
        for (const auto& t : g.mesh->triangles)
        {
            Triangle temp_t;
            for (int i = 0; i < 3; ++i)
            {
                temp_t.v[i] = glm::vec3(g.transform * glm::vec4(t.v[i], 1.0f));
                temp_t.n[i] = glm::normalize(glm::vec3(g.invTranspose * glm::vec4(t.n[i], 0.0f)));
                temp_t.tex[i] = t.tex[i];
            }
            temp_t.geomIdx = gpuGeoms.size() - 1;
            triangles.push_back(temp_t);
        }
    }


    this->bvhRoot = this->bvhConstructor.recursiveBuild(this->triangles);
    this->bvhConstructor.recursiveBuildGpuBVHInfo(bvhRoot, this->gpuBVHNodeInfos);
#if USE_MTBVH
    this->bvhConstructor.buildGpuMTBVH(this->gpuBVHNodeInfos, this->gpuBVHNodes);
#else
    this->bvhConstructor.buildGpuBVH(this->gpuBVHNodeInfos, this->gpuBVHNodes);
#endif

    this->tempDevScene.initiate(*this);
    cudaMalloc(&dev_scene, sizeof(DevScene));
    cudaMemcpy(dev_scene, &tempDevScene, sizeof(DevScene), cudaMemcpyHostToDevice);
    checkCUDAError("dev_scene");

    bvhRoot->destroy();
    gpuBVHNodeInfos.clear();
}

MeshData* Resource::loadObj(const string& filename, const int _geomIdx)
{
    auto exist = meshDataIdx.find(filename);
    if (exist != meshDataIdx.end())
    {
        //protoId = exist->second;
        std::cout << "---obj file existed " << filename << "---" << std::endl;
        return meshDataPool[exist->second];
    }

    MeshData* model = new MeshData;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::string warn, err;
    //model->area = 0;

    std::cout << "---Loading model " << filename << "---" << std::endl;
    if (!tinyobj::LoadObj(&attrib, &shapes, nullptr, &warn, &err, filename.c_str()))
    {
        std::cout << "---Fail Error msg " << err << "---" << std::endl;
        return nullptr;
    }

    if (!warn.empty())
    {
        std::cout << "---Warn Error msg " << warn << "---" << std::endl;
    }
    bool hasTex = !attrib.texcoords.empty();

    glm::vec3 min_vert = glm::vec3{ std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity() };
    glm::vec3 max_vert = glm::vec3{ -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity() };

    for (const auto& shape : shapes)
    {
        size_t index_offset = 0;
        for (const auto& fn : shape.mesh.num_face_vertices)
        {
            for (int i = 0; i < fn - 2; ++i)
            {
                auto idx0 = shape.mesh.indices[index_offset], idx1 = shape.mesh.indices[index_offset + i + 1], idx2 = shape.mesh.indices[index_offset + i + 2];

                std::array<glm::vec3, 3> _v{ *((glm::vec3*)attrib.vertices.data() + idx0.vertex_index),
                                             *((glm::vec3*)attrib.vertices.data() + idx1.vertex_index),
                                             *((glm::vec3*)attrib.vertices.data() + idx2.vertex_index) };

                std::array<glm::vec3, 3> _n{ *((glm::vec3*)attrib.normals.data() + idx0.normal_index),
                                             *((glm::vec3*)attrib.normals.data() + idx1.normal_index),
                                             *((glm::vec3*)attrib.normals.data() + idx2.normal_index) };

                std::array<glm::vec2, 3> _tex;
                if (hasTex)
                {
                    _tex = std::array<glm::vec2, 3>{ *((glm::vec2*)attrib.texcoords.data() + idx0.texcoord_index),
                        * ((glm::vec2*)attrib.texcoords.data() + idx1.texcoord_index),
                        * ((glm::vec2*)attrib.texcoords.data() + idx2.texcoord_index) };
                }
                Triangle tri(_v, _n, _tex, _geomIdx);
                model->triangles.push_back(tri);
                //model->area += tri.area;
                for (int j = 0; j < 3; ++j)
                {
                    min_vert = glm::min(min_vert, tri.v[i]);
                    max_vert = glm::max(max_vert, tri.v[i]);
                }
            }
            index_offset += fn;
        }
    }
    //model->boundingBox = Bounds3(min_vert, max_vert);
    Resource::meshDataPool.push_back(model);
    Resource::meshDataIdx[filename] = meshCount++;
    //protoId = meshCount++;
    return model;
}

void Scene::clear()
{
    tempDevScene.destroy();
    cudaSafeFree(dev_scene);
}

void Resource::clear()
{
    for (auto& m : meshDataPool)
    {
        delete m;
    }
    meshDataPool.clear();
}

void DevScene::initiate(const Scene& scene)
{
    tri_num = scene.triangles.size();
    bvh_size = scene.gpuBVHNodeInfos.size();
    cudaMalloc(&dev_triangles, sizeof(Triangle) * scene.triangles.size());
    cudaMemcpy(dev_triangles, scene.triangles.data(), sizeof(Triangle) * scene.triangles.size(), cudaMemcpyHostToDevice);
    checkCUDAError("DevScene initiate::triangles");

    cudaMalloc(&dev_gpuBVH, sizeof(GpuBVHNode) * scene.gpuBVHNodes.size());
    cudaMemcpy(dev_gpuBVH, scene.gpuBVHNodes.data(), sizeof(GpuBVHNode) * scene.gpuBVHNodes.size(), cudaMemcpyHostToDevice);
    checkCUDAError("DevScene initiate::gpu bvh tree");
}

void DevScene::destroy()
{
    cudaSafeFree(dev_triangles);
    cudaFree(dev_gpuBVH);
}