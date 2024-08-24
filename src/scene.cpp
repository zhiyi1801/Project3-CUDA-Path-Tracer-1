#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "stb_image.h"
#include "stb_image_write.h"

extern float theta, phi;
extern bool posInit;

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
    std::map<std::string, MeshData*> meshPool;
    std::map<std::string, image*> texturePool;
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
    stbi_set_flip_vertically_on_load(true);
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
            else if (strcmp(tokens[0].c_str(), "ENV") == 0)
            {
                this->envMapID = loadTexture(tokens[1]);
            }
        }
    }

}

int Scene::loadGeom(string objectName) {
    int id = geoms.size();
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    if (geomNameMap.find(objectName) != geomNameMap.end())
    {
        cout << "ERROR: OBJECT Name already exists" << endl;
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
            if (materialNameMap.find(tokens[1]) != materialNameMap.end())
            {
                newGeom.materialid = materialNameMap[tokens[1]];
            }
            else
            {
                newGeom.materialid = atoi(tokens[1].c_str());
            }
            cout << "Connecting Geom " << id << ": " << objectName << " to Material " << newGeom.materialid << "..." << endl;
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
        geomNameMap[objectName] = id;
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
        if (strcmp(tokens[0].c_str(), "ROTAT") == 0)
        {
            theta = glm::clamp(static_cast<float>(atof(tokens[1].c_str())), -89.f, 89.f);
            phi = atof(tokens[2].c_str());
            posInit = false;
        }
        else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            posInit = true;
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
int Scene::loadMaterial(string materialName) {
    int id = materials.size();
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    }
    if (materialNameMap.find(materialName) != materialNameMap.end())
    {
        cout << "ERROR: MATERIAL Name already exists" << endl;
        return -1;
    }
    else {
        cout << "Loading Material " << id << ": " << materialName << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 6; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (tokens.size() <= 0) break;
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
                newMaterial.albedoMapID = this->loadTexture(tokens[1]);
                if (newMaterial.albedoMapID < 0)
                {
                   newMaterial.albedo = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                   newMaterial.albedoSampler = devTexSampler(newMaterial.albedo);
                }
            }
            else if (strcmp(tokens[0].c_str(), "METALLIC") == 0) {
                newMaterial.metallicMapID = this->loadTexture(tokens[1]);
                if (newMaterial.metallicMapID < 0)
                {
                    newMaterial.metallic = atof(tokens[1].c_str());
                    newMaterial.metallicSampler = devTexSampler(newMaterial.metallic);
                }
            }
            else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
                newMaterial.roughnessMapID = this->loadTexture(tokens[1]);
                if (newMaterial.roughnessMapID < 0)
                {
                    newMaterial.roughness = glm::max(atof(tokens[1].c_str()), ROUGHNESS_MIN);
                    newMaterial.roughnessSampler = devTexSampler(newMaterial.roughness);
                }
            }
            else if (strcmp(tokens[0].c_str(), "NORMAL") == 0) {
                newMaterial.normalMapID = this->loadTexture(tokens[1]);
                if (newMaterial.normalMapID < 0)
                {
                    // map [-1, 1] to [0, 1]
                    newMaterial.normalSampler = devTexSampler(glm::vec3(0.5f, 0.5f, 1.f));
                }
            }
            else if (strcmp(tokens[0].c_str(), "IOR") == 0) {
                newMaterial.ior = atof(tokens[1].c_str());
            }
        }
        newMaterial.normalSampler = devTexSampler(glm::vec3(0.5f, 0.5f, 1.f));
        materials.push_back(newMaterial);
        materialNameMap[materialName] = id;
        return 1;
    }
}

int Scene::loadTexture(const string &fileName, float gamma)
{
    cout << "Loading texture: " + fileName + "\n";
    image* tex = Resource::loadTexture(fileName, gamma);
    if (!tex)
    {
        return -1;
    }
    auto texID = textureIdMap.find(tex);
    if (texID != textureIdMap.end())
    {
        cout << "Already exist: " + fileName + "\n";
        return texID->second;
    }

    int ret = textures.size();
    textures.emplace_back(tex);
    textureIdMap[tex] = ret;
    return ret;
}


MeshData* Resource::loadObj(const string& filename, const int _geomIdx)
{
    auto exist = meshPool.find(filename);
    if (exist != meshPool.end())
    {
        //protoId = exist->second;
        std::cout << "---obj file existed " << filename << "---" << std::endl;
        return exist->second;
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
    bool hasNor = !attrib.normals.empty();
    if (!hasNor)
    {
        cout << "model does not have vertex normal, use face normal\n";
    }

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
                std::array<glm::vec3, 3> _n;
#if VERTEX_NORMAL
                if (hasNor)
                {
                    _n = std::array<glm::vec3, 3> { *((glm::vec3*)attrib.normals.data() + idx0.normal_index),
                                                    *((glm::vec3*)attrib.normals.data() + idx1.normal_index),
                                                    *((glm::vec3*)attrib.normals.data() + idx2.normal_index) };
                }
                else
                {
                    glm::vec3 faceNor = glm::normalize(glm::cross(_v[1] - _v[0], _v[2] - _v[0]));
                    _n = std::array<glm::vec3, 3>{ faceNor, faceNor, faceNor };
                }
#else
                //face normal
                glm::vec3 faceNor = glm::normalize(glm::cross(_v[1] - _v[0], _v[2] - _v[0]));
                _n = { faceNor, faceNor, faceNor };
#endif

                std::array<glm::vec2, 3> _tex;
                if (hasTex)
                {
                    _tex = std::array<glm::vec2, 3>{ *((glm::vec2*)attrib.texcoords.data() + idx0.texcoord_index),
                        * ((glm::vec2*)attrib.texcoords.data() + idx1.texcoord_index),
                        * ((glm::vec2*)attrib.texcoords.data() + idx2.texcoord_index) };
                }
                else
                {
                    std::cout << "--- This obj has no texcoords " << "---" << std::endl;
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
    Resource::meshPool[filename] = model;
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
    for (auto& m : meshPool)
    {
        delete m.second;
    }
    meshPool.clear();

    for (auto& t : texturePool)
    {
        delete t.second;
    }
    texturePool.clear();
}

image* Resource::loadTexture(const std::string& filename, float gamma) {
    auto find = texturePool.find(filename);
    if (find != texturePool.end()) {
        return find->second;
    }
    auto texture = new image(filename, gamma);
    if (!texture->pixels)
    {
        return nullptr;
    }
    texturePool[filename] = texture;
    return texture;
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

                temp_t.tangent = glm::vec3(0.f);
                temp_t.bitangent = glm::vec3(0.f);
                glm::vec3 edge1 = temp_t.v[1] - temp_t.v[0];
                glm::vec3 edge2 = temp_t.v[2] - temp_t.v[0];
                glm::vec2 deltaUV1 = temp_t.tex[1] - temp_t.tex[0];
                glm::vec2 deltaUV2 = temp_t.tex[2] - temp_t.tex[0];
                float f = (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
                if (glm::abs(f) < 1e-8)
                {
                    continue;
                }
                temp_t.tangent = glm::normalize((deltaUV2.y * edge1 - deltaUV1.y * edge2) / f);
                temp_t.bitangent = glm::normalize((-deltaUV2.x * edge1 + deltaUV1.x * edge2) / f);
            }
            temp_t.geomIdx = gpuGeoms.size() - 1;
            triangles.push_back(temp_t);
        }
    }

    if (envMapID >= 0)
    {
        image* envMap = this->textures[envMapID];
        std::vector<float> envVals;
        std::vector<unsigned char> envValsChar;
        for (int i = 0; i < envMap->height; i++) {
            for (int j = 0; j < envMap->width; j++) {
                int idx = i * envMap->width + j;
                envVals.emplace_back(math::rgb2Luminance(envMap->data()[idx]) * glm::sin((.5f + i) / envMap->height * PI));
                envValsChar.emplace_back(envVals.back() * 255.f);
            }
        }
        envDistribution = Distribution1D(envVals);

        stbi_write_png("output.png", envMap->width, envMap->height, 1, envValsChar.data(), envMap->width * 1);
    }

    this->bvhRoot = this->bvhConstructor.recursiveBuild(this->triangles);
    this->bvhConstructor.recursiveBuildGpuBVHInfo(bvhRoot, this->gpuBVHNodeInfos);
#if USE_MTBVH
    this->bvhConstructor.buildGpuMTBVH(this->gpuBVHNodeInfos, this->gpuBVHNodes);
#else
    this->bvhConstructor.buildGpuBVH(this->gpuBVHNodeInfos, this->gpuBVHNodes);
#endif
    for (int i = 0; i < geoms.size(); ++i)
    {
        if (materials[geoms[i].materialid].type == Material::Type::Light && geoms[i].type != GeomType::OBJ)
        {
            int geomID = i;
            int triangleID = -1;
            GeomType type = geoms[i].type;
            lights.emplace_back(lightPrim(geomID, triangleID, type));
        }
    }

    for (int i = 0; i < triangles.size(); ++i)
    {
        int geomID = triangles[i].geomIdx;
        if (materials[geoms[geomID].materialid].type == Material::Type::Light)
        {
            int triangleID = i;
            GeomType type = GeomType::OBJ;
            lights.emplace_back(lightPrim(geomID, triangleID, type));
        }
    }

    this->tempDevScene.initiate(*this);
    cudaMalloc(&dev_scene, sizeof(DevScene));
    cudaMemcpy(dev_scene, &tempDevScene, sizeof(DevScene), cudaMemcpyHostToDevice);
    checkCUDAError("dev_scene");

    bvhRoot->destroy();
    gpuBVHNodeInfos.clear();
}

void DevScene::initiate(Scene& scene)
{
    tri_num = scene.triangles.size();
    bvh_size = scene.gpuBVHNodeInfos.size();

    tex_num = scene.textures.size();
    std::vector<devTexObj> tempDevTextures;
    long long int textureTotalBytes = 0;
    for (auto tex : scene.textures)
    {
        textureTotalBytes += tex->byteSize();
    }
    cudaMalloc(&dev_texture_data, textureTotalBytes);
    checkCUDAError("Devscene: dev_texture_data malloc");

    long long int offset = 0;
    for (auto tex : scene.textures)
    {
        cudaMemcpy(dev_texture_data + offset, tex->data(), tex->byteSize(), cudaMemcpyHostToDevice);
        checkCUDAError("Devscene: dev_texture_data copy");
        tempDevTextures.emplace_back(devTexObj(tex, dev_texture_data + offset));
        offset += tex->byteSize() / sizeof(glm::vec3);
    }
    
    cudaMalloc(&dev_textures, tempDevTextures.size() * sizeof(devTexObj));
    checkCUDAError("DevScene::dev_textures::malloc");

    cudaMemcpy(dev_textures, tempDevTextures.data(), tempDevTextures.size() * sizeof(devTexObj), cudaMemcpyHostToDevice);
    checkCUDAError("DevScene::dev_textures::copy");

    for (auto &mat : scene.materials)
    {
        if (mat.albedoMapID >= 0 && mat.albedoMapID < scene.textures.size())
        {
            mat.albedoSampler.tex = dev_textures + mat.albedoMapID;
        }

        if (mat.metallicMapID >= 0 && mat.metallicMapID < scene.textures.size())
        {
            mat.metallicSampler.tex = dev_textures + mat.metallicMapID;
        }

        if (mat.roughnessMapID >= 0 && mat.roughnessMapID < scene.textures.size())
        {
            mat.roughnessSampler.tex = dev_textures + mat.roughnessMapID;
        }

        if (mat.normalMapID >= 0 && mat.normalMapID < scene.textures.size())
        {
            mat.normalSampler.tex = dev_textures + mat.normalMapID;
        }
    }

    this->envMapID = scene.envMapID;
    if (this->envMapID >= 0)
    {
        this->envSampler.tex = dev_textures + this->envMapID;
        this->envDistribution.create(scene.envDistribution);
    }

    cudaMalloc(&dev_triangles, sizeof(Triangle) * scene.triangles.size());
    cudaMemcpy(dev_triangles, scene.triangles.data(), sizeof(Triangle) * scene.triangles.size(), cudaMemcpyHostToDevice);
    checkCUDAError("DevScene initiate::triangles");

    cudaMalloc(&dev_gpuBVH, sizeof(GpuBVHNode) * scene.gpuBVHNodes.size());
    cudaMemcpy(dev_gpuBVH, scene.gpuBVHNodes.data(), sizeof(GpuBVHNode) * scene.gpuBVHNodes.size(), cudaMemcpyHostToDevice);
    checkCUDAError("DevScene initiate::gpu bvh tree");

    cudaMalloc(&dev_geoms, sizeof(Geom) * scene.geoms.size());
    cudaMemcpy(dev_geoms, scene.geoms.data(), sizeof(Geom) * scene.geoms.size(), cudaMemcpyHostToDevice);
    checkCUDAError("DevScene initiate::gpu geoms");

    cudaMalloc(&dev_materials, scene.materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    checkCUDAError("DevScene initiate::gpu material");

    cudaMalloc(&dev_lights, sizeof(lightPrim) * scene.lights.size());
    cudaMemcpy(dev_lights, scene.lights.data(), sizeof(lightPrim) * scene.lights.size(), cudaMemcpyHostToDevice);
    checkCUDAError("DevScene initiate::gpu lights");

    LightSampler& lightSampler = this->dev_lightSampler;
    lightSampler.bvhRoot = dev_gpuBVH;
    lightSampler.bvhSize = this->bvh_size;

    lightSampler.geoms = this->dev_geoms;
    lightSampler.geomsSize = scene.geoms.size();

    lightSampler.triangles = this->dev_triangles;
    lightSampler.triangleSize = this->tri_num;

    lightSampler.mats = this->dev_materials;

    lightSampler.lights = this->dev_lights;
    lightSampler.lightSize = scene.lights.size();
}

void DevScene::destroy()
{
    cudaSafeFree(dev_triangles);
    cudaSafeFree(dev_gpuBVH);
    cudaSafeFree(dev_textures);
    cudaSafeFree(dev_texture_data);
    envDistribution.destroy();
}