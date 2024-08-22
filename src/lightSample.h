#pragma once
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"

//https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables#Distribution1D
class Distribution1D
{
public:
	std::vector<float> func, cdf;
	float funcInt;

	Distribution1D() = default;

	Distribution1D(std::vector<float> vals) : func(vals), cdf(vals.size() + 1)
	{
		int n = vals.size();
		cdf[0] = 0;
		for (int i = 1; i < n + 1; ++i)
		{
			cdf[i] = cdf[i - 1] + func[i - 1] / n;
		}
		funcInt = cdf[n];
		if (funcInt == 0) {
			for (int i = 1; i < n + 1; ++i)
				cdf[i] = static_cast<float>(i) / static_cast<float>(n);
		}
		else {
			for (int i = 1; i < n + 1; ++i)
				cdf[i] /= funcInt;
		}
	}

	Distribution1D(const float *vals, int n) : func(vals, vals + n), cdf(n + 1)
	{
		cdf[0] = 0;
		for (int i = 1; i < n + 1; ++i)
		{
			cdf[i] = cdf[i - 1] + func[i - 1] / n;
		}
		funcInt = cdf[n];
		if (funcInt == 0) {
			for (int i = 1; i < n + 1; ++i)
				cdf[i] = static_cast<float>(i) / static_cast<float>(n);
		}
		else {
			for (int i = 1; i < n + 1; ++i)
				cdf[i] /= funcInt;
		}
	}

	int Count() const { return func.size(); }

	__host__ __device__ float sampleContinuous(float u, float& pdf)
	{
		u = glm::clamp(u, 0.f, 1.f);
		int left = 0, right = cdf.size() - 1;
		while (right > left)
		{
			int mid = (right + left) / 2;
			if (cdf[mid] <= u)
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		int offset = glm::clamp(right - 1, 0, static_cast<int>(cdf.size() - 2));

		pdf = func[offset] / funcInt;
		float du = u - cdf[offset];
		if ((cdf[offset + 1] - cdf[offset]) > 0)
		{
			du /= (cdf[offset + 1] - cdf[offset]);
		}
		else
		{
			du = 0;
		}
		return (offset + du)/Count();
	}

	__host__ __device__ int sampleDiscrete(float u, float& pdf)
	{
		u = glm::clamp(u, 0.f, 1.f);
		int left = 0, right = cdf.size() - 1;
		while (right > left)
		{
			int mid = (right + left) / 2;
			if (cdf[mid] <= u)
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		int offset = glm::clamp(right - 1, 0, static_cast<int>(cdf.size() - 2));

		pdf = func[offset] / funcInt;

		return offset;
	}
};

class DevDistri1D
{
public:
	float* func = nullptr, * cdf = nullptr;
	float funcInt = 0;
	int size = 0;

	void create(Distribution1D& hstSampler)
	{
		size = hstSampler.func.size();
		cudaMalloc(&func, size * sizeof(float));
		cudaMemcpy(func, hstSampler.func.data(), size * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc(&cdf, (size + 1) * sizeof(float));
		cudaMemcpy(cdf, hstSampler.cdf.data(), (size + 1) * sizeof(float), cudaMemcpyHostToDevice);

		this->size = size;
	}

	void destroy()
	{
		cudaSafeFree(func);
		cudaSafeFree(cdf);
		size = 0;
	}

	__host__ __device__ float sampleContinuous(float u, float& pdf)
	{
		u = glm::clamp(u, 0.f, 1.f);
		int left = 0, right = size;
		while (right > left)
		{
			int mid = (right + left) / 2;
			if (cdf[mid] <= u)
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		int offset = glm::clamp(right - 1, 0, size - 1);

		pdf = func[offset] / funcInt;
		float du = u - cdf[offset];
		if ((cdf[offset + 1] - cdf[offset]) > 0)
		{
			du /= (cdf[offset + 1] - cdf[offset]);
		}
		else
		{
			du = 0;
		}
		return (offset + du)/size;
	}

	__host__ __device__ int sampleDiscrete(float u, float& pdf)
	{
		u = glm::clamp(u, 0.f, 1.f);
		int left = 0, right = size;
		while (right > left)
		{
			int mid = (right + left) / 2;
			if (cdf[mid] <= u)
			{
				left = mid + 1;
			}
			else
			{
				right = mid;
			}
		}
		volatile float t1 = cdf[right];
		int offset = glm::clamp(right - 1, 0, size - 1);
		volatile float t2 = cdf[right - 1];

		pdf = func[offset] / funcInt;

		return offset;
	}
};

class Distribution2D
{
public:
	std::vector<std::unique_ptr<Distribution1D>> pConditionalUV;
	std::unique_ptr<Distribution1D> pMarginalV;

	Distribution2D() = default;

	Distribution2D(const std::vector<float> data, int nu, int nv)
	{
		for (int v = 0; v < nv; ++v)
		{
			//p(u|v) a piecewise constant 1D function for every discrete
			pConditionalUV.emplace_back(new Distribution1D(data.data() + nu * v, nv));
		}

		std::vector<float> marginalFunc;
		for (int v = 0; v < nv; ++v)
		{
			marginalFunc.emplace_back(pConditionalUV[v]->funcInt);
		}
		// p(v)
		pMarginalV = std::make_unique<Distribution1D>(marginalFunc);
	}
};

class lightSampleRecord
{
public:
	glm::vec3 pos;
	glm::vec3 emit;
	float pdf;
};

class lightPrim
{
public:
	int geomID;
	int triangleID;
	GeomType type;

	lightPrim(int _geomID, int _triangleID, GeomType _type) : geomID(_geomID), triangleID(_triangleID), type(_type) {}
};

class LightSampler
{
public:
	lightPrim* lights;
	Geom* geoms;
	Triangle* triangles;
	Material* mats;
	GpuBVHNode* bvhRoot;

	int geomsSize;
	int lightSize;
	int bvhSize;
	int triangleSize;

	devTexSampler envSampler;

	struct Transform {
		glm::mat4 T;
		glm::mat4 invT;
		glm::mat3 invTransT;
		glm::vec3 scale;
	};

	__host__ __device__ bool occulusionTest(const glm::vec3 &ori, const glm::vec3 &dir, const glm::vec3 &des)const
	{
		float minT = glm::length(des - ori);
		Ray ray{ ori, dir };
		glm::vec3 nor;
		glm::vec3 interPoint;
		bool outside;
		float t = 0;
		for (int i = 0; i < geomsSize; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, ray, interPoint, nor, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, ray, interPoint, nor, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && minT - 1e-5f > t && glm::abs(t - minT) > 1e-2)
			{
				return true;
			}
		}

#if USE_BVH
		int bvhIdx = 0;
		Triangle tempTri;
		float tempT = FLT_MAX;
		int offset = 0;
#if USE_MTBVH
		offset = ((abs(dir[0]) > abs(dir[1])) && (abs(dir[0]) > abs(dir[2]))) ? 0 : (abs(dir[1]) > abs(dir[2]) ? 1 : 2);
		offset = offset + (dir[offset] > 0 ? 0 : 3);
		offset *= bvhSize;
#endif // !USE_MTBVH
		GpuBVHNode* curBVH = bvhRoot + offset;
		volatile int count = 0;
		while (bvhIdx != -1)
		{
			count++;
			if (!(curBVH[bvhIdx].bBox.IntersectP(ray, tempT)) || tempT > minT)
			{
				bvhIdx = curBVH[bvhIdx].miss;
				continue;
			}
			//it indicates gpuBVH[bvhIdx] is a leaf node
			if (curBVH[bvhIdx].end - curBVH[bvhIdx].start <= MAX_PRIM)
			{
				for (int i = curBVH[bvhIdx].start; i < curBVH[bvhIdx].end; ++i)
				{
					tempTri = triangles[i];
					float u, v;
					bool isHit = tempTri.getInterSect(ray, t, u, v);
					if (isHit && minT - 1e-5f > t && glm::abs(t - minT) > 1e-4)
					{
						return true;
					}
				}
			}
			bvhIdx = curBVH[bvhIdx].hit;
		}
#else // !USE_BVH
		for (int i = 0; i < triangleSize; ++i)
		{
			float u, v;
			Triangle tempTri = triangles[i];
			bool isHit = tempTri.getInterSect(ray, t, u, v);
			if (isHit && minT - 1e-5f > t)
			{
				return false;
			}
		}
#endif// !USE_BVH

		return false;
	}

	__host__ __device__ float lightPDF(const glm::vec3& viewPos, const glm::vec3 &lightPos, const glm::vec3 &normal, int triID, int geomID, Sampler sampler)const
	{
		float pdf = -1.f;
		Geom geom = geoms[geomID];
		if (triID >= 0)
		{
			Triangle tri = triangles[triID];


			// just use interpolated normal
			float area = glm::length(glm::cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[0])) / 2.f;
			pdf = 1.f / lightSize;
			// both sides lights
			pdf = pdf * glm::length2(lightPos - viewPos) / (area * glm::abs(glm::dot(glm::normalize(viewPos - lightPos), normal)));
		}

		if (geoms[geomID].type == GeomType::SPHERE)
		{
			Transform tr{ geom.transform, geom.inverseTransform, glm::mat3(geom.invTranspose), geom.scale };
			glm::vec3 viewPosL = glm::vec3(tr.invT * glm::vec4(viewPos, 1.f));
			glm::vec3 center = glm::vec3(0.f);

			float sinThetaMax2 = (0.5f * 0.5f) / glm::dot(viewPosL - center, viewPosL - center); // Again, radius is 1
			float cosThetaMax = sqrt(max(0.0f, 1.0f - sinThetaMax2));

			pdf = 1.0f / (TWO_PI * (1 - cosThetaMax) * lightSize);
		}
		return pdf;
	}

	__host__ __device__ void lightSample(const glm::vec3& viewPos, Sampler sampler, lightSampleRecord& rec)const
	{
		if (lightSize == 0)
		{
			return;
		}

		int lightID = glm::min(sample1D(sampler) * lightSize, lightSize - 1.f);
		lightPrim light = lights[lightID];
		Geom geom = geoms[light.geomID];
		Material lightMat = mats[geom.materialid];
		glm::vec3 normal(0.f);
		glm::vec3 lightPos;
		float pdf = 0;
		volatile float s1 = 1, s2 = 1, s3 = 1, v1 = 1, v2 = 1, v3 = 1, l1= 1, l2 = 1, l3 = 1;
		if (light.triangleID >= 0)
		{
			glm::vec2 baryCentric = math::sampleTriangleUniform(sample2D(sampler));
			float u = baryCentric.x, v = baryCentric.y;
			int id = light.triangleID;
			Triangle tri = triangles[id];
			lightPos = u * tri.v[0] + v * tri.v[1] + (1 - u - v) * tri.v[2];

			// just use interpolated normal
			normal = glm::normalize(u * tri.n[0] + v * tri.n[1] + (1 - u - v) * tri.n[2]);
			float area = glm::length(glm::cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[0])) / 2.f;
			pdf = 1.f / lightSize;
			// both sides lights
			pdf = pdf * glm::length2(lightPos - viewPos) / (area * glm::abs(glm::dot(glm::normalize(viewPos - lightPos), normal)));
		}

		else if (light.type == GeomType::SPHERE)
		{
			glm::vec2 xi = sample2D(sampler);
			Transform tr{ geom.transform, geom.inverseTransform, glm::mat3(geom.invTranspose), geom.scale };
			s1 = tr.scale.x, s2 = tr.scale.y, s3 = tr.scale.z;
			glm::vec3 viewPosL = glm::vec3(tr.invT * glm::vec4(viewPos, 1.f));
			v1 = viewPosL.x, v2 = viewPosL.y, v3 = viewPosL.z;
			glm::vec3 center = glm::vec3(tr.T * glm::vec4(0., 0., 0., 1.));
			center = glm::vec3(0.f);
			glm::vec3 centerToRef = glm::normalize(center - viewPosL);
			glm::vec3 tan, bit;

			math::localRefMatrix_Pixar(centerToRef, tan, bit);

			float sinThetaMax2 = (0.5f * 0.5f) / glm::dot(viewPosL - center, viewPosL - center); // Again, radius is 1
			float cosThetaMax = sqrt(max(0.0f, 1.0f - sinThetaMax2));
			float cosTheta = (1.0f - xi.x) + xi.x * cosThetaMax;
			float sinTheta = sqrt(max(0.f, 1.0f - cosTheta * cosTheta));
			float phi = xi.y * TWO_PI;

			float dc = glm::distance(viewPosL, center);
			float ds = dc * cosTheta - sqrt(max(0.0f, 0.5f * 0.5f - dc * dc * sinTheta * sinTheta));

			float sinAlpha = ds * sinTheta / 0.5f;
			float cosAlpha = glm::sqrt(glm::max(0.f, 1.f - sinAlpha * sinAlpha));

			glm::vec3 nObj = sinAlpha * cos(phi) * tan + sinAlpha * sin(phi) * bit + cosAlpha * -centerToRef;
			glm::vec3 pObj = nObj * 0.5f; // Would multiply by radius, but it is always 1 in object space

			lightPos = glm::vec3(tr.T * glm::vec4(pObj, 1.0f));
			l1 = lightPos.x, l2 = lightPos.y, l3 = lightPos.z;
			pdf = 1.0f / (TWO_PI * (1 - cosThetaMax) * lightSize);
		}

		glm::vec3 rayDir = glm::normalize(lightPos - viewPos);
		bool occlution = occulusionTest(viewPos + 1e-5f * rayDir, rayDir, lightPos);
		if (occlution)
		{
			rec.emit = glm::vec3(0.f);
			rec.pdf = -1.f;
			return;
		}
		rec.emit = lightMat.albedo;
		rec.pdf = pdf;
		rec.pos = lightPos;
		return;
	}
};