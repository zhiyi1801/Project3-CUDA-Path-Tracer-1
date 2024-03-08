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
#include <windows.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

class lightSampleRecord
{
public:
	glm::vec3 pos;
	glm::vec3 BSDF;
	float pdf;
};

class lightPrim
{
public:
	int geomID;
	int triangleID;
	GeomType type;
};

class devLightSampler
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

	__host__ __device__ bool occulusionTest(const glm::vec3 &ori, const glm::vec3 &dir, const glm::vec3 &des)
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
			if (t > 0.0f && minT - 1e-5f > t)
			{
				return false;
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
					if (isHit && minT - 1e-5f > t)
					{
						return false;
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

		return true;
	}

	__host__ __device__ float lightPDF(int lightID, Sampler sampler)
	{
		lightPrim light = lights[lightID];

		if (light.triangleID >= 0)
		{
			glm::vec2 baryCentric = math::sampleTriangleUniform(sample2D(sampler));

		}
	}

	__host__ __device__ void lightSample(int lightID, const glm::vec3& viewPos, Sampler sampler, lightSampleRecord& rec)
	{
		lightPrim light = lights[lightID];
		Material lightMat = mats[geoms[light.geomID].materialid];
		glm::vec3 lightPos;
		if (light.triangleID >= 0)
		{
			glm::vec2 baryCentric = math::sampleTriangleUniform(sample2D(sampler));
			float u = baryCentric.x, v = baryCentric.y;
			int id = light.triangleID;
			lightPos = u * triangles[id].v[0] + v * triangles[id].v[1] + (1 - u - v) * triangles[id].v[2];
		}

		else if (light.type == GeomType::SPHERE)
		{

		}

		bool occlution = occulusionTest(viewPos, glm::normalize(lightPos - viewPos), lightPos);
	}
};