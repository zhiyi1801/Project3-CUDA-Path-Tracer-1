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
			if (t > 0.0f && minT - 1e-5f > t)
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
					if (isHit && minT - 1e-5f > t)
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

	__host__ __device__ float lightPDF(int lightID, Sampler sampler)
	{
		lightPrim light = lights[lightID];

		if (light.triangleID >= 0)
		{
			glm::vec2 baryCentric = math::sampleTriangleUniform(sample2D(sampler));

		}
	}

	__host__ __device__ void lightSample(const glm::vec3& viewPos, const glm::vec3& viewNor, Sampler sampler, lightSampleRecord& rec)const
	{
		int lightID = glm::min(sample1D(sampler) * lightSize, lightSize - 1.f);
		lightPrim light = lights[lightID];
		Geom geom = geoms[light.geomID];
		Material lightMat = mats[geom.materialid];
		glm::vec3 normal(0.f);
		glm::vec3 lightPos;
		float pdf = 0;
		if (light.triangleID >= 0)
		{
			glm::vec2 baryCentric = math::sampleTriangleUniform(sample2D(sampler));
			float u = baryCentric.x, v = baryCentric.y;
			int id = light.triangleID;
			Triangle tri = triangles[id];
			lightPos = u * tri.v[0] + v * tri.v[1] + (1 - u - v) * tri.v[2];

			// just use interpolated normal
			normal = glm::normalize(u * tri.n[0] + v * tri.n[1] + (1 - u - v) * tri.n[2]);
			float area = glm::length(glm::cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[0])) / 2;
			pdf = 1.f / lightSize;
			// both sides lights
			pdf = pdf * glm::length2(lightPos - viewPos) / (area * glm::abs(glm::dot(glm::normalize(viewPos - lightPos), normal)));
		}

		else if (light.type == GeomType::SPHERE)
		{
			glm::vec2 xi = sample2D(sampler);
			glm::vec3 scale(glm::length(geom.transform[0]), glm::length(geom.transform[1]), glm::length(geom.transform[2]));
			Transform tr{ geom.transform, geom.inverseTransform, glm::mat3(geom.invTranspose), scale };

			glm::vec3 center = glm::vec3(tr.T * glm::vec4(0., 0., 0., 1.));
			glm::vec3 centerToRef = glm::normalize(center - viewPos);
			glm::vec3 tan, bit;

			math::localRefMatrix_Pixar(centerToRef, tan, bit);

			glm::vec3 pOrigin;
			if (glm::dot(center - viewPos, viewNor) > 0) {
				pOrigin = viewPos + viewNor * 1e-5f;
			}
			else {
				pOrigin = viewPos - viewNor * 1e-5f;
			}

			// Inside the sphere
			//if (dot(pOrigin - center, pOrigin - center) <= 1.f) // Radius is 1, so r^2 is also 1
			//	return sampleFromInsideSphere(xi, pdf);

			float sinThetaMax2 = 1.f / glm::dot(viewPos - center, viewPos - center); // Again, radius is 1
			float cosThetaMax = sqrt(max(0.0f, 1.0f - sinThetaMax2));
			float cosTheta = (1.0f - xi.x) + xi.x * cosThetaMax;
			float sinTheta = sqrt(max(0.f, 1.0f - cosTheta * cosTheta));
			float phi = xi.y * TWO_PI;

			float dc = glm::distance(viewPos, center);
			float ds = dc * cosTheta - sqrt(max(0.0f, 1 - dc * dc * sinTheta * sinTheta));

			float cosAlpha = (dc * dc + 1 - ds * ds) / (2 * dc * 1);
			float sinAlpha = sqrt(max(0.0f, 1.0f - cosAlpha * cosAlpha));

			glm::vec3 nObj = sinAlpha * cos(phi) * -tan + sinAlpha * sin(phi) * -bit + cosAlpha * -centerToRef;
			glm::vec3 pObj = glm::vec3(nObj); // Would multiply by radius, but it is always 1 in object space

			lightPos = glm::vec3(tr.T * glm::vec4(pObj, 1.0f));
			pdf = 1.0f / (TWO_PI * (1 - cosThetaMax) * lightSize * tr.scale.x * tr.scale.y);
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