#pragma once
#include<limits>
#include<array>
#include<vector>
#include <thrust/execution_policy.h>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "sceneStructs.h"

class Bounds3
{
public:
	glm::vec3 pMin, pMax;

	__host__ __device__
	Bounds3()
	{
		double minNum = std::numeric_limits<double>::lowest();
		double maxNum = std::numeric_limits<double>::max();
		pMin = glm::vec3(maxNum, maxNum, maxNum);
		pMax = glm::vec3(minNum, minNum, minNum);
	}

	__host__ __device__
	Bounds3(const glm::vec3& p) : pMin(p), pMax(p) {}

	__host__ __device__
	Bounds3(const glm::vec3& p1, const glm::vec3& p2)
	{
		//pMin = glm::vec3(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
		//pMax = glm::vec3(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
		pMin = glm::min(p1, p2);
		pMax = glm::max(p1, p2);
	}

	__host__ __device__
	Bounds3& operator=(const Bounds3& b)
	{
		this->pMin = b.pMin;
		this->pMax = b.pMax;
		return *this;
	}

	__host__ __device__
	glm::vec3 Diagonal() const { return pMax - pMin; }

	__host__ __device__
	glm::vec3 Centroid()const { return (pMax + pMin) / 2.0f; }

	__host__ __device__
	int MaxExtent()
	{
		glm::vec3 diag = Diagonal();
		if (diag.x > diag.y && diag.x > diag.z) return 0;
		else if (diag.y > diag.z) return 1;
		else return 2;
	}

	__host__ __device__
	double SurfaceArea()
	{
		glm::vec3 d = Diagonal();
		double area = 2 * ((double)d.x * (double)d.y + (double)d.y * (double)d.z + (double)d.z * (double)d.x);
	}

	__host__ __device__
	glm::vec3 Centroid()
	{
		return (pMax + pMin) / 2.0f;
	}

	__host__ __device__
	glm::vec3 Offset(const glm::vec3 p)
	{
		glm::vec3 o = p - pMin;

		if (pMax.x > pMin.x) o.x /= (pMax.x - pMin.x);
		if (pMax.y > pMin.y) o.y /= (pMax.y - pMin.y);
		if (pMax.z > pMin.z) o.z /= (pMax.z - pMin.z);

		return o;
	}

	__host__ __device__
	bool Overlaps(const Bounds3& b)
	{
		bool x = (pMax.x >= b.pMin.x) && (pMin.x <= b.pMax.x);
		bool y = (pMax.y >= b.pMin.y) && (pMin.y <= b.pMax.y);
		bool z = (pMax.z >= b.pMin.z) && (pMin.z <= b.pMax.z);

		return x && y && z;
	}

	__host__ __device__
	bool Inside(const glm::vec3& p)
	{
		return(p.x < pMax.x&& p.x > pMin.x && p.y < pMax.y&& p.y > pMin.y && p.z < pMax.z&& p.z > pMin.z);
	}

	__host__ __device__
	inline const glm::vec3& operator[](int i)
	{
		return (i == 0 ? pMin : pMax);
	}

	__host__ __device__
	inline bool IntersectP(const Ray& ray);

	__host__ __device__
	Bounds3& Union(const Bounds3& b)
	{
		pMin = glm::min(pMin, b.pMin);
		pMax = glm::max(pMax, b.pMax);
		return *this;
	}
};

__host__ __device__
inline bool Bounds3::IntersectP(const Ray& ray)
{
	glm::vec3 o = ray.origin;
	const glm::vec3 invDir = 1.0f/ray.direction;
	double tEnter = std::numeric_limits<double>::lowest();
	double tExit = std::numeric_limits<double>::max();

	for (int i = 0; i < 3; ++i)
	{
		double tMax;
		double tMin;
		if (ray.direction[i] == 0)
		{
			if (o[i] < pMin[i] || o[i] > pMax[i]) return false;
		}
		else
		{
			tMax = ((double)pMax[i] - o[i]) * invDir[i];
			tMin = ((double)pMin[i] - o[i]) * invDir[i];
			if (ray.direction[i] < 0) thrust::swap(tMax, tMin);
			tEnter = tMin > tEnter ? tMin : tEnter;
			tExit = tMax < tExit ? tMax : tExit;
		}
	}

	return tEnter <= tExit && tEnter > 0;
}

__host__ __device__
inline Bounds3 Union(const Bounds3& b, const glm::vec3& p)
{
	Bounds3 ret;
	ret.pMin = glm::vec3(glm::min(b.pMin, p));
	ret.pMax = glm::vec3(glm::max(b.pMax, p));
	return ret;
}

__host__ __device__
inline Bounds3 Union(const Bounds3& b1, const Bounds3& b2)
{
	Bounds3 ret;
	ret.pMin = glm::vec3(glm::min(b1.pMin, b2.pMax));
	ret.pMax = glm::vec3(glm::max(b1.pMax, b2.pMax));
	return ret;
}

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