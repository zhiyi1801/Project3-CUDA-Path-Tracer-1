#pragma once
#include<limits>
#include<array>
#include<vector>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "sceneStructs.h"

class Bounds3
{
public:
	glm::vec3 pMin, pMax;
	Bounds3()
	{
		double minNum = std::numeric_limits<double>::lowest();
		double maxNum = std::numeric_limits<double>::max();
		pMin = glm::vec3(minNum, minNum, minNum);
		pMax = glm::vec3(maxNum, maxNum, maxNum);
	}

	Bounds3(const glm::vec3& p) : pMin(p), pMax(p) {}
	Bounds3(const glm::vec3& p1, const glm::vec3& p2)
	{
		//pMin = glm::vec3(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
		//pMax = glm::vec3(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
		pMin = glm::min(p1, p2);
		pMax = glm::max(p1, p2);
	}

	Bounds3& operator=(const Bounds3& b)
	{
		this->pMin = b.pMin;
		this->pMax = b.pMax;
		return *this;
	}

	glm::vec3 Diagonal() const { return pMax - pMin; }

	int MaxExtent()
	{
		glm::vec3 diag = Diagonal();
		if (diag.x > diag.y && diag.x > diag.z) return 0;
		else if (diag.y > diag.z) return 1;
		else return 2;
	}

	double SurfaceArea()
	{
		glm::vec3 d = Diagonal();
		double area = 2 * ((double)d.x * (double)d.y + (double)d.y * (double)d.z + (double)d.z * (double)d.x);
	}

	glm::vec3 Centroid()
	{
		return (pMax + pMin) / 2.0f;
	}

	glm::vec3 Offset(const glm::vec3 p)
	{
		glm::vec3 o = p - pMin;

		if (pMax.x > pMin.x) o.x /= (pMax.x - pMin.x);
		if (pMax.y > pMin.y) o.y /= (pMax.y - pMin.y);
		if (pMax.z > pMin.z) o.z /= (pMax.z - pMin.z);

		return o;
	}

	bool Overlaps(const Bounds3& b)
	{
		bool x = (pMax.x >= b.pMin.x) && (pMin.x <= b.pMax.x);
		bool y = (pMax.y >= b.pMin.y) && (pMin.y <= b.pMax.y);
		bool z = (pMax.z >= b.pMin.z) && (pMin.z <= b.pMax.z);

		return x && y && z;
	}

	bool Inside(const glm::vec3& p)
	{
		return(p.x < pMax.x&& p.x > pMin.x && p.y < pMax.y&& p.y > pMin.y && p.z < pMax.z&& p.z > pMin.z);
	}

	inline const glm::vec3& operator[](int i)
	{
		return (i == 0 ? pMin : pMax);
	}

	inline bool IntersectP(const Ray& ray);
};

inline bool Bounds3::IntersectP(const Ray& ray)
{
	glm::vec3 o = ray.origin;
	const glm::vec3 &invDir = ray.direction_inv;
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
		}
		if (ray.direction[i] < 0) std::swap(tMax, tMin);
		tEnter = std::max(tMax, tEnter);
		tExit = std::min(tMin, tExit);
	}

	return tEnter <= tExit && tExit > 0;
}

inline Bounds3 Union(const Bounds3& b, const glm::vec3& p)
{
	Bounds3 ret;
	ret.pMin = glm::vec3(glm::min(b.pMin, p));
	ret.pMax = glm::vec3(glm::max(b.pMax, p));
	return ret;
}

inline Bounds3 Union(const Bounds3& b1, const Bounds3& b2)
{
	Bounds3 ret;
	ret.pMin = glm::vec3(glm::min(b1.pMin, b2.pMax));
	ret.pMax = glm::vec3(glm::max(b1.pMax, b2.pMax));
	return ret;
}