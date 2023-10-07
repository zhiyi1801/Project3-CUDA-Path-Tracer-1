#pragma once
#ifndef BVH_H
#define BVH_H

#include"utilities.h"
#include "Bounds3.hpp"

//struct BVHBuildNode;

struct BVHNode;

struct BVHPrimitiveInfo;

struct BVHBuildInfo;

class BVHAccel 
{	
public:
	enum class SplitMethod {NAIVE, SAH};
	
	void buildBVH(std::vector<Triangle>& t,
		std::vector<BVHNode>& nodes,
		SplitMethod splitMethod);

	void buildSAH(std::vector<Triangle>& t,
		std::vector<BVHNode>& nodes);
	void buildNAIVE(std::vector<Triangle>& t,
		std::vector<BVHNode>& nodes);

	//bool Intersect(const Ray& ray, float& t, int& primIdx);
};

struct BVHNode
{
	Bounds3 bBox;
	int primitiveId;
	int miss;
};

struct BVHBuildInfo
{
	int start, end;
	//index in the buffer
	int idx;
	//offset is the position in the layer of the binary tree, depth is the depth of the binary tree, all start from 0
	int depth, offset;
	//record father node's miss pointer
	int preMiss, siblingIdx;
};

//struct BVHBuildNode
//{
//	int primId;
//	bool isLeaf;
//};

#endif // !BVH_H