#pragma once
#ifndef BVH_H
#define BVH_H

#define MAX_PRIM 100
#define BUCKET_NUM 20

#include"utilities.h"
#include "Bounds3.hpp"

//struct BVHBuildNode;

struct BVHNode;
struct BVHPrimitiveInfo;
struct BVHBuildInfo;
struct RecursiveBVHNode;
struct GpuMTBVHNode;
struct GpuBVHNode;
struct GpuBVHNodeInfo;

class BVHAccel 
{	
public:
	RecursiveBVHNode* recursiveBuild(std::vector<Triangle>& t);
	RecursiveBVHNode* recursiveBuildNaive(std::vector<Triangle>& t, const int start, const int end);
	RecursiveBVHNode* recursiveBuildSAH(std::vector<Triangle>& t, const int start, const int end);

	int recursiveBuildGpuBVHInfo(const RecursiveBVHNode* root, std::vector<GpuBVHNodeInfo>& bvhInfo, const int parent = -1);

	void buildGpuBVH(const std::vector<GpuBVHNodeInfo>& bvhInfo, std::vector<GpuBVHNode>& bvh);
	void buildGpuMTBVH(const std::vector<GpuBVHNodeInfo>& bvhInfo, std::vector<GpuBVHNode>& MTbvh);

	//bool Intersect(const Ray& ray, float& t, int& primIdx);
};
// recursiveBVHNode -> GpuBVHNodeInfo -> GpuBVHNode/GpuMTBVHNode
struct BVHNode
{
	Bounds3 bBox;
	int primitiveId;
	// hit index is the next in the buffer
	int miss;
};

struct GpuBVHNode
{
	Bounds3 bBox;
	int start, end;
	// hit index is the next in the buffer
	int miss, hit;
};

struct GpuBVHNodeInfo
{
	const RecursiveBVHNode* BVHNode;
	int parent, left, right;
	GpuBVHNodeInfo() :BVHNode(nullptr), parent(-1), left(-1), right(-1) {}
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

// primitive[start, end)
struct RecursiveBVHNode
{
	int start, end;
	RecursiveBVHNode* leftChild;
	RecursiveBVHNode* rightChild;
	Bounds3 bBox;
	RecursiveBVHNode() :start(-1), end(-1), leftChild(nullptr), rightChild(nullptr), bBox() {}

	void destroy();
};

#endif // !BVH_H