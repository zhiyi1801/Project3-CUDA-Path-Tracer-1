#pragma once
#include"utilities.h"

struct BVHBuildNode;

struct BVHPrimitiveInfo;

int leafNodes, totalLeafNodes, totalPrimitives, interiorNodes;

class BVHAccel 
{	
public:
	enum class SplitMethod {NAIVE, SAH};
};