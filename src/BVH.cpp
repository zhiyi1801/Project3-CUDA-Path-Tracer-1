#include "BVH.h"

RecursiveBVHNode* BVHAccel::recursiveBuild(std::vector<Triangle>& t)
{
#if USE_SAH
	return recursiveBuildSAH(t, 0, t.size());
#else
	return recursiveBuildNaive(t, 0, t.size());
#endif // USE_SAH
}

//don't forget define destroy function !!!!!!!!!!!!!!!!!!!!!!!!!!
RecursiveBVHNode* BVHAccel::recursiveBuildSAH(std::vector<Triangle>& t, const int start, const int end)
{
	struct BucketInfo
	{
		int num;
		Bounds3 bBox;
		BucketInfo() : num(0), bBox() {}
	};
	std::vector<BucketInfo> buckets;
	buckets.resize(BUCKET_NUM);
	RecursiveBVHNode* root = new RecursiveBVHNode();
	glm::vec3 centerMin(FLT_MAX), centerMax(-FLT_MAX);
	root->start = start;
	root->end = end;
	for (int i = start; i < end; ++i)
	{
		root->bBox.Union(t[i].getBound());
		centerMin = glm::min(centerMin, t[i].Centroid());
		centerMax = glm::max(centerMax, t[i].Centroid());
	}

	Bounds3 centerBox(centerMin, centerMax);
	int maxExtent = centerBox.MaxExtent();

	if (end - start <= glm::max(MAX_PRIM, 1))
	{
		return root;
	}
	std::sort(t.begin() + start, t.begin() + end, [maxExtent](const Triangle& lhs, const Triangle& rhs) {
		return lhs.getBound().Centroid()[maxExtent] < rhs.getBound().Centroid()[maxExtent];
		});

	float Loss = FLT_MAX;
	int mid = 0;
	for (int i = start; i < end; ++i)
	{
		float offset = glm::clamp((t[i].Centroid()[maxExtent] - centerBox.pMin[maxExtent]) / (centerBox.pMax[maxExtent] - centerBox.pMin[maxExtent]), 0.f, 1.f);
		int bucketIdx = offset == 1.f ? BUCKET_NUM - 1 : static_cast<int>(offset * BUCKET_NUM);
		buckets[bucketIdx].num += 1;
		buckets[bucketIdx].bBox.Union(t[i].getBound());
	}

	for (int i = 0; i < BUCKET_NUM - 1; ++i)
	{
		Bounds3 bBoxL{}, bBoxR{};
		int numL = 0, numR = 0;
		float tempLoss;
		for (int j = 0; j <= i; ++j)
		{
			bBoxL.Union(buckets[j].bBox);
			numL += buckets[j].num;
		}
		for (int j = i + 1; j < BUCKET_NUM; ++j)
		{
			bBoxR.Union(buckets[j].bBox);
			numR += buckets[j].num;
		}
		tempLoss = (numL * bBoxL.SurfaceArea() + numR * bBoxR.SurfaceArea()) / root->bBox.SurfaceArea();
		if (tempLoss < Loss && numL != 0 && numR != 0)
		{
			Loss = tempLoss;
			mid = start + numL;
		}
	}
	assert(mid >= start && mid <= end);
	root->leftChild = recursiveBuildSAH(t, start, mid);
	root->rightChild = recursiveBuildSAH(t, mid, end);

	//root->bBox = Union(root->leftChild->bBox, root->rightChild->bBox);
	return root;
}

RecursiveBVHNode* BVHAccel::recursiveBuildNaive(std::vector<Triangle>& t, const int start, const int end)
{
	RecursiveBVHNode* root = new RecursiveBVHNode();
	root->start = start;
	root->end = end;
	for (int i = start; i < end; ++i)
	{
		root->bBox.Union(t[i].getBound());
	}
	if (end - start <= MAX_PRIM)
	{
		return root;
	}

	int maxExtent = root->bBox.MaxExtent();
	std::sort(t.begin() + start, t.begin() + end, [maxExtent](const Triangle& lhs, const Triangle& rhs) {
		return lhs.getBound().Centroid()[maxExtent] < rhs.getBound().Centroid()[maxExtent];
		});
	int mid = (start + end) / 2;
	root->leftChild = recursiveBuildNaive(t, start, mid);
	root->rightChild = recursiveBuildNaive(t, mid, end);

	//root->bBox = Union(root->leftChild->bBox, root->rightChild->bBox);
	return root;
}


int BVHAccel::recursiveBuildGpuBVHInfo(const RecursiveBVHNode* root, std::vector<GpuBVHNodeInfo>& bvhInfo, const int parent)
{
	if (parent == -1)
	{
		bvhInfo.clear();
	}
	if (!root)
	{
		return -1;
	}

	GpuBVHNodeInfo nodeInfo{};
	nodeInfo.BVHNode = root;
	nodeInfo.parent = parent;
	const int currentID = bvhInfo.size();
	bvhInfo.push_back(nodeInfo);
	if (root->leftChild)
	{
		assert(root->rightChild != nullptr);
		int left = recursiveBuildGpuBVHInfo(root->leftChild, bvhInfo, currentID);
		int right = recursiveBuildGpuBVHInfo(root->rightChild, bvhInfo, currentID);
		bvhInfo[currentID].left = left;
		bvhInfo[currentID].right = right;
	}
	
	return currentID;
}

void BVHAccel::buildGpuBVH(const std::vector<GpuBVHNodeInfo>& bvhInfo, std::vector<GpuBVHNode>& bvh)
{
	bvh.resize(bvhInfo.size());
	for (int i = 0; i < bvhInfo.size(); ++i)
	{
		bvh[i].bBox = bvhInfo[i].BVHNode->bBox;
		bvh[i].start = bvhInfo[i].BVHNode->start;
		bvh[i].end = bvhInfo[i].BVHNode->end;

		//since <<recursiveBuildGpuBVHInfo>> is a preorder traversal build
		bvh[i].hit = i + 1;
		if (i == bvhInfo.size() - 1) bvh[i].hit = -1;
		int parent = bvhInfo[i].parent;

		// if i is the left child of parent, miss index is the right child of parent
		if (i == 0)
		{
			bvh[i].miss = -1;
		}
		else if (i == bvhInfo[parent].left)
		{
			bvh[i].miss = bvhInfo[parent].right;
		}
		// if i is the right child of parent, miss link is parent's miss link (there is a fact that parent < i so bvh[parent].miss is assigned)
		else
		{
			bvh[i].miss = bvh[parent].miss;
		}
	}
}

void BVHAccel::buildGpuMTBVH(const std::vector<GpuBVHNodeInfo>& bvhInfo, std::vector<GpuBVHNode>& MTbvh)
{
	int bvhSize = bvhInfo.size();
	MTbvh.resize(6 * bvhSize);
	//the order of 6 direction is [x,y,z,-x,-y,-z]
	for (int dir = 0; dir < 6; ++dir)
	{
		int offset = dir * bvhSize;
		int axis = dir % 3;
		int sign = (dir < 3 ? 1 : -1);
		for (int i = 0; i < bvhSize; ++i)
		{
			MTbvh[offset + i].bBox = bvhInfo[i].BVHNode->bBox;
			MTbvh[offset + i].start = bvhInfo[i].BVHNode->start;
			MTbvh[offset + i].end = bvhInfo[i].BVHNode->end;
			int left = bvhInfo[i].left, right = bvhInfo[i].right;
			int parent = bvhInfo[i].parent;

			//assign hit link
			if (left != -1)
			{
				if ((bvhInfo[left].BVHNode->bBox.Centroid()[axis] * sign > bvhInfo[right].BVHNode->bBox.Centroid()[axis] * sign))
				{
					std::swap(left, right);
				}
				MTbvh[offset + i].hit = left;
			}
			else
			{
				if (parent == -1)
				{
					MTbvh[offset + i].hit = -1;
				}
				else if (i == MTbvh[offset + parent].hit)
				{
					MTbvh[offset + i].hit = (i == bvhInfo[parent].left) ? bvhInfo[parent].right : bvhInfo[parent].left;
				}

				else
				{
					MTbvh[offset + i].hit = MTbvh[offset + parent].miss;
				}
			}

			//assign miss link
			if (i == 0)
			{
				MTbvh[offset + i].miss = -1;
			}
			else if (i == MTbvh[offset + parent].hit)
			{
				MTbvh[offset + i].miss = (i == bvhInfo[parent].left) ? bvhInfo[parent].right : bvhInfo[parent].left;
			}
			else
			{
				MTbvh[offset + i].miss = MTbvh[offset + parent].miss;
			}
		}
	}
}

void RecursiveBVHNode::destroy()
{
	if (!this) return;

	if (this->leftChild)
	{
		assert(this->rightChild != nullptr);
		this->leftChild->destroy();
		this->rightChild->destroy();
	}
	delete this;
}