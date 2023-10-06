#include "BVH.h"

void BVHAccel::buildBVH(std::vector<Triangle>& t,
	std::vector<BVHNode>& nodes,
	SplitMethod splitMethod = SplitMethod::SAH)
{
	switch (splitMethod)
	{
		case SplitMethod::SAH: buildSAH(t, nodes); break;
		case SplitMethod::NAIVE: buildNAIVE(t, nodes); break;
	}
}

void BVHAccel::buildSAH(std::vector<Triangle>& t,
	std::vector<BVHNode>& nodes)
{
	int primNum = t.size();
	int bvhNodeNum = 2 * primNum - 1;
	int maxDepth = int(ceil(log2f(bvhNodeNum)));

	nodes.resize(bvhNodeNum);
}

void BVHAccel::buildNAIVE(std::vector<Triangle>& t,
	std::vector<BVHNode>& nodes)
{
	int primNum = t.size();
	int bvhNodeNum = 2 * primNum - 1;
	if (bvhNodeNum == 0) return;
	nodes.resize(bvhNodeNum);

	int maxPerfectDepth = 1;
	while (bvhNodeNum >>= 1) { ++maxPerfectDepth; }
	bvhNodeNum = 2 * primNum - 1;
	int lastLayerNum = bvhNodeNum;
	if ((1 << maxPerfectDepth) - 1 != bvhNodeNum)
	{
		--maxPerfectDepth;
	}
	lastLayerNum -= ((1 << maxPerfectDepth) - 1);
	--maxPerfectDepth;

	//for (int i = 0; i < primNum; ++i)
	//{
	//	bboxes[0].Union(t[i].getBound());
	//}

	std::vector<BVHBuildInfo> infoStack(bvhNodeNum);
	int top = 0;

	//{start, end, index, depth, offset, preMiss, siblingIdx} [start,end)
	infoStack[top++] = { 0,primNum,0,0,0,-1,-1};

	while (top)
	{
		BVHBuildInfo info = infoStack[--top];
		int nodeId = info.idx;
		Bounds3& bBox = nodes[nodeId].bBox;
		bBox = Bounds3();
		int start = info.start, end = info.end, depth = info.depth, offset = info.offset;
		int lastLayerMax = (1 << (maxPerfectDepth + 1));
		int lastLayerStartIdx = lastLayerMax * offset / (1 << depth);
		int miss = -1;
		for (int i = start; i < end; ++i)
		{
			bBox.Union(t[i].getBound());
		}

		if (nodeId == 0) { miss = -1; }
		else if (!(offset & 1))
		{
			//miss = nodeId
			//	+ (1 << (maxPerfectDepth - depth)) - 1
			//	+ std::max(std::min(1 << (maxPerfectDepth - depth), lastLayerNum - lastLayerStartIdx), 0);
			miss = nodeId
				+ (1 << (maxPerfectDepth - depth + 1)) - 2
				+ std::max(std::min(1 << (maxPerfectDepth - depth + 1), lastLayerNum - lastLayerStartIdx), 0)
				+ 1;
			assert(miss == info.siblingIdx);
			miss = info.siblingIdx;
		}
		//else if (nodeId == 0)
		//{
		//	miss = -1;
		//}
		else
		{
			miss = info.preMiss;
		}

		assert(miss < bvhNodeNum);
		nodes[nodeId].miss = miss;
		nodes[nodeId].primitiveId = -1;

		if (end - start == 1)
		{
			nodes[nodeId].primitiveId = start;
		}

		else if (end - start == 2)
		{
			assert(end <= primNum);
			int leftChild = nodeId + 1;
			int rightChild = nodeId
					+ (1 << (maxPerfectDepth - depth)) - 1
					+ std::max(std::min(1 << (maxPerfectDepth - depth), lastLayerNum - lastLayerStartIdx), 0)
					+1;

			nodes[leftChild].primitiveId = start;
			nodes[leftChild].miss = rightChild;
			nodes[leftChild].bBox = t[start].getBound();

			nodes[rightChild].primitiveId = start + 1;
			nodes[rightChild].miss = miss;
			nodes[rightChild].bBox = t[start + 1].getBound();
		}

		else
		{
			int tNum = end - start;
			int log2tNum = log2(tNum);
			//int leftTNum = (1 << log2tNum);
			int lastLayerLStart = lastLayerStartIdx;
			int leftTNum = (1 << (maxPerfectDepth - depth - 1)) + std::max(std::min((lastLayerNum - lastLayerStartIdx)/2, 1 << (maxPerfectDepth - depth - 1)), 0);

			if (leftTNum == tNum) { leftTNum >>= 1; }

			int rightTNum = tNum - leftTNum;
			int leftStart = start, leftEnd = start + leftTNum,
				rightStart = leftEnd, rightEnd = end;
			int leftIdx = nodeId + 1,
				rightIdx = nodeId
				+ (1 << (maxPerfectDepth - depth)) - 1
				+ std::max(std::min(1 << (maxPerfectDepth - depth), lastLayerNum - lastLayerStartIdx), 0)
				+ 1;
			int maxIdx = bBox.MaxExtent();
			// sort the t[start] to t[end] by index "maxIdx" using lambda func
			std::sort(t.begin() + start, t.begin() + end, [maxIdx](Triangle lhs, Triangle rhs) {
				return lhs.getBound().Centroid()[maxIdx] < rhs.getBound().Centroid()[maxIdx];
				});
			//{start, end, index, depth, offset, preMiss, siblingIdx} [start,end)
			infoStack[top++] = { rightStart, rightEnd, rightIdx, depth + 1, 2 * offset + 1, miss, leftIdx };
			infoStack[top++] = { leftStart, leftEnd, leftIdx, depth + 1, 2 * offset, miss, rightIdx };
		}
	}
}