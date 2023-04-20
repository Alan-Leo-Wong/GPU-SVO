#pragma once
#include "BaseModel.h"
#include <vector>
#include <Eigen\Dense>
#define MORTON_32_FLAG 0x80000000

using std::vector;

typedef struct SparseVoxelOctreeNode
{
	Eigen::Vector3f origin;
	float width;
	uint32_t mortonCode;

	int parent = -1;
	int childs[8] = { -1 };
	int neighbors[27] = { -1 };
}SVONode;

// 右移三位，并让最高标志位为 1 (同时使之前右移三位后的标志位为0)即可
__inline__ CUDA_CALLABLE_MEMBER uint32_t getParentMorton(const uint32_t morton)
{
	return ((morton >> 3) & 0x8fffffff);
}

__inline__ CUDA_CALLABLE_MEMBER bool isSameParent(const uint32_t morton_1, const uint32_t morton_2)
{
	return getParentMorton(morton_1) == getParentMorton(morton_2);
}

struct SparseVoxelOctree : public BaseModel
{
private:
	int depth;
	Eigen::Vector3i surfaceVoxelGridSize;
	vector<size_t> depthNumNodes; // 每一层的八叉树节点数
	vector<vector<SVONode>> depthNodes;
	// 临时
	vector<vector<uint32_t>> tempNodeArray;

public:
	SparseVoxelOctree() : depth(0) {}
	SparseVoxelOctree(const int& _depth,
		const Eigen::Vector3i& _gridSize) :depth(_depth), surfaceVoxelGridSize(_gridSize)
	{
		depthNodes.resize(_depth);
	}
	SparseVoxelOctree(const int& _depth,
		const int& _grid_x,
		const int& _grid_y,
		const int& _grid_z) :depth(_depth), surfaceVoxelGridSize(Eigen::Vector3i(_grid_x, _grid_y, _grid_z))
	{
		depthNodes.resize(_depth);
	}

public:

	bool constructFineNodes(thrust::device_vector<uint32_t>& refineNodeParentArray); // construct nodes in `depth - 1`

	void createOctree();
};