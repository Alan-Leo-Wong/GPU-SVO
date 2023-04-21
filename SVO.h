#pragma once
#include "BaseModel.h"
#include <vector>
#include <limits>
#include <Eigen\Dense>

using std::vector;

typedef struct SparseVoxelOctreeNode
{
	uint32_t mortonCode = 0;

	Eigen::Vector3f origin;
	float width;

	unsigned int parent = UINT_MAX;
	unsigned int childs[8] = { UINT_MAX };
	unsigned int neighbors[27] = { UINT_MAX };
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

	bool constructFineNodes(thrust::device_vector<uint32_t>& d_surfaceNodeParentArray,
		thrust::device_vector<thrust::device_vector<size_t>>& d_allMorton2Idx); // construct nodes in `depth - 1`

	void createOctree();
};