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

// ������λ��������߱�־λΪ 1 (ͬʱʹ֮ǰ������λ��ı�־λΪ0)����
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
	vector<size_t> depthNumNodes; // ÿһ��İ˲����ڵ���
	vector<vector<SVONode>> depthNodes;
	// ��ʱ
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