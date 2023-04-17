#pragma once
#include "BaseModel.h"
#include <vector>
#include <Eigen\Dense>

using std::vector;

typedef struct SparseVoxelOctreeNode
{
	Eigen::Vector3f origin;
	float width;

	int parent = -1;
	int childs[8] = { -1 };
	int neighbors[27] = { -1 };
}SVONode;

struct SparseVoxelOctree : public BaseModel
{
private:
	int depth;
	Eigen::Vector3i surfaceVoxelGridSize;
	vector<vector<SVONode>> depthNodes;

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
	void constructFineNodes(); // construct nodes in `depth - 1`
};