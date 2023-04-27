#pragma once
#include "BaseModel.h"
#include <vector>
#include <limits>
#include <Eigen\Dense>

using std::vector;

#if defined(__clang__) || defined(__GNUC__)

#define _FORCE_INLINE_ __attribute__((always_inline))

#elif defined(_MSC_VER)

#define _FORCE_INLINE_ __forceinline

#endif

namespace detail {
	template <class T, T... inds, class F>
	constexpr __host__ __device__ _FORCE_INLINE_ void
		loop(std::integer_sequence<T, inds...>, F&& f) {
		(f(std::integral_constant<T, inds>{}), ...);
	}

	template <class T, T count, class F>
	constexpr __host__ __device__ _FORCE_INLINE_ void Loop(F&& f) {
		loop(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
	}
} // namespace detail


struct SVONode
{
	uint32_t mortonCode = 0;
	bool isLeaf = true;

	Eigen::Vector3f origin;
	float width;

	unsigned int parent{ UINT_MAX };
	unsigned int childs[8]{ UINT_MAX };
	unsigned int neighbors[27]{ UINT_MAX };

	SVONode() 
	{
		detail::Loop<unsigned int, 8>([&](auto i) {childs[i] = UINT_MAX; });
		detail::Loop<unsigned int, 27>([&](auto i) {neighbors[i] = UINT_MAX; });
	}
};

using thrust_edge = thrust::pair<Eigen::Vector3f, Eigen::Vector3f>;

struct SparseVoxelOctree : public BaseModel
{
private:
	int treeDepth = 0;
	size_t numTreeNodes;
	Eigen::Vector3i surfaceVoxelGridSize;
	vector<size_t> depthNumNodes; // ÿһ��İ˲����ڵ���
	vector<vector<SVONode>> SVONodes;

	// ��ʱ
	//vector<vector<uint32_t>> tempNodeArray;
	vector<SVONode> svoNodeArray;


	vector<thrust::pair<Eigen::Vector3f, uint32_t>> nodeVertexArray;
	vector<thrust::pair<thrust_edge, uint32_t>> nodeEdgeArray;
	/*vector<Eigen::Vector3f> nodeVertexArray;
	vector<Eigen::Vector3f> nodeEdgeArray;*/

public:
	SparseVoxelOctree() : treeDepth(0) {}
	SparseVoxelOctree(const std::string& filename,
		const Eigen::Vector3i& _gridSize) :BaseModel(filename), surfaceVoxelGridSize(_gridSize)
	{
		//depthNodes.resize(_depth);
	}
	SparseVoxelOctree(const std::string& filename,
		const int& _grid_x,
		const int& _grid_y,
		const int& _grid_z) :BaseModel(filename), surfaceVoxelGridSize(Eigen::Vector3i(_grid_x, _grid_y, _grid_z))
	{
		//depthNodes.resize(_depth);
	}

public:
	void createOctree();

	void writeTree(const std::string base_filename);

	void writeVoxel(const vector<uint32_t>& voxelArray, const std::string& base_filename, const float& width);

private:
	void meshVoxelize(const Eigen::Vector3i* d_surfaceVoxelGridSize,
		const Eigen::Vector3f* d_unitVoxelSize,
		const Eigen::Vector3f* d_gridOrigin,
		thrust::device_vector<uint32_t>& d_CNodeMortonArray); // construct nodes in `depth - 1`

	void constructNodeNeighbors(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
		thrust::device_vector<SVONode>& d_SVONodeArray);
	void constructNodeVertexAndEdge(thrust::device_vector<SVONode>& d_SVONodeArray);

	void constructNodeAtrributes(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
		thrust::device_vector<SVONode>& d_SVONodeArray);
};