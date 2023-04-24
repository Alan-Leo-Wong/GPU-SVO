#include "IO.h"
#include "SVO.h"
#include "CUDAUtil.h"
#include "MortonLUT.h"
#include "libmorton\morton.h"
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <crt/device_functions.h>
namespace cg = cooperative_groups;

//thrust::device_vector<size_t> d_morton2Idx; // 存储某一层morton code到nodeArray下标的映射
//thrust::device_vector<thrust::device_vector<size_t>> d_allMorton2Idx; // 存储所有层morton code到nodeArray下标的映射
//thrust::device_vector<thrust::device_vector<SVONode>> d_allSVONodeArray; // save all sparse octree nodes

// Estimate best block and grid size using CUDA Occupancy Calculator
int blockSize;   // The launch configurator returned block size 
int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
int gridSize;    // The actual grid size needed, based on input size 

// 右移三位，并让最高标志位为 1 (同时使之前右移三位后的标志位为0)即可
CUDA_CALLABLE_MEMBER uint32_t getParentMorton(const uint32_t morton)
{
	return ((morton >> 3) & 0x8fffffff);
}

CUDA_CALLABLE_MEMBER bool isSameParent(const uint32_t morton_1, const uint32_t morton_2)
{
	return getParentMorton(morton_1) == getParentMorton(morton_2);
}

template <typename T>
struct scanMortonFlag : public thrust::unary_function<T, T> {
	__host__ __device__ T operator()(const T& x) {
		// printf("%lu %d\n", b, (b >> 31) & 1);
		return (x >> 31) & 1;
	}
};

__global__ void surfaceVoxelize(const int nTris,
	const Eigen::Vector3i d_surfaceVoxelGridSize,
	const Eigen::Vector3f d_gridOrigin,
	const Eigen::Vector3f d_unitVoxelSize,
	float* d_triangle_data,
	uint32_t* d_voxelArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	Eigen::Vector3f delta_p{ d_unitVoxelSize.x(), d_unitVoxelSize.y(), d_unitVoxelSize.z() };
	Eigen::Vector3i grid_max{ d_surfaceVoxelGridSize.x() - 1, d_surfaceVoxelGridSize.y() - 1, d_surfaceVoxelGridSize.z() - 1 }; // grid max (grid runs from 0 to gridsize-1)

	while (tid < nTris) { // every thread works on specific triangles in its stride
		size_t t = tid * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using modelBBox
		Eigen::Vector3f v0 = Eigen::Vector3f(d_triangle_data[t], d_triangle_data[t + 1], d_triangle_data[t + 2]) - d_gridOrigin;
		Eigen::Vector3f v1 = Eigen::Vector3f(d_triangle_data[t + 3], d_triangle_data[t + 4], d_triangle_data[t + 5]) - d_gridOrigin;
		Eigen::Vector3f v2 = Eigen::Vector3f(d_triangle_data[t + 6], d_triangle_data[t + 7], d_triangle_data[t + 8]) - d_gridOrigin;
		// Edge vectors
		Eigen::Vector3f e0 = v1 - v0;
		Eigen::Vector3f e1 = v2 - v1;
		Eigen::Vector3f e2 = v0 - v2;
		// Normal vector pointing up from the triangle
		Eigen::Vector3f n = e0.cross(e1).normalized();

		// COMPUTE TRIANGLE BBOX IN GRID
		// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
		AABox<Eigen::Vector3f> t_bbox_world(fminf(v0, fminf(v1, v2)), fmaxf(v0, fmaxf(v1, v2)));
		// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
		AABox<Eigen::Vector3i> t_bbox_grid;
		t_bbox_grid.min = clamp(
			Eigen::Vector3i(floor(t_bbox_world.min.x() / d_unitVoxelSize.x()), floor(t_bbox_world.min.y() / d_unitVoxelSize.y()), floor(t_bbox_world.min.z() / d_unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);
		t_bbox_grid.max = clamp(
			Eigen::Vector3i(ceil(t_bbox_world.max.x() / d_unitVoxelSize.x()), ceil(t_bbox_world.max.y() / d_unitVoxelSize.y()), ceil(t_bbox_world.max.z() / d_unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);

		// PREPARE PLANE TEST PROPERTIES
		Eigen::Vector3f c(0.0f, 0.0f, 0.0f);
		if (n.x() > 0.0f) { c.x() = d_unitVoxelSize.x(); }
		if (n.y() > 0.0f) { c.y() = d_unitVoxelSize.y(); }
		if (n.z() > 0.0f) { c.z() = d_unitVoxelSize.z(); }
		float d1 = n.dot((c - v0));
		float d2 = n.dot(((delta_p - c) - v0));

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		Eigen::Vector2f n_xy_e0(-1.0f * e0.y(), e0.x());
		Eigen::Vector2f n_xy_e1(-1.0f * e1.y(), e1.x());
		Eigen::Vector2f n_xy_e2(-1.0f * e2.y(), e2.x());
		if (n.z() < 0.0f)
		{
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		float d_xy_e0 = (-1.0f * n_xy_e0.dot(Eigen::Vector2f(v0.x(), v0.y()))) + fmaxf(0.0f, d_unitVoxelSize.x() * n_xy_e0[0]) + fmaxf(0.0f, d_unitVoxelSize.y() * n_xy_e0[1]);
		float d_xy_e1 = (-1.0f * n_xy_e1.dot(Eigen::Vector2f(v1.x(), v1.y()))) + fmaxf(0.0f, d_unitVoxelSize.x() * n_xy_e1[0]) + fmaxf(0.0f, d_unitVoxelSize.y() * n_xy_e1[1]);
		float d_xy_e2 = (-1.0f * n_xy_e2.dot(Eigen::Vector2f(v2.x(), v2.y()))) + fmaxf(0.0f, d_unitVoxelSize.x() * n_xy_e2[0]) + fmaxf(0.0f, d_unitVoxelSize.y() * n_xy_e2[1]);
		// YZ plane
		Eigen::Vector2f n_yz_e0(-1.0f * e0.z(), e0.y());
		Eigen::Vector2f n_yz_e1(-1.0f * e1.z(), e1.y());
		Eigen::Vector2f n_yz_e2(-1.0f * e2.z(), e2.y());
		if (n.x() < 0.0f) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		float d_yz_e0 = (-1.0f * n_yz_e0.dot(Eigen::Vector2f(v0.y(), v0.z()))) + fmaxf(0.0f, d_unitVoxelSize.y() * n_yz_e0[0]) + fmaxf(0.0f, d_unitVoxelSize.z() * n_yz_e0[1]);
		float d_yz_e1 = (-1.0f * n_yz_e1.dot(Eigen::Vector2f(v1.y(), v1.z()))) + fmaxf(0.0f, d_unitVoxelSize.y() * n_yz_e1[0]) + fmaxf(0.0f, d_unitVoxelSize.z() * n_yz_e1[1]);
		float d_yz_e2 = (-1.0f * n_yz_e2.dot(Eigen::Vector2f(v2.y(), v2.z()))) + fmaxf(0.0f, d_unitVoxelSize.y() * n_yz_e2[0]) + fmaxf(0.0f, d_unitVoxelSize.z() * n_yz_e2[1]);
		// ZX plane																							 													  
		Eigen::Vector2f n_zx_e0(-1.0f * e0.x(), e0.z());
		Eigen::Vector2f n_zx_e1(-1.0f * e1.x(), e1.z());
		Eigen::Vector2f n_zx_e2(-1.0f * e2.x(), e2.z());
		if (n.y() < 0.0f) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		float d_xz_e0 = (-1.0f * n_zx_e0.dot(Eigen::Vector2f(v0.z(), v0.x()))) + fmaxf(0.0f, d_unitVoxelSize.z() * n_zx_e0[0]) + fmaxf(0.0f, d_unitVoxelSize.x() * n_zx_e0[1]);
		float d_xz_e1 = (-1.0f * n_zx_e1.dot(Eigen::Vector2f(v1.z(), v1.x()))) + fmaxf(0.0f, d_unitVoxelSize.z() * n_zx_e1[0]) + fmaxf(0.0f, d_unitVoxelSize.x() * n_zx_e1[1]);
		float d_xz_e2 = (-1.0f * n_zx_e2.dot(Eigen::Vector2f(v2.z(), v2.x()))) + fmaxf(0.0f, d_unitVoxelSize.z() * n_zx_e2[0]) + fmaxf(0.0f, d_unitVoxelSize.x() * n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (uint16_t z = t_bbox_grid.min.z(); z <= t_bbox_grid.max.z(); z++) {
			for (uint16_t y = t_bbox_grid.min.y(); y <= t_bbox_grid.max.y(); y++) {
				for (uint16_t x = t_bbox_grid.min.x(); x <= t_bbox_grid.max.x(); x++) {
					// if (checkBit(voxel_table, location)){ continue; }
					// TRIANGLE PLANE THROUGH BOX TEST
					Eigen::Vector3f p(x * d_unitVoxelSize.x(), y * d_unitVoxelSize.y(), z * d_unitVoxelSize.z());
					float nDOTp = n.dot(p);
					if ((nDOTp + d1) * (nDOTp + d2) > 0.0f) { continue; }

					// PROJECTION TESTS
					// XY
					Eigen::Vector2f p_xy(p.x(), p.y());
					if ((n_xy_e0.dot(p_xy) + d_xy_e0) < 0.0f) { continue; }
					if ((n_xy_e1.dot(p_xy) + d_xy_e1) < 0.0f) { continue; }
					if ((n_xy_e2.dot(p_xy) + d_xy_e2) < 0.0f) { continue; }

					// YZ
					Eigen::Vector2f p_yz(p.y(), p.z());
					if ((n_yz_e0.dot(p_yz) + d_yz_e0) < 0.0f) { continue; }
					if ((n_yz_e1.dot(p_yz) + d_yz_e1) < 0.0f) { continue; }
					if ((n_yz_e2.dot(p_yz) + d_yz_e2) < 0.0f) { continue; }

					// XZ	
					Eigen::Vector2f p_zx(p.z(), p.x());
					if ((n_zx_e0.dot(p_zx) + d_xz_e0) < 0.0f) { continue; }
					if ((n_zx_e1.dot(p_zx) + d_xz_e1) < 0.0f) { continue; }
					if ((n_zx_e2.dot(p_zx) + d_xz_e2) < 0.0f) { continue; }

					//size_t mortonCode = mortonEncode_LUT(x, y, z);
					uint32_t mortonCode = mortonEncode_LUT(x, y, z);
					atomicExch(d_voxelArray + mortonCode, mortonCode | E_MORTON_32_FLAG); // 最高位设置为1，代表这是个表面的voxel
				}
			}
		}
		tid += stride;
	}
}

void SparseVoxelOctree::meshVoxelize(const Eigen::Vector3i& d_surfaceVoxelGridSize,
	const Eigen::Vector3f& d_unitVoxelSize,
	const Eigen::Vector3f& d_gridOrigin,
	thrust::device_vector<uint32_t>& d_CNodeMortonArray)
{
	thrust::device_vector<Eigen::Vector3f> d_triangleThrustVec;
	const size_t faces = idx2Points.size();
	for (int i = 0; i < faces; ++i)
	{
		d_triangleThrustVec.push_back(modelPoints[idx2Points[i].x()]);
		d_triangleThrustVec.push_back(modelPoints[idx2Points[i].y()]);
		d_triangleThrustVec.push_back(modelPoints[idx2Points[i].z()]);
	}
	float* d_triangleData = (float*)thrust::raw_pointer_cast(&(d_triangleThrustVec[0]));
	getOccupancyMaxPotentialBlockSize(nModelTris, minGridSize, blockSize, gridSize, surfaceVoxelize, 0, 0);
	surfaceVoxelize << <gridSize, blockSize >> > (nModelTris, d_surfaceVoxelGridSize,
		d_gridOrigin, d_unitVoxelSize, d_triangleData, d_CNodeMortonArray.data().get());
}

__global__ void compactArray(const int n,
	const bool* d_isValidArray,
	const uint32_t* d_dataArray,
	const size_t* d_esumDataArray,
	uint32_t* d_pactDataArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < n && d_isValidArray[tid])
		d_pactDataArray[d_esumDataArray[tid]] = d_dataArray[tid];
}

// 计算表面voxel共对应多少个八叉树节点
__global__ void cpNumNodes(const size_t n,
	const uint32_t* d_pactDataArray,
	short int* d_nNodesArray,
	uint32_t* d_parentMortonArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= 1 && tid < n)
	{
		if (isSameParent(d_pactDataArray[tid], d_pactDataArray[tid - 1])) d_nNodesArray[tid] = 0;
		else
		{
			const uint32_t parentMorton = getParentMorton(d_pactDataArray[tid]);
			d_parentMortonArray[parentMorton] = parentMorton;
			d_nNodesArray[tid] = 8;
		}
	}
}

// 根据d_sumNodesArray和d_pactVoxelArray(存储了莫顿码)设置节点数组，本质也是compact
// blockSize必须是32的整数倍，因为最底层节点个数是8的整数倍
__global__ void createNode(const size_t nNodes,
	const size_t pactSize,
	const size_t* d_sumNodesArray,
	const uint32_t* d_pactDataArray,
	const Eigen::Vector3f d_gridOrigin,
	const float d_width,
	SVONode* d_nodeArray/*,
	size_t* d_morton2Idx*/)
{
	extern __shared__ uint32_t sh_nodeMorton[]; // blockSize / 8，数值为8的整数倍

	cg::thread_block ctb = cg::this_thread_block();
	cg::thread_group tile8 = cg::tiled_partition(ctb, 8);

	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	/*sh_nodeMorton[threadIdx.x / 8] = 0; // 默认是0
	__syncthreads();*/

	if (tid < nNodes)
	{
		uint16_t x, y, z;
		if (tid < pactSize)
		{
			const uint32_t key = d_pactDataArray[tid] & LOWER_3BIT_MASK;
			const uint32_t morton = d_pactDataArray[tid] & D_MORTON_32_FLAG; // 去除符号位的实际莫顿码
			// 得到mortonCode对应的实际存储节点的位置
			const size_t address = d_sumNodesArray[tid] + key;

			d_nodeArray[address].mortonCode = morton;
			morton3D_32_decode(morton, x, y, z);
			d_nodeArray[address].origin = d_gridOrigin + d_width * Eigen::Vector3f((float)x, (float)y, (float)z);
			d_nodeArray[address].width = d_width;

			//d_morton2Idx[morton] = address; // 莫顿码到节点数组下标的映射

			// (d_pactDataArray[tid] / 8) * 8 得到d_pactDataArray[tid](莫顿码)对应的以8为整数倍的下标
			// 用于计算那些在这个if中没计算出来的节点莫顿码
			if ((morton / 8) * 8 != 0) sh_nodeMorton[threadIdx.x / 8] = (morton / 8) * 8; // 八个节点为一组
		}
		cg::sync(tile8);
		//__syncthreads();

		// 计算不在voxel里的节点的莫顿码
		if (d_nodeArray[tid].mortonCode == 0)
		{
			const uint32_t morton = (tid & LOWER_3BIT_MASK) + sh_nodeMorton[threadIdx.x / 8];
			d_nodeArray[tid].mortonCode = morton;
			morton3D_32_decode(morton, x, y, z);
			d_nodeArray[tid].origin = d_gridOrigin + d_width * Eigen::Vector3f((float)x, (float)y, (float)z);
			d_nodeArray[tid].width = d_width;

			//d_morton2Idx[morton] = tid;
		}
	}
}

__global__ void createNode(const size_t nNodes,
	const size_t pactSize,
	const size_t d_preChildDepthTreeNodes, // 子节点层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
	const size_t d_preDepthTreeNodes, // 当前层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
	const size_t* d_sumNodesArray, // 这一层的节点数量inclusive scan数组
	const uint32_t* d_pactDataArray,
	const Eigen::Vector3f d_gridOrigin,
	const float d_width,
	SVONode* d_nodeArray,
	SVONode* d_childArray/*,
	size_t* d_morton2Idx*/)
{
	extern __shared__ uint32_t sh_nodeMorton[]; // blockSize / 8，数值为8的整数倍

	cg::thread_block ctb = cg::this_thread_block();
	cg::thread_group tile8 = cg::tiled_partition(ctb, 8);

	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	/*sh_nodeMorton[threadIdx.x / 8] = 0; // 默认是0
	__syncthreads();*/

	if (tid < nNodes)
	{
		uint16_t x, y, z;
		if (tid < pactSize)
		{
			const uint32_t key = d_pactDataArray[tid] & LOWER_3BIT_MASK;
			const uint32_t morton = d_pactDataArray[tid] & D_MORTON_32_FLAG;
			const size_t address = d_sumNodesArray[tid] + key;

			d_nodeArray[address].mortonCode = morton;
			morton3D_32_decode(morton, x, y, z);
			d_nodeArray[address].origin = d_gridOrigin + d_width * Eigen::Vector3f((float)x, (float)y, (float)z);
			d_nodeArray[address].width = d_width;
			d_nodeArray[address].isLeaf = false;

			for (int i = 0; i < 8; ++i)
			{
				d_nodeArray[address].childs[i] = d_preChildDepthTreeNodes + tid * 8 + i;
				d_childArray[tid * 8 + i].parent = d_preDepthTreeNodes + tid;
			}

			//d_morton2Idx[morton] = address; // 莫顿码到节点数组下标的映射

			// (d_pactDataArray[tid] / 8) * 8 得到d_pactDataArray[tid](莫顿码)对应的以8为整数倍的下标
			// 用于计算那些在这个if中没计算出来的节点莫顿码
			if ((morton / 8) * 8 != 0) sh_nodeMorton[threadIdx.x / 8] = (morton / 8) * 8; // 八个节点为一组
		}
		cg::sync(tile8);
		//__syncthreads();

		// 计算不在voxel里的节点的莫顿码(此时的节点为叶子节点)
		if (d_nodeArray[tid].mortonCode == 0)
		{
			const uint32_t morton = (tid & LOWER_3BIT_MASK) + sh_nodeMorton[threadIdx.x / 8];

			d_nodeArray[tid].mortonCode = morton;
			morton3D_32_decode(morton, x, y, z);
			d_nodeArray[tid].origin = d_gridOrigin + d_width * Eigen::Vector3f((float)x, (float)y, (float)z);
			d_nodeArray[tid].width = d_width;

			//d_morton2Idx[morton] = tid;
		}
	}
}

void SparseVoxelOctree::createOctree()
{
	assert(surfaceVoxelGridSize.x() >= 1 && surfaceVoxelGridSize.y() >= 1 && surfaceVoxelGridSize.z() >= 1);
	size_t gridCNodeSize = (size_t)mortonEncode_LUT((uint16_t)(surfaceVoxelGridSize.x() - 1), (uint16_t)(surfaceVoxelGridSize.y() - 1), (uint16_t)(surfaceVoxelGridSize.z() - 1)) + 1;
	//size_t gridCNodeSize = (size_t)((size_t)surfaceVoxelGridSize.x() * (size_t)surfaceVoxelGridSize.y() * (size_t)surfaceVoxelGridSize.z());
	size_t gridTreeNodeSize = gridCNodeSize + 8 - (gridCNodeSize % 8);
	///	TODO: 调成同样大小(只要把模型的bbox设置为立方体就可以了，具体可参考cuda_voxelizer中的createMeshBBCube方法)
	Eigen::Vector3f unitVoxelSize = Eigen::Vector3f((modelBBox.max.x() - modelBBox.min.x()) / surfaceVoxelGridSize.x(),
		(modelBBox.max.y() - modelBBox.min.y()) / surfaceVoxelGridSize.y(),
		(modelBBox.max.z() - modelBBox.min.z()) / surfaceVoxelGridSize.z());
	float unitNodeWidth = unitVoxelSize.x();

	Eigen::Vector3i d_surfaceVoxelGridSize;
	CUDA_CHECK(cudaMemcpyToSymbol(d_surfaceVoxelGridSize, &surfaceVoxelGridSize, sizeof(Eigen::Vector3i), 0, cudaMemcpyHostToDevice));
	Eigen::Vector3f d_gridOrigin;
	CUDA_CHECK(cudaMemcpyToSymbol(d_gridOrigin, &modelBBox.min, sizeof(Eigen::Vector3f), 0, cudaMemcpyHostToDevice));
	Eigen::Vector3f d_unitVoxelSize;
	CUDA_CHECK(cudaMemcpyToSymbol(d_unitVoxelSize, &unitVoxelSize, sizeof(Eigen::Vector3f), 0, cudaMemcpyHostToDevice));
	float d_unitNodeWidth;
	CUDA_CHECK(cudaMemcpyToSymbol(d_unitNodeWidth, &unitNodeWidth, sizeof(float), 0, cudaMemcpyHostToDevice));

	thrust::device_vector<uint32_t> d_CNodeMortonArray(gridCNodeSize, 0);
	thrust::device_vector<bool> d_isValidCNodeArray;
	thrust::device_vector<size_t> d_esumCNodesArray; // exclusive scan
	thrust::device_vector<uint32_t> d_pactCNodeArray;
	thrust::device_vector<short int> d_numTreeNodesArray; // 节点数量记录数组
	thrust::device_vector<size_t> d_sumTreeNodesArray; // inlusive scan
	thrust::device_vector<size_t> d_esumTreeNodesArray; // 存储每一层节点数量的exclusive scan数组
	thrust::device_vector<SVONode> d_nodeArray; // 存储某一层的节点数组
	thrust::device_vector<SVONode> d_SVONodeArray; // save all sparse octree nodes

	// mesh voxelize
	//size_t voxelArraySize = (size_t)((size_t)(surfaceVoxelGridSize.x() + 1) * (size_t)(surfaceVoxelGridSize.y() + 1) * (size_t)(surfaceVoxelGridSize.z() + 1));
	// 不需要+1（莫顿码为0代表坐标位于原点的第一个八叉树节点，八个顶点坐标需要令算）
	resizeThrust(d_CNodeMortonArray, gridCNodeSize, (uint32_t)0);
	meshVoxelize(d_surfaceVoxelGridSize, d_unitVoxelSize, d_gridOrigin, d_CNodeMortonArray);

	//// create octree
	//// 最后出来的树会比原始模型大7个格子, TODO: 到最顶层的时候只建立与模型bb相同的一个格子，它的周围7个格子不要建出来
	//while (true)
	//{
	//	// compute the number of 'coarse nodes'(eg: voxels)
	//	size_t pactCNodeArraySize = 0;
	//	resizeThrust(d_isValidCNodeArray, gridCNodeSize);
	//	resizeThrust(d_esumCNodesArray, gridCNodeSize);
	//	thrust::transform(d_CNodeMortonArray.begin(), d_CNodeMortonArray.end(), d_isValidCNodeArray.begin(), 0, scanMortonFlag<uint32_t>());
	//	thrust::exclusive_scan(d_isValidCNodeArray.begin(), d_isValidCNodeArray.end(), d_esumCNodesArray.begin());
	//	size_t numCNodes = *(d_esumCNodesArray.end()) + d_isValidCNodeArray[gridCNodeSize - 1];
	//	if (!numCNodes) { printf("Sparse Voxel Octree depth: %d\n", treeDepth); break; }

	//	treeDepth++;

	//	// compact coarse node array
	//	resizeThrust(d_pactCNodeArray, numCNodes);
	//	getOccupancyMaxPotentialBlockSize(gridCNodeSize, minGridSize, blockSize, gridSize, compactArray, 0, 0);
	//	compactArray << <gridSize, blockSize >> > (gridCNodeSize, d_isValidCNodeArray.data().get(),
	//		d_CNodeMortonArray.data().get(), d_esumCNodesArray.data().get(), d_pactCNodeArray.data().get());

	//	// compute the number of (real)octree nodes by coarse node array
	//	// and set parent's morton code to 'd_CNodeMortonArray'
	//	size_t numNodes = 1;
	//	if (numCNodes > 1)
	//	{
	//		resizeThrust(d_numTreeNodesArray, numCNodes, (short int)0);
	//		resizeThrust(d_CNodeMortonArray, gridTreeNodeSize, (uint32_t)0); // 此时用于记录父节点层的coarse node
	//		getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, cpNumNodes, 0, 0);
	//		const uint32_t firstMortonCode = getParentMorton(d_pactCNodeArray[0]);
	//		d_CNodeMortonArray[firstMortonCode] = firstMortonCode;
	//		cpNumNodes << <gridSize, blockSize >> > (numCNodes, d_pactCNodeArray.data().get(), d_numTreeNodesArray.data().get(), d_CNodeMortonArray.data().get());
	//		resizeThrust(d_sumTreeNodesArray, numCNodes, (size_t)0); // inlusive scan
	//		thrust::inclusive_scan(d_numTreeNodesArray.begin(), d_numTreeNodesArray.end(), d_sumTreeNodesArray.begin());
	//		numNodes = *(d_sumTreeNodesArray.end()) + 8;
	//		depthNumNodes.emplace_back(numNodes);
	//	}

	//	// set octree node array
	//	resizeThrust(d_nodeArray, numNodes, SVONode());
	//	uint32_t maxMortonCode = (*d_pactCNodeArray.end()) & D_MORTON_32_FLAG;
	//	//resizeThrust(d_morton2Idx, gridTreeNodeSize);
	//	blockSize = 128; gridSize = (numNodes + blockSize - 1) / blockSize;
	//	if (treeDepth < 2)
	//	{
	//		createNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize >> > (numNodes, numCNodes, d_sumTreeNodesArray.data().get(),
	//			d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_nodeArray.data().get()/*, d_morton2Idx.data().get()*/);
	//		d_esumTreeNodesArray.push_back(0);
	//	}
	//	else
	//	{
	//		//createNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize >> > (numNodes, numCNodes, *(d_esumTreeNodesArray.end() - 1), *(d_esumTreeNodesArray.end()),
	//		//	d_sumTreeNodesArray.data().get(), d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_nodeArray.data().get(),
	//		//	(d_allSVONodeArray.data() + d_allSVONodeArray.size() - 1)->data().get()/*, d_morton2Idx.data().get()*/);
	//		createNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize >> > (numNodes, numCNodes, *(d_esumTreeNodesArray.end() - 1), *(d_esumTreeNodesArray.end()),
	//			d_sumTreeNodesArray.data().get(), d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_nodeArray.data().get(),
	//			(d_SVONodeArray.data() + (*d_esumTreeNodesArray.end() - 1)).get()/*, d_morton2Idx.data().get()*/);
	//	}
	//	d_SVONodeArray.insert(d_SVONodeArray.end(), d_nodeArray.begin(), d_nodeArray.end());
	//	//d_allSVONodeArray.push_back(d_nodeArray);
	//	//d_allMorton2Idx.push_back(d_morton2Idx);
	//	d_esumTreeNodesArray.push_back(numNodes + (*d_esumTreeNodesArray.end()));

	//	// special condition
	//	if (treeDepth == 1 && gridCNodeSize == 1) { printf("Sparse Voxel Octree depth: %d\n", treeDepth); break; }
	//	// resize parent array 'd_CNodeMortonArray' to nexe loop
	//	uint32_t numParentNodes = *thrust::max_element(d_CNodeMortonArray.begin(), d_CNodeMortonArray.end());
	//	bool isValidMorton = (numParentNodes >> 31) & 1;
	//	numParentNodes = (numParentNodes & D_MORTON_32_FLAG) + isValidMorton; // '+ isValidMorton' to prevent '(numParentNodes & D_MORTON_32_FLAG) = 0'
	//	if (numParentNodes != 0)
	//	{
	//		d_CNodeMortonArray.resize(numParentNodes); d_CNodeMortonArray.shrink_to_fit();
	//		unitNodeWidth *= 2.0; CUDA_CHECK(cudaMemcpyToSymbol(d_unitNodeWidth, &unitNodeWidth, sizeof(float), 0, cudaMemcpyHostToDevice));
	//		gridCNodeSize = numNodes / 8; gridTreeNodeSize = gridCNodeSize + 8 - (gridCNodeSize % 8);
	//		if (gridCNodeSize == 0) { printf("Sparse Voxel Octree depth: %d\n", treeDepth); break; }
	//	}
	//	else { printf("Sparse Voxel Octree depth: %d\n", treeDepth); break; }
	//}
	//numTreeNodes = d_esumTreeNodesArray[treeDepth];
	///// TODO: copy to host
	//svoNodeArray.resize(numTreeNodes);
	//CUDA_CHECK(cudaMemcpy(svoNodeArray.data(), d_SVONodeArray.data().get(), sizeof(SVONode) * numTreeNodes, cudaMemcpyDeviceToHost));
	//auto freeResOfCreateTree = [&]()
	//{
	//	cleanupThrust(d_CNodeMortonArray);
	//	cleanupThrust(d_isValidCNodeArray);
	//	cleanupThrust(d_esumCNodesArray);
	//	cleanupThrust(d_pactCNodeArray);
	//	cleanupThrust(d_numTreeNodesArray);
	//	cleanupThrust(d_sumTreeNodesArray);
	//	cleanupThrust(d_nodeArray);
	//};
	//freeResOfCreateTree();

	//constructNodeAtrributes(d_esumTreeNodesArray);
	//cleanupThrust(d_numTreeNodesArray);
}

__global__ void findNeighbors(const size_t nNodes,
	const size_t preESumTreeNodes,
	SVONode* d_nodeArray)
{
	size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (tid_x < nNodes && tid_y < 27)
	{
		SVONode t = d_nodeArray[preESumTreeNodes + tid_x];
		SVONode p = d_nodeArray[t.parent];
		const uint8_t key = (t.mortonCode) & LOWER_3BIT_MASK;
		const unsigned int p_neighborIdx = p.neighbors[neighbor_LUTparent[key][tid_y]];
		if (p_neighborIdx != UINT32_MAX)
		{
			SVONode h = d_nodeArray[p_neighborIdx];
			t.neighbors[tid_y] = h.childs[neighbor_LUTchild[key][tid_y]];
		}
		else t.neighbors[tid_y] = UINT32_MAX;
	}
}

void SparseVoxelOctree::constructNodeNeighbors(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
	thrust::device_vector<SVONode>& d_SVONodeArray)
{
	dim3 gridSize, blockSize;
	blockSize.x = 32, blockSize.y = 32;
	gridSize.y = 1;
	// find neighbors(up to bottom)
	//assert(treeDepth >= 1 && treeDepth == d_allSVONodeArray.size());
	assert(treeDepth >= 1);
	(d_SVONodeArray.data() + d_SVONodeArray.size() - 1)->neighbors[13] = d_SVONodeArray.size() - 1;
	//(d_allSVONodeArray.data() + depth - 1)->data()->neighbors[13] = d_esumTreeNodesArray[depth - 1];
	for (int i = treeDepth - 2; i >= 0; --i)
	{
		const size_t nNodes = depthNumNodes[i];
		gridSize.x = (nNodes + blockSize.x - 1) / blockSize.x;
		/*findNeighbors << <gridSize, blockSize >> > (nNodes, (d_allMorton2Idx.data() + i + 1)->data().get(),
			(d_allSVONodeArray.data() + i)->data().get(), (d_allSVONodeArray.data() + i + 1)->data().get());*/
		findNeighbors << <gridSize, blockSize >> > (nNodes, d_esumTreeNodesArray[i], d_SVONodeArray.data().get());
	}
}

//template<typename T1, typename T2>
//struct cuPair
//{
//	T1 first;
//	T2 second;
//	CUDA_CALLABLE_MEMBER NodeVertexPair(const T1& _first, const T2& _second) :first(_first), second(_second) {}
//};
using thrust_edge = thrust::pair<Eigen::Vector3f, Eigen::Vector3f>;
thrust::device_vector<thrust::pair<thrust_edge, uint32_t>> d_nodeEdgeArray;
__constant__ short int d_vertSharedLUT[64] =
{
	0, 1, 3, 4, 9, 10, 12, 13,

	1, 2, 4, 5, 10, 11, 13 ,14,

	3, 4, 6, 7, 12, 13, 15, 16,

	4, 5, 7, 8, 13, 14, 16, 17,

	9, 10, 12, 13, 18, 19, 21, 22,

	10, 11, 13, 14, 19, 20, 22, 23,

	12, 13, 15, 16, 21, 22, 24, 25,

	13, 14, 16, 17, 22, 23, 25, 26
};
__global__ void determineNodeVertex(const size_t nNodes,
	const SVONode* d_nodeArray,
	thrust::pair<Eigen::Vector3f, uint32_t>* d_nodeVertArray)
{
	size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid_x < nNodes)
	{
		uint16_t x, y, z;
		Eigen::Vector3f origin = d_nodeArray[tid_x].origin;
		float width = d_nodeArray[tid_x].width;
#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			morton3D_32_decode(i, x, y, z);
			Eigen::Vector3f corner = width * Eigen::Vector3f((float)x, (float)y, (float)z);

			uint32_t morton = UINT_MAX, idx = tid_x;
			for (int j = 0; j < 8; ++j)
				if (d_nodeArray[tid_x].neighbors[d_vertSharedLUT[i * 8 + j]] < morton) idx = d_nodeArray[tid_x].neighbors[d_vertSharedLUT[i * 8 + j]];

			d_nodeVertArray[tid_x * 8 + i] = thrust::make_pair(corner, idx);
		}
	}
}

// edge: 02 23 31 10   46 67 75 54   04 26 37 15 
__constant__ short int d_edgeSharedLUT[48] =
{
	 3, 4, 12, 13,
	 4, 7, 13, 16,
	 4, 5, 13, 14,
	 1, 4, 10, 13,

	 12, 13, 21, 22,
	 13, 16, 22, 25,
	 13, 14, 22, 23,
	 10, 13, 19, 22,

	 9, 10, 12, 13,
	 12, 13, 15, 16,
	 13, 14, 16, 17,
	 10, 11, 13, 14
};
__global__ void determineNodeEdge(const size_t nNodes,
	const SVONode* d_nodeArray,
	thrust::pair<thrust_edge, uint32_t>* d_nodeEdgeArray)
{
	size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid_x < nNodes)
	{
		Eigen::Vector3f origin = d_nodeArray[tid_x].origin;
		float width = d_nodeArray[tid_x].width;

		thrust_edge edges[12] =
		{
			thrust::make_pair(origin, origin + Eigen::Vector3f(0, width, 0)),
			thrust::make_pair(origin + Eigen::Vector3f(0, width, 0), origin + Eigen::Vector3f(width, width, 0)),
			thrust::make_pair(origin + Eigen::Vector3f(width, width, 0), origin + Eigen::Vector3f(width, 0, 0)),
			thrust::make_pair(origin + Eigen::Vector3f(width, 0, 0), origin),

			thrust::make_pair(origin + Eigen::Vector3f(0, 0, width), origin + Eigen::Vector3f(0, width, width)),
			thrust::make_pair(origin + Eigen::Vector3f(0, width, width), origin + Eigen::Vector3f(width, width, width)),
			thrust::make_pair(origin + Eigen::Vector3f(width, width, width), origin + Eigen::Vector3f(width, 0, width)),
			thrust::make_pair(origin + Eigen::Vector3f(width, 0, width), origin + Eigen::Vector3f(0, 0, width)),

			thrust::make_pair(origin, origin + Eigen::Vector3f(0, 0, width)),
			thrust::make_pair(origin + Eigen::Vector3f(0, width, 0), origin + Eigen::Vector3f(0, width, width)),
			thrust::make_pair(origin + Eigen::Vector3f(width, width, 0), origin + Eigen::Vector3f(width, width, width)),
			thrust::make_pair(origin + Eigen::Vector3f(width, 0, 0), origin + Eigen::Vector3f(width, 0, width)),
		};

#pragma unroll
		for (int i = 0; i < 12; ++i)
		{
			thrust_edge edge = edges[i];

			uint32_t morton = UINT_MAX, idx = tid_x;
			for (int j = 0; j < 4; ++j)
				if (d_nodeArray[tid_x].neighbors[d_edgeSharedLUT[i * 4 + j]] < morton) idx = d_nodeArray[tid_x].neighbors[d_edgeSharedLUT[i * 4 + j]];

			d_nodeEdgeArray[tid_x * 12 + i] = thrust::make_pair(edge, idx);
		}
	}
}

template <typename T>
struct uniqueVert : public thrust::binary_function<T, T, T> {
	__host__ __device__ bool operator()(const T& a, const T& b) {
		return a.first == b.first;
	}
};

void SparseVoxelOctree::constructNodeVertexAndEdge(thrust::device_vector<SVONode>& d_SVONodeArray)
{
	cudaStream_t streams[2];
	for (int i = 0; i < 2; ++i)
		CUDA_CHECK(cudaStreamCreate(&streams[i]));

	thrust::device_vector < thrust::pair<Eigen::Vector3f, uint32_t>> d_nodeVertArray(numTreeNodes * 8);
	getOccupancyMaxPotentialBlockSize(numTreeNodes, minGridSize, blockSize, gridSize, determineNodeVertex, 0, 0);
	determineNodeVertex << <gridSize, blockSize, 0, streams[0] >> > (numTreeNodes, d_SVONodeArray.data().get(), d_nodeVertArray.data().get());

	auto newEnd = thrust::unique(d_nodeVertArray.begin(), d_nodeVertArray.end(), uniqueVert<thrust::pair<Eigen::Vector3f, uint32_t>>());
	const size_t newSize = newEnd - d_nodeVertArray.begin();
	resizeThrust(d_nodeVertArray, newSize);

	thrust::device_vector < thrust::pair<thrust_edge, uint32_t>> d_nodeEdgeArray(numTreeNodes * 12);
	getOccupancyMaxPotentialBlockSize(numTreeNodes, minGridSize, blockSize, gridSize, determineNodeEdge, 0, 0);
	determineNodeEdge << <gridSize, blockSize, 0, streams[1] >> > (numTreeNodes, d_SVONodeArray.data().get(), d_nodeEdgeArray.data().get());
	//for (int i = 0; i < depth; ++i)
	//{
	//	thrust::pair<Eigen::Vector3f, uint32_t>* d_nodeVertArray;
	//	//thrust::device_vector<thrust::pair<Eigen::Vector3f, uint32_t>> d_nodeVertArray((d_allSVONodeArray.data() + i)->size() * 8);
	//}

	for (int i = 0; i < 2; ++i)
		CUDA_CHECK(cudaStreamDestroy(streams[i]));
}

void SparseVoxelOctree::constructNodeAtrributes(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
	thrust::device_vector<SVONode>& d_SVONodeArray)
{
	constructNodeNeighbors(d_esumTreeNodesArray, d_SVONodeArray);

	constructNodeVertexAndEdge(d_SVONodeArray);
	//constructNodeEdgeArray();
}

void SparseVoxelOctree::writeTree(const std::string base_filename)
{
	std::string filename_output = base_filename + std::string("_") + std::to_string(treeDepth) + std::string("_tree.obj");
	std::ofstream output(filename_output.c_str(), std::ios::out);
	assert(output);

#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in obj voxels format to file %s \n", filename_output.c_str());
	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = numTreeNodes / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");
#endif

	size_t nFaces = 0;
	for (const auto& node : svoNodeArray)
	{
#ifndef SILENT			
		voxels_seen++;
		if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(3)) { fprintf(stdout, "75%%..."); }
#endif
		write_cube(node.origin, Eigen::Vector3f(node.width, node.width, node.width), output, nFaces);

	}
#ifndef SILENT
	fprintf(stdout, "100%% \n");
#endif

	output.close();
}