#include "SVO.h"
#include "CUDAUtil.h"
#include "MortonLUT.h"
#include "libmorton\morton.h"
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <crt/device_functions.h>
namespace cg = cooperative_groups;

template <typename T>
struct scanMortonFlag : public thrust::unary_function<T, T> {
	__host__ __device__ T operator()(const T& x) {
		// printf("%lu %d\n", b, (b >> 31) & 1);
		return (x >> 31) & 1;
	}
};

//template <typename T>
//struct reduceFlagMax : public thrust::binary_function<T, T, T> {
//	__host__ __device__ T operator()(const T& a, const T& b) {
//		return thrust::max(a, b);
//	}
//};

__global__ __inline__ void surfaceVoxelize(const int nTris,
	const Eigen::Vector3i surfaceVoxelGridSize,
	const AABox<Eigen::Vector3f> modelBBox,
	const Eigen::Vector3f unitVoxelSize,
	float* d_triangle_data,
	uint32_t* d_voxelArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	Eigen::Vector3f delta_p{ unitVoxelSize.x(), unitVoxelSize.y(), unitVoxelSize.z() };
	Eigen::Vector3i grid_max{ surfaceVoxelGridSize.x() - 1, surfaceVoxelGridSize.y() - 1, surfaceVoxelGridSize.z() - 1 }; // grid max (grid runs from 0 to gridsize-1)

	while (tid < nTris) { // every thread works on specific triangles in its stride
		size_t t = tid * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using modelBBox
		Eigen::Vector3f v0 = Eigen::Vector3f(d_triangle_data[t], d_triangle_data[t + 1], d_triangle_data[t + 2]) - modelBBox.min;
		Eigen::Vector3f v1 = Eigen::Vector3f(d_triangle_data[t + 3], d_triangle_data[t + 4], d_triangle_data[t + 5]) - modelBBox.min;
		Eigen::Vector3f v2 = Eigen::Vector3f(d_triangle_data[t + 6], d_triangle_data[t + 7], d_triangle_data[t + 8]) - modelBBox.min;
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
			Eigen::Vector3i(floor(t_bbox_world.min.x() / unitVoxelSize.x()), floor(t_bbox_world.min.y() / unitVoxelSize.y()), floor(t_bbox_world.min.z() / unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);
		t_bbox_grid.max = clamp(
			Eigen::Vector3i(ceil(t_bbox_world.max.x() / unitVoxelSize.x()), ceil(t_bbox_world.max.y() / unitVoxelSize.y()), ceil(t_bbox_world.max.z() / unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);

		// PREPARE PLANE TEST PROPERTIES
		Eigen::Vector3f c(0.0f, 0.0f, 0.0f);
		if (n.x() > 0.0f) { c.x() = unitVoxelSize.x(); }
		if (n.y() > 0.0f) { c.y() = unitVoxelSize.y(); }
		if (n.z() > 0.0f) { c.z() = unitVoxelSize.z(); }
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
		float d_xy_e0 = (-1.0f * n_xy_e0.dot(Eigen::Vector2f(v0.x(), v0.y()))) + fmaxf(0.0f, unitVoxelSize.x() * n_xy_e0[0]) + fmaxf(0.0f, unitVoxelSize.y() * n_xy_e0[1]);
		float d_xy_e1 = (-1.0f * n_xy_e1.dot(Eigen::Vector2f(v1.x(), v1.y()))) + fmaxf(0.0f, unitVoxelSize.x() * n_xy_e1[0]) + fmaxf(0.0f, unitVoxelSize.y() * n_xy_e1[1]);
		float d_xy_e2 = (-1.0f * n_xy_e2.dot(Eigen::Vector2f(v2.x(), v2.y()))) + fmaxf(0.0f, unitVoxelSize.x() * n_xy_e2[0]) + fmaxf(0.0f, unitVoxelSize.y() * n_xy_e2[1]);
		// YZ plane
		Eigen::Vector2f n_yz_e0(-1.0f * e0.z(), e0.y());
		Eigen::Vector2f n_yz_e1(-1.0f * e1.z(), e1.y());
		Eigen::Vector2f n_yz_e2(-1.0f * e2.z(), e2.y());
		if (n.x() < 0.0f) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		float d_yz_e0 = (-1.0f * n_yz_e0.dot(Eigen::Vector2f(v0.y(), v0.z()))) + fmaxf(0.0f, unitVoxelSize.y() * n_yz_e0[0]) + fmaxf(0.0f, unitVoxelSize.z() * n_yz_e0[1]);
		float d_yz_e1 = (-1.0f * n_yz_e1.dot(Eigen::Vector2f(v1.y(), v1.z()))) + fmaxf(0.0f, unitVoxelSize.y() * n_yz_e1[0]) + fmaxf(0.0f, unitVoxelSize.z() * n_yz_e1[1]);
		float d_yz_e2 = (-1.0f * n_yz_e2.dot(Eigen::Vector2f(v2.y(), v2.z()))) + fmaxf(0.0f, unitVoxelSize.y() * n_yz_e2[0]) + fmaxf(0.0f, unitVoxelSize.z() * n_yz_e2[1]);
		// ZX plane																							 													  
		Eigen::Vector2f n_zx_e0(-1.0f * e0.x(), e0.z());
		Eigen::Vector2f n_zx_e1(-1.0f * e1.x(), e1.z());
		Eigen::Vector2f n_zx_e2(-1.0f * e2.x(), e2.z());
		if (n.y() < 0.0f) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		float d_xz_e0 = (-1.0f * n_zx_e0.dot(Eigen::Vector2f(v0.z(), v0.x()))) + fmaxf(0.0f, unitVoxelSize.z() * n_zx_e0[0]) + fmaxf(0.0f, unitVoxelSize.x() * n_zx_e0[1]);
		float d_xz_e1 = (-1.0f * n_zx_e1.dot(Eigen::Vector2f(v1.z(), v1.x()))) + fmaxf(0.0f, unitVoxelSize.z() * n_zx_e1[0]) + fmaxf(0.0f, unitVoxelSize.x() * n_zx_e1[1]);
		float d_xz_e2 = (-1.0f * n_zx_e2.dot(Eigen::Vector2f(v2.z(), v2.x()))) + fmaxf(0.0f, unitVoxelSize.z() * n_zx_e2[0]) + fmaxf(0.0f, unitVoxelSize.x() * n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (uint16_t z = t_bbox_grid.min.z(); z <= t_bbox_grid.max.z(); z++) {
			for (uint16_t y = t_bbox_grid.min.y(); y <= t_bbox_grid.max.y(); y++) {
				for (uint16_t x = t_bbox_grid.min.x(); x <= t_bbox_grid.max.x(); x++) {
					// if (checkBit(voxel_table, location)){ continue; }
					// TRIANGLE PLANE THROUGH BOX TEST
					Eigen::Vector3f p(x * unitVoxelSize.x(), y * unitVoxelSize.y(), z * unitVoxelSize.z());
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

__global__ __inline__ void compactArray(const int n,
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
__global__ __inline__ void cpNumNodes(const size_t n,
	const uint32_t* d_pactDataArray,
	size_t* d_nNodesArray,
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
__global__ __inline__ void createNode(const size_t nNodes,
	const size_t pactSize,
	const size_t* d_sumNodesArray,
	const size_t* d_pactDataArray,
	const Eigen::Vector3f d_gridOrigin,
	const float d_width,
	SVONode* d_nodeArray,
	size_t* d_morton2Idx)
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
			// 得到mortonCode对应的实际存储节点的位置
			const size_t address = d_sumNodesArray[tid] + (d_pactDataArray[tid] & LOWER_3BIT_MASK);
			d_nodeArray[address].mortonCode = d_pactDataArray[tid];
			morton3D_32_decode(d_nodeArray[address].mortonCode, x, y, z);
			d_nodeArray[address].origin = d_gridOrigin + d_width * Eigen::Vector3f((float)x, (float)y, (float)z);
			d_nodeArray[address].width = d_width;
			d_morton2Idx[d_pactDataArray[tid]] = address; // 莫顿码到节点数组下标的映射

			// (d_pactDataArray[tid] / 8) * 8 得到d_pactDataArray[tid](莫顿码)对应的以8为整数倍的下标
			// 用于计算那些在这个if中没计算出来的节点莫顿码
			const uint32_t morton = d_pactDataArray[tid] & D_MORTON_32_FLAG; // 去除符号位的实际莫顿码
			if ((morton / 8) * 8 != 0) sh_nodeMorton[threadIdx.x / 8] = (morton / 8) * 8; // 八个节点为一组
		}
		cg::sync(tile8);
		//__syncthreads();

		// 计算不在voxel里的节点的莫顿码
		if (d_nodeArray[tid].mortonCode == 0)
		{
			const uint32_t morton = (tid & LOWER_3BIT_MASK) + sh_nodeMorton[threadIdx.x / 8];
			d_nodeArray[tid].mortonCode = morton | E_MORTON_32_FLAG;
			morton3D_32_decode(d_nodeArray[tid].mortonCode, x, y, z);
			d_nodeArray[tid].origin = d_gridOrigin + d_width * Eigen::Vector3f((float)x, (float)y, (float)z);
			d_nodeArray[tid].width = d_width;
			d_morton2Idx[morton] = tid;
		}
	}
}

bool SparseVoxelOctree::constructFineNodes(thrust::device_vector<uint32_t>& d_parentMortonArray,
	thrust::device_vector<thrust::device_vector<size_t>>& d_allMorton2Idx)
{
	// +1防止存储时越界，原因是莫顿码的构成形式
	//size_t voxelArraySize = (size_t)((size_t)(surfaceVoxelGridSize.x() + 1) * (size_t)(surfaceVoxelGridSize.y() + 1) * (size_t)(surfaceVoxelGridSize.z() + 1));
	// 不需要+1（莫顿码为0代表坐标位于原点的第一个八叉树节点，八个顶点坐标需要令算）
	size_t voxelArraySize = (size_t)((size_t)surfaceVoxelGridSize.x() * (size_t)surfaceVoxelGridSize.y() * (size_t)surfaceVoxelGridSize.z());
	thrust::device_vector<uint32_t> d_voxelArray(voxelArraySize, 0);
	d_parentMortonArray.resize(voxelArraySize);

	//size_t voxelTabelize = surfaceVoxelGridSize.x() * surfaceVoxelGridSize.y() * surfaceVoxelGridSize.z() / 32.0f;

	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 
	getOccupancyMaxPotentialBlockSize(nModelTris, minGridSize, blockSize, gridSize, surfaceVoxelize, 0, 0);

	///	TODO: 调成同样大小(只要把模型的bbox设置为立方体就可以了，具体可参考cuda_voxelizer中的createMeshBBCube方法)
	/*Eigen::Vector3f unitVoxelSize{
		(modelBBox.max.x() - modelBBox.min.x()) / surfaceVoxelGridSize.x(),
		(modelBBox.max.y() - modelBBox.min.y()) / surfaceVoxelGridSize.y(),
		(modelBBox.max.z() - modelBBox.min.z()) / surfaceVoxelGridSize.z(),
	};*/
	float unitVoxelSize = (modelBBox.max.x() - modelBBox.min.x()) / surfaceVoxelGridSize.x();

	float* d_triangle_data = meshToGPU_thrust(modelPoints);
	surfaceVoxelize << <gridSize, blockSize >> > (nModelTris, modelBBox, unitVoxelSize, d_triangleData, d_voxelArray.data().get());

	// compute number of surface voxels
	thrust::device_vector<bool> d_isValidVoxel(voxelArraySize); // 必须初始化分配大小
	thrust::device_vector<size_t> d_esumVoxelsArray(voxelArraySize); // exclusive scan
	/*uint32_t lastVoxelFlag = d_voxelArray[voxelArraySize - 1];
	CUDA_CHECK(cudaMemcpy(&lastVoxelFlag, d_voxelArray.data().get() + voxelArraySize - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if ((lastVoxelFlag >> 31) & 1) lastVoxelFlag = 1;*/
	thrust::transform(d_voxelArray.begin(), d_voxelArray.end(), d_isValidVoxel.begin(), 0, scanMortonFlag<uint32_t>());
	thrust::exclusive_scan(d_isValidVoxel.begin(), d_isValidVoxel.end(), d_esumVoxelsArray.begin());
	size_t nVoxels = *(d_esumVoxelsArray.end()) + d_isValidVoxel[voxelArraySize - 1];
	if (!nVoxels) { printf("There is no valid voxels\n"); return false; }

	// compact surface voxels
	thrust::device_vector<uint32_t> d_pactVoxelArray(nVoxels);
	getOccupancyMaxPotentialBlockSize(nModelTris, minGridSize, blockSize, gridSize, compactArray, 0, 0);
	compactArray << <gridSize, blockSize >> > (nModelTris, d_isValidVoxel.data().get(), d_voxelArray.data().get(), d_esumVoxelsArray.data().get(), d_pactVoxelArray.data().get());

	// get the number of octree nodes by surface voxels
	thrust::device_vector<int> d_nNodesArray(nVoxels, 0); // 节点数量数组
	getOccupancyMaxPotentialBlockSize(nVoxels, minGridSize, blockSize, gridSize, cpNumNodes, 0, 0);
	d_parentMortonArray[getParentMorton(d_pactVoxelArray[0])];
	cpNumNodes << <gridSize, blockSize >> > (nVoxels, d_pactVoxelArray.data().get(), d_nNodesArray.data().get(), d_parentMortonArray.data().get());
	thrust::device_vector<size_t> d_sumNodesArray(nVoxels); // inlusive scan
	thrust::inclusive_scan(d_nNodesArray.begin(), d_nNodesArray.end(), d_sumNodesArray.begin());
	size_t nNodes = *(d_sumNodesArray.end()) + 8; // 最底层的八叉树节点数量
	//size_t nNodes = thrust::reduce(d_nNodesArray.begin(), d_nNodesArray.end(),0, thrust::plus<int>()) + 8; // 最底层的八叉树节点数量
	//depthNumNodes.emplace_back(nNodes);

	// 设置节点数组，同时设置父节点的莫顿码数组
	thrust::device_vector<uint32_t> d_nodeArray;
	uint32_t maxMortonCode = (*d_pactVoxelArray.end()) & D_MORTON_32_FLAG; // 去除符号位的实际莫顿码
	// 存储morton码到nodeArray下标的映射，实际上最大的莫顿码为maxMortonCode + 8 - (maxMortonCode % 8)
	// 因为还有未发掘的节点莫顿码
	thrust::device_vector<size_t> d_morton2Idx(maxMortonCode + 8 - (maxMortonCode % 8));
	blockSize = 256;
	gridSize = (nNodes + blockSize - 1) / blockSize;
	createNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize >> > (nNodes, nVoxels, d_sumNodesArray.data().get(),
		d_pactVoxelArray.data().get(), d_nodeArray.data().get(), d_morton2Idx.data().get());
	std::vector<uint32_t> t_vec(d_nNodesArray.size());
	CUDA_CHECK(cudaMemcpy(t_vec.data(), d_nodeArray.data().get(), sizeof(uint32_t) * d_nodeArray.size(), cudaMemcpyDeviceToHost));
	tempNodeArray.emplace_back(t_vec);
	d_allMorton2Idx.push_back(d_morton2Idx);

	// 对表面节点父节点进行compact，去除末端不合法的莫顿码，&0x7fffffff 是为了得到最后一个合法莫顿码所在的位置（去除符号位）
	uint32_t nParentNodes = *thrust::max_element(d_parentMortonArray.begin(), d_parentMortonArray.end());
	//size_t nParentNodes = (thrust::reduce(d_parentMortonArray.begin(), d_parentMortonArray.end(), 0, reduceFlagMax<uint32_t>())) & 0x7fffffff;
	//nParentNodes = nParentNodes + 8 - (nParentNodes % 8);
	bool isValidMorton = (nParentNodes >> 31 ) & 1 /*& E_MORTON_32_FLAG*/; // 以防有效莫顿码为0
	nParentNodes = (nParentNodes & D_MORTON_32_FLAG) + isValidMorton;
	if (nParentNodes != 0) { d_parentMortonArray.resize(nParentNodes); return true; }
	else return false;
}

__global__ __inline__ void createNode(const size_t nNodes,
	const size_t pactSize,
	const size_t* d_sumNodesArray,
	const size_t* d_pactDataArray,
	const Eigen::Vector3f d_gridOrigin,
	const float d_width,
	SVONode* d_nodeArray,
	SVONode* d_childArray,
	size_t* d_morton2Idx)
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
			const size_t address = d_sumNodesArray[tid] + (d_pactDataArray[tid] & LOWER_3BIT_MASK);
			d_nodeArray[address].mortonCode = d_pactDataArray[tid];
			morton3D_32_decode(d_nodeArray[address].mortonCode, x, y, z);
			d_nodeArray[address].origin = d_gridOrigin + d_width * Eigen::Vector3f((float)x, (float)y, (float)z);
			d_nodeArray[address].width = d_width;
			d_morton2Idx[d_pactDataArray[tid]] = address; // 莫顿码到节点数组下标的映射
			for (int i = 0; i < 8; ++i)
			{
				d_nodeArray[address].childs[i] = address * 8 + i;
				d_childArray[address * 8 + i].parent = address;
			}

			// (d_pactDataArray[tid] / 8) * 8 得到d_pactDataArray[tid](莫顿码)对应的以8为整数倍的下标
			// 用于计算那些在这个if中没计算出来的节点莫顿码
			const uint32_t morton = d_pactDataArray[tid] & D_MORTON_32_FLAG; // 去除符号位的实际莫顿码
			if ((morton / 8) * 8 != 0) sh_nodeMorton[threadIdx.x / 8] = (morton / 8) * 8; // 八个节点为一组
		}
		cg::sync(tile8);
		//__syncthreads();

		// 计算不在voxel里的节点的莫顿码
		if (d_nodeArray[tid].mortonCode == 0)
		{
			const uint32_t morton = (tid & LOWER_3BIT_MASK) + sh_nodeMorton[threadIdx.x / 8];
			d_nodeArray[tid].mortonCode = morton | E_MORTON_32_FLAG;
			morton3D_32_decode(d_nodeArray[tid].mortonCode, x, y, z);
			d_nodeArray[tid].origin = d_gridOrigin + d_width * Eigen::Vector3f((float)x, (float)y, (float)z);
			d_nodeArray[tid].width = d_width;
			for (int i = 0; i < 8; ++i)
			{
				d_nodeArray[tid].childs[i] = tid * 8 + i;
				d_childArray[tid * 8 + i].parent = tid;
			}
			d_morton2Idx[morton] = tid;
		}
	}
}

void SparseVoxelOctree::createOctree()
{
	thrust::device_vector<uint32_t> d_parentMortonArray;
	thrust::device_vector<thrust::device_vector<size_t>> d_allMorton2Idx;
	int depth = 1;
	if (constructFineNodes(d_parentMortonArray, d_allMorton2Idx)) // 如果表面节点有父节点，则开始从父节点这一层自下向上构建
	{
		size_t nodeArraySize = d_parentMortonArray.size();

		thrust::device_vector<bool> d_isValidNodeArray;
		thrust::device_vector<size_t> d_esumNodesArray; // exclusive scan
		thrust::device_vector<uint32_t> d_pactNodeArray(nodeArraySize);
		thrust::device_vector<int> d_nNodesArray; // 节点数量记录数组
		thrust::device_vector<uint32_t> d_nodeArray; // 节点数组
		thrust::device_vector<size_t> d_sumNodesArray; // inlusive scan
		thrust::device_vector<size_t> d_morton2Idx; // 存储morton码到nodeArray下标的映射

		// Estimate best block and grid size using CUDA Occupancy Calculator
		int blockSize;   // The launch configurator returned block size 
		int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch 
		int gridSize;    // The actual grid size needed, based on input size 
		while (nodeArraySize != 0)
		{
			// 也可以不需要d_isValidNodeArray和inclusive_scan，直接将exclusive_scan的结果给d_esumNodeArray
			// 然后在global核函数中判断d_esumNodeArray[tid]对应的node是否是合法的(即最高位为1的)
			thrust::exclusive_scan(d_parentMortonArray.begin(), d_parentMortonArray.end(), d_isValidNodeArray.begin(), 0, scanMortonFlag<uint32_t>());
			thrust::inclusive_scan(d_isValidNodeArray.begin(), d_isValidNodeArray.end(), d_esumNodesArray.begin());
			// compact d_nodeParentArray
			getOccupancyMaxPotentialBlockSize(nodeArraySize, minGridSize, blockSize, gridSize, compactArray, 0, 0);
			compactArray << <gridSize, blockSize >> > (nodeArraySize, d_isValidNodeArray, d_esumNodesArray, d_pactNodeArray);
			size_t pactSize = *d_esumNodesArray.end() + 1;

			// 计算d_nodeParentArray这一层的节点数量
			getOccupancyMaxPotentialBlockSize(pactSize, minGridSize, blockSize, gridSize, cpNumNodes, 0, 0);
			d_nNodesArray.resize(pactSize, 0); // 节点数量数组
			cpNumNodes << <gridSize, blockSize >> > (pactSize, d_pactNodeArray, d_nNodesArray, d_parentMortonArray);
			size_t nNodes = thrust::reduce(d_nNodesArray.begin(), d_nNodesArray.end(), 0, thrust::plus<int>()) + 8; // 八叉树节点数量

			// 设置节点数组
			uint32_t maxMortonCode = *d_pactNodeArray.end();
			d_morton2Idx.resize(maxMortonCode + 8 - (maxMortonCode % 8));
			d_nodeArray.resize(nNodes);
			d_sumNodesArray.resize(nNodes);
			thrust::inclusive_scan(d_nNodesArray.begin(), d_nNodesArray.end(), d_sumNodesArray.begin());
			size_t nNodes = *(d_sumNodesArray.end()) + 8; // 最底层的八叉树节点数量
			createNode << <gridSize, blockSize >> > (nNodes, pactSize, d_sumNodesArray, d_pactNodeArray, d_nodeArray, d_morton2Idx);

			// 对表面节点父节点进行compact，去除末端不合法的莫顿码，&0x7fffffff 是为了得到最后一个合法莫顿码所在的位置（去除符号位）
			nodeArraySize = *thrust::max_element(d_parentMortonArray.begin(), d_parentMortonArray.end());
			//size_t nParentNodes = (thrust::reduce(d_parentMortonArray.begin(), d_parentMortonArray.end(), 0, reduceFlagMax<uint32_t>())) & 0x7fffffff;
			//nParentNodes = nParentNodes + 8 - (nParentNodes % 8);
			bool isValidMorton = (nodeArraySize >> 31) & 1/* & E_MORTON_32_FLAG*/; // 以防有效莫顿码为0
			nodeArraySize = (nodeArraySize & 0x7fffffff) + isValidMorton;

			depth++;
			nodeArraySize /= 8;
		}
	}
	else
	{

	}
}