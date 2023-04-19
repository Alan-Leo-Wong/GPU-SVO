#include "SVO.h"
#include "CUDAUtil.h"
#include "MortonLUT.h"
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <crt/device_functions.h>
namespace cg = cooperative_groups;

#define MORTON_32_FLAG 0x80000000

template <typename T>
struct sumFlag : public thrust::binary_function<T, T, T> {
	__host__ __device__ T operator()(const T&, const T& b) {
		// printf("%lu %d\n", b, (b >> 31) & 1);
		return (b >> 31) & 1;
	}
};

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
		for (int z = t_bbox_grid.min.z(); z <= t_bbox_grid.max.z(); z++) {
			for (int y = t_bbox_grid.min.y(); y <= t_bbox_grid.max.y(); y++) {
				for (int x = t_bbox_grid.min.x(); x <= t_bbox_grid.max.x(); x++) {
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

					size_t mortonCode = mortonEncode_LUT(x, y, z);
					atomicExch(d_voxelArray + mortonCode, mortonCode | MORTON_32_FLAG); // 最高位设置为1，代表这是个表面的voxel
				}
			}
		}
		tid += stride;
	}
}

__global__ __inline__ void compactVoxel(const int nTris,
	const bool* d_isValidVoxel,
	uint32_t* d_voxelArray,
	size_t* d_esumVoxels,
	uint32_t* d_pactVoxelArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < nTris && d_isValidVoxel[tid])
		d_pactVoxelArray[d_esumVoxels[tid]] = d_voxelArray[tid];
}

__device__ __inline__ bool isSameParent(const uint32_t morton_1, const uint32_t morton_2)
{

}

// 计算表面voxel共对应多少个八叉树节点
__global__ __inline__ void cpNumNodes(const size_t nVoxels,
	const uint32_t* d_pactVoxelArray,
	size_t* d_nNodesArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= 1 && tid < nVoxels)
	{
		if (isSameParent(d_pactVoxelArray[tid], d_pactVoxelArray[tid - 1])) d_nNodesArray[tid] = 0;
		else d_nNodesArray[tid] = 8;
	}
}

// 根据d_sumNodesArray和d_pactVoxelArray(存储了莫顿码)设置节点数组，本质也是compact
// blockSize必须是32的整数倍，因为最底层节点个数是8的整数倍
__global__ __inline__ void voxelToNode(const size_t nNodes,
	const size_t nVoxels,
	const size_t* d_sumNodesArray,
	const size_t* d_pactVoxelArray,
	uint32_t* d_nodeArray)
{
	extern __shared__ uint32_t sh_nodeMorton[]; // blockSize / 8，数值为8的整数倍
	
	cg::thread_block ctb = cg::this_thread_block(); 
	cg::thread_group tile8 = cg::tiled_partition(ctb, 8);
	
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	/*sh_nodeMorton[threadIdx.x / 8] = 0;
	__syncthreads();*/

	if (tid < nNodes)
	{
		if (tid < nVoxels)
		{
			const size_t address = d_sumNodesArray[tid] + (d_pactVoxelArray[tid] % 8);
			d_nodeArray[address] = d_pactVoxelArray[tid];
			if((d_pactVoxelArray[tid] / 8) * 8 != 0) sh_nodeMorton[threadIdx.x / 8] = (d_pactVoxelArray[tid] / 8) * 8;
		}
		cg::sync(tile8);
		//__syncthreads();

		// 计算不在voxel里的节点的莫顿码
		if (d_nodeArray[tid] == 0)
		{
			d_nodeArray[tid] = tid % 8 + sh_nodeMorton[threadIdx.x / 8];
		}
	}
}

void SparseVoxelOctree::constructFineNodes()
{
	uint32_t* d_voxelArray;
	size_t voxelTabeSize = (size_t)((size_t)surfaceVoxelGridSize.x() * (size_t)surfaceVoxelGridSize.y() * (size_t)surfaceVoxelGridSize.z());
	//size_t voxelTabelize = surfaceVoxelGridSize.x() * surfaceVoxelGridSize.y() * surfaceVoxelGridSize.z() / 32.0f;
	CUDA_CHECK(cudaMalloc((void**)&d_voxelArray, sizeof(uint32_t) * voxelTabeSize));
	CUDA_CHECK(cudaMemset(d_voxelArray, 0, sizeof(uint32_t) * voxelTabeSize));

	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 
	getOccupancyMaxPotentialBlockSize(nModelTris, minGridSize, blockSize, gridSize, surfaceVoxelize, 0, 0);

	///	TODO: 调成同样大小(只要把模型的bbox设置为立方体就可以了，具体可参考cuda_voxelizer中的createMeshBBCube方法)
	Eigen::Vector3f unitVoxelSize{
		(modelBBox.max.x() - modelBBox.min.x()) / surfaceVoxelGridSize.x(),
		(modelBBox.max.y() - modelBBox.min.y()) / surfaceVoxelGridSize.y(),
		(modelBBox.max.z() - modelBBox.min.z()) / surfaceVoxelGridSize.z(),
	};
	float* d_triangle_data = meshToGPU_thrust(modelPoints);
	surfaceVoxelize << <gridSize, blockSize >> > (nModelTris, modelBBox, unitVoxelSize, d_triangleData, d_voxelArray);

	// compute number of surface voxels
	thrust::device_vector<bool> d_isValidVoxel;
	thrust::device_vector<size_t> d_esumVoxels; // exclusive scan
	uint32_t lastVoxelFlag = 0;
	CUDA_CHECK(cudaMemcpy(&lastVoxelFlag, d_voxelArray + voxelTabeSize - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if ((lastVoxelFlag >> 31) & 1) lastVoxelFlag = 1;
	thrust::exclusive_scan(d_voxelArray, d_voxelArray + voxelTabeSize, d_isValidVoxel.begin(), 0, sumFlag<uint32_t>());
	thrust::inclusive_scan(d_isValidVoxel.begin(), d_isValidVoxel.end(), d_esumVoxels.begin());
	size_t nVoxels = *(d_esumVoxels.end()) + lastVoxelFlag;
	if (!nVoxels)
	{
		printf("There is no valid voxels\n");
		return;
	}

	// compact surface voxels
	thrust::device_vector<uint32_t> d_pactVoxelArray(nVoxels);
	getOccupancyMaxPotentialBlockSize(nModelTris, minGridSize, blockSize, gridSize, compactVoxel, 0, 0);
	compactVoxel << <gridSize, blockSize >> > (nModelTris, thrust::raw_pointer_cast(d_isValidVoxel.data()),
		d_voxelArray, thrust::raw_pointer_cast(d_esumVoxels.data()), thrust::raw_pointer_cast(d_pactVoxelArray.data()));

	// get surface octree nodes by surface voxels
	thrust::device_vector<int> d_nNodesArray(nVoxels, 0); // 节点数量数组
	thrust::device_vector<size_t> d_sumNodesArray(nVoxels); // inlusive scan
	getOccupancyMaxPotentialBlockSize(nVoxels, minGridSize, blockSize, gridSize, cpNumNodes, 0, 0);
	cpNumNodes << <gridSize, blockSize >> > (nVoxels, thrust::raw_pointer_cast(d_pactVoxelArray.data()), thrust::raw_pointer_cast(d_nNodesArray.data()));
	thrust::inclusive_scan(d_nNodesArray.begin(), d_nNodesArray.end(), d_sumNodesArray.begin());
	size_t nNodes = *(d_sumNodesArray.end()) + 8; // 最底层的八叉树节点数量
	depthNumNodes.emplace_back(nNodes);

	// 节点数组
	thrust::device_vector<uint32_t> d_nodeArray;
	voxelToNode << <gridSize, blockSize >> > (nNodes, nVoxels, d_sumNodesArray, 
		thrust::raw_pointer_cast(d_pactVoxelArray.data()), thrust::raw_pointer_cast(d_nodeArray.data()));
}
