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
	return (((morton >> 3) & 0xfffffff));
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
	const Eigen::Vector3i* d_surfaceVoxelGridSize,
	const Eigen::Vector3f* d_gridOrigin,
	const Eigen::Vector3f* d_unitVoxelSize,
	float* d_triangle_data,
	uint32_t* d_voxelArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;
	int t_tid = tid;

	const Eigen::Vector3i surfaceVoxelGridSize = *d_surfaceVoxelGridSize;
	const Eigen::Vector3f unitVoxelSize = *d_unitVoxelSize;
	const Eigen::Vector3f gridOrigin = *d_gridOrigin;
	Eigen::Vector3f delta_p{ unitVoxelSize.x(), unitVoxelSize.y(), unitVoxelSize.z() };
	Eigen::Vector3i grid_max{ surfaceVoxelGridSize.x() - 1, surfaceVoxelGridSize.y() - 1, surfaceVoxelGridSize.z() - 1 }; // grid max (grid runs from 0 to gridsize-1)
	while (tid < nTris) { // every thread works on specific triangles in its stride
		size_t t = tid * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using modelBBox
		Eigen::Vector3f v0 = Eigen::Vector3f(d_triangle_data[t], d_triangle_data[t + 1], d_triangle_data[t + 2]) - gridOrigin;
		Eigen::Vector3f v1 = Eigen::Vector3f(d_triangle_data[t + 3], d_triangle_data[t + 4], d_triangle_data[t + 5]) - gridOrigin;
		Eigen::Vector3f v2 = Eigen::Vector3f(d_triangle_data[t + 6], d_triangle_data[t + 7], d_triangle_data[t + 8]) - gridOrigin;
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
			Eigen::Vector3i((t_bbox_world.min.x() / unitVoxelSize.x()), (t_bbox_world.min.y() / unitVoxelSize.y()), (t_bbox_world.min.z() / unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);
		t_bbox_grid.max = clamp(
			Eigen::Vector3i((t_bbox_world.max.x() / unitVoxelSize.x()), (t_bbox_world.max.y() / unitVoxelSize.y()), (t_bbox_world.max.z() / unitVoxelSize.z())),
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
					/*if (mortonCode == 63)
					{
						printf("x = %d, y = %d, z = %d, OK!\n", (int)x, (int)y, (int)z);
					}*/
					atomicExch(d_voxelArray + mortonCode, mortonCode | E_MORTON_32_FLAG); // 最高位设置为1，代表这是个表面的voxel
				}
			}
		}
		tid += stride;
	}
}

void SparseVoxelOctree::meshVoxelize(const Eigen::Vector3i* d_surfaceVoxelGridSize,
	const Eigen::Vector3f* d_unitVoxelSize,
	const Eigen::Vector3f* d_gridOrigin,
	thrust::device_vector<uint32_t>& d_CNodeMortonArray)
{
	thrust::device_vector<Eigen::Vector3f> d_triangleThrustVec;
	for (int i = 0; i < nModelTris; ++i)
	{
		d_triangleThrustVec.push_back(modelPoints[idx2Points[i].x()]);
		d_triangleThrustVec.push_back(modelPoints[idx2Points[i].y()]);
		d_triangleThrustVec.push_back(modelPoints[idx2Points[i].z()]);
	}
	float* d_triangleData = (float*)thrust::raw_pointer_cast(&(d_triangleThrustVec[0]));
	getOccupancyMaxPotentialBlockSize(nModelTris, minGridSize, blockSize, gridSize, surfaceVoxelize, 0, 0);
	surfaceVoxelize << <gridSize, blockSize >> > (nModelTris, d_surfaceVoxelGridSize,
		d_gridOrigin, d_unitVoxelSize, d_triangleData, d_CNodeMortonArray.data().get());
	getLastCudaError("Kernel 'surfaceVoxelize' launch failed!\n");
	cudaDeviceSynchronize();
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

// 计算表面voxel共对应多少个八叉树节点同时设置父节点的莫顿码数组
__global__ void cpNumNodes(const size_t n,
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
			d_parentMortonArray[parentMorton] = parentMorton | E_MORTON_32_FLAG;
			d_nNodesArray[tid] = 8;
		}
	}
}

// 根据d_sumNodesArray和d_pactVoxelArray(存储了莫顿码)设置节点数组，本质也是compact
// blockSize必须是32的整数倍，因为最底层节点个数是8的整数倍
//__global__ void createNode(const size_t nNodes,
//	const size_t pactSize,
//	const size_t* d_sumNodesArray,
//	const uint32_t* d_pactDataArray,
//	const Eigen::Vector3f* d_gridOrigin,
//	const float* d_width,
//	SVONode* d_nodeArray/*,
//	size_t* d_morton2Idx*/)
//{
//	extern __shared__ uint32_t sh_nodeMorton[]; // blockSize
//
//	//cg::thread_block ctb = cg::this_thread_block();
//	//cg::thread_group tile8 = cg::tiled_partition(ctb, 8);
//
//	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//	const Eigen::Vector3f gridOrigin = *d_gridOrigin;
//	const float width = *d_width;
//	/*sh_nodeMorton[threadIdx.x / 8] = 0; // 默认是0
//	__syncthreads();*/
//
//	if (tid < nNodes)
//	{
//		uint16_t x, y, z;
//		if (tid < pactSize)
//		{
//			const int begSumNodes = d_sumNodesArray[blockIdx.x * blockDim.x];
//			/*if (threadIdx.x == 0) sh_nodeMorton[0] = d_sumNodesArray[blockIdx.x * blockDim.x];
//			__syncthreads();*/
//
//			const uint32_t key = d_pactDataArray[tid] & LOWER_3BIT_MASK;
//			const uint32_t morton = d_pactDataArray[tid] & D_MORTON_32_FLAG; // 去除符号位的实际莫顿码
//			// 得到mortonCode对应的实际存储节点的位置
//			const size_t address = d_sumNodesArray[tid] + key;
//
//			/*if (morton == 392)
//			{
//				printf("tid = %d, begSumNodes = %d, key = %d, d_pact = %d, morton = %d, d_sumNodesArray = %d, address = %d\n",
//					tid, (int)begSumNodes, (int)key, (int)(d_pactDataArray[tid] & D_MORTON_32_FLAG), (int)morton, (int)d_sumNodesArray[tid], (int)address);
//			}*/
//
//			d_nodeArray[address].mortonCode = morton;
//			morton3D_32_decode(morton, x, y, z);
//			d_nodeArray[address].origin = gridOrigin + width * Eigen::Vector3f((float)x, (float)y, (float)z);
//			d_nodeArray[address].width = width;
//
//			//d_morton2Idx[morton] = address; // 莫顿码到节点数组下标的映射
//
//			// (d_pactDataArray[tid] / 8) * 8 得到d_pactDataArray[tid](莫顿码)对应的以8为整数倍的下标
//			// 用于计算那些在这个if中没计算出来的节点莫顿码
//			//if ((morton / 8) * 8 != 0) sh_nodeMorton[threadIdx.x / 8] = (morton / 8) * 8; // 八个节点为一组
//			const int idx = (d_sumNodesArray[tid] - begSumNodes) / 8;
//			sh_nodeMorton[idx] = (morton / 8) * 8;
//
//			if (tid == 259)
//			{
//				printf("d_sumNodesArray[%d] = %d, begSumNodes = %d, morton = %d, sh_nodeMorton = \n",
//					tid, d_sumNodesArray[tid], begSumNodes, (int)morton, (int)(sh_nodeMorton[idx]));
//			}
//		}
//		//cg::sync(tile8);
//		__syncthreads();
//
//		/*if (tid == 159)
//		{
//			printf("sh_nodeMorton = %d\n", (int)sh_nodeMorton[32]);
//		}*/
//
//		// 计算不在voxel里的节点的莫顿码
//		if (d_nodeArray[tid].mortonCode == 0)
//		{
//			if (tid == 259)
//			{
//				printf("threadIdx.x = %d, sh_nodeMorton[%d] = %d\n", threadIdx.x, threadIdx.x / 8, (int)(sh_nodeMorton[threadIdx.x / 8]));
//			}
//
//			// threadIdx.x / 8不会越界，因为最坏情况也是连续7个CNode都是同一个父亲，也就是最多连续7个CNode的d_sumNodesArray是一样的
//			const uint32_t morton = (tid & LOWER_3BIT_MASK) + sh_nodeMorton[threadIdx.x / 8];
//			d_nodeArray[tid].mortonCode = morton;
//			morton3D_32_decode(morton, x, y, z);
//			d_nodeArray[tid].origin = gridOrigin + width * Eigen::Vector3f((float)x, (float)y, (float)z);
//			d_nodeArray[tid].width = width;
//
//			//d_morton2Idx[morton] = tid;
//		}
//	}
//}

//__global__ void createNode(const size_t nNodes,
//	const size_t pactSize,
//	const size_t d_preChildDepthTreeNodes, // 子节点层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
//	const size_t d_preDepthTreeNodes, // 当前层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
//	const size_t* d_sumNodesArray, // 这一层的节点数量inclusive scan数组
//	const uint32_t* d_pactDataArray,
//	const Eigen::Vector3f* d_gridOrigin,
//	const float* d_width,
//	SVONode* d_nodeArray,
//	SVONode* d_childArray/*,
//	size_t* d_morton2Idx*/)
//{
//	extern __shared__ uint32_t sh_nodeMorton[]; // blockSize / 8，数值为8的整数倍
//
//	/*cg::thread_block ctb = cg::this_thread_block();
//	cg::thread_group tile8 = cg::tiled_partition(ctb, 8);*/
//
//	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//	const Eigen::Vector3f gridOrigin = *d_gridOrigin;
//	const float width = *d_width;
//	/*sh_nodeMorton[threadIdx.x / 8] = 0; // 默认是0
//	__syncthreads();*/
//
//	if (tid < nNodes)
//	{
//		uint16_t x, y, z;
//		if (tid < pactSize)
//		{
//			const int begSumNodes = d_sumNodesArray[blockIdx.x * blockDim.x];
//			/*const size_t preChildDepthTreeNodes = *d_preChildDepthTreeNodes;
//			const size_t preDepthTreeNodes = *d_preDepthTreeNodes;*/
//
//			const uint32_t key = d_pactDataArray[tid] & LOWER_3BIT_MASK;
//			const uint32_t morton = d_pactDataArray[tid] & D_MORTON_32_FLAG;
//			const size_t address = d_sumNodesArray[tid] + key;
//
//			SVONode& tNode = d_nodeArray[address];
//			tNode.mortonCode = morton;
//			morton3D_32_decode(morton, x, y, z);
//			tNode.origin = gridOrigin + width * Eigen::Vector3f((float)x, (float)y, (float)z);
//			tNode.width = width;
//			tNode.isLeaf = false;
//
//			/*if (tid == 0) printf("d_preChildDepthTreeNodes = %llu, d_preDepthTreeNodes = %llu\n", d_preChildDepthTreeNodes, d_preDepthTreeNodes);
//			if (tid == 0) printf("width = %f\n", width);*/
//
//#pragma unroll
//			for (int i = 0; i < 8; ++i)
//			{
//				tNode.childs[i] = d_preChildDepthTreeNodes + tid * 8 + i;
//				d_childArray[tid * 8 + i].parent = d_preDepthTreeNodes + tid;
//				//if (tid == 0 && i == 0) printf("child = %d, parent = %d\n", d_nodeArray[address].childs[i], d_childArray[tid * 8 + i].parent);
//			}
//
//			//d_morton2Idx[morton] = address; // 莫顿码到节点数组下标的映射
//
//			// (d_pactDataArray[tid] / 8) * 8 得到d_pactDataArray[tid](莫顿码)对应的以8为整数倍的下标
//			// 用于计算那些在这个if中没计算出来的节点莫顿码
//			//if ((morton / 8) * 8 != 0) sh_nodeMorton[threadIdx.x / 8] = (morton / 8) * 8; // 八个节点为一组
//			const int idx = (d_sumNodesArray[tid] - begSumNodes) / 8;
//			sh_nodeMorton[idx] = (morton / 8) * 8;
//		}
//		//cg::sync(tile8);
//		__syncthreads();
//
//		// 计算不在voxel里的节点的莫顿码(此时的节点为叶子节点)
//		SVONode& tNode = d_nodeArray[tid];
//		if (tNode.mortonCode == 0)
//		{
//			const uint32_t morton = (tid & LOWER_3BIT_MASK) + sh_nodeMorton[threadIdx.x / 8];
//
//			tNode.mortonCode = morton;
//			morton3D_32_decode(morton, x, y, z);
//			tNode.origin = gridOrigin + width * Eigen::Vector3f((float)x, (float)y, (float)z);
//			tNode.width = width;
//
//			//d_morton2Idx[morton] = tid;
//		}
//	}
//}

__global__ void createNode_1(const size_t pactSize,
	const size_t* d_sumNodesArray,
	const uint32_t* d_pactDataArray,
	const Eigen::Vector3f* d_gridOrigin,
	const float* d_width,
	uint32_t* d_begMortonArray,
	SVONode* d_nodeArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < pactSize)
	{
		const Eigen::Vector3f gridOrigin = *d_gridOrigin;
		const float width = *d_width;

		const int sumNodes = d_sumNodesArray[tid];
		const uint32_t pactData = d_pactDataArray[tid];

		const uint32_t key = pactData & LOWER_3BIT_MASK;
		const uint32_t morton = pactData & D_MORTON_32_FLAG; // 去除符号位的实际莫顿码
		// 得到mortonCode对应的实际存储节点的位置
		const size_t address = sumNodes + key;

		SVONode& tNode = d_nodeArray[address];
		tNode.mortonCode = morton;
		morton3D_32_decode(morton, x, y, z);
		tNode.origin = gridOrigin + width * Eigen::Vector3f((float)x, (float)y, (float)z);
		tNode.width = width;

		d_begMortonArray[tid] = (morton / 8) * 8;
	}
}

__global__ void createNode_2(const size_t pactSize,
	const size_t d_preChildDepthTreeNodes, // 子节点层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
	const size_t d_preDepthTreeNodes, // 当前层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
	const size_t* d_sumNodesArray, // 这一层的节点数量inclusive scan数组
	const uint32_t* d_pactDataArray,
	const Eigen::Vector3f* d_gridOrigin,
	const float* d_width,
	uint32_t* d_begMortonArray,
	SVONode* d_nodeArray,
	SVONode* d_childArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < pactSize)
	{
		const Eigen::Vector3f gridOrigin = *d_gridOrigin;
		const float width = *d_width;

		const int sumNodes = d_sumNodesArray[tid];
		const uint32_t pactData = d_pactDataArray[tid];

		const uint32_t key = pactData & LOWER_3BIT_MASK;
		const uint32_t morton = pactData & D_MORTON_32_FLAG;
		const size_t address = sumNodes + key;

		SVONode& tNode = d_nodeArray[address];
		tNode.mortonCode = morton;
		morton3D_32_decode(morton, x, y, z);
		tNode.origin = gridOrigin + width * Eigen::Vector3f((float)x, (float)y, (float)z);
		tNode.width = width;
		tNode.isLeaf = false;

		//printf("tid = %llu, morton = %d\n", tid, (int)(morton));
		d_begMortonArray[tid] = (morton / 8) * 8;

#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			tNode.childs[i] = d_preChildDepthTreeNodes + tid * 8 + i;
			d_childArray[tid * 8 + i].parent = d_preDepthTreeNodes + sumNodes + key; // +key是因为后面还需要构建remain nodes
		}
	}
}

__global__ void createRemainNode(const size_t nNodes,
	const Eigen::Vector3f* d_gridOrigin,
	const float* d_width,
	const uint32_t* d_begMortonArray,
	SVONode* d_nodeArray)
{
	extern __shared__ uint32_t sh_begMortonArray[];
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < nNodes)
	{
		sh_begMortonArray[threadIdx.x / 8] = d_begMortonArray[tid / 8];
		/*printf("threadIdx.x = %d, tid = %llu, d_begMortonArray[%llu] = %d, sh_begMortonArray[%d] = %d\n",
			threadIdx.x, tid, tid / 8, (int)d_begMortonArray[tid / 8], threadIdx.x / 8, (int)sh_begMortonArray[threadIdx.x / 8]);*/
		__syncthreads();

		if (d_nodeArray[tid].mortonCode == 0)
		{
			const Eigen::Vector3f gridOrigin = *d_gridOrigin;
			const float width = *d_width;

			const uint32_t key = tid & LOWER_3BIT_MASK;
			const uint32_t morton = sh_begMortonArray[threadIdx.x / 8] + key;

			SVONode& tNode = d_nodeArray[tid];
			tNode.mortonCode = morton;
			//printf("tid = %llu, morton = %d\n", tid, (int)(d_nodeArray[tid].mortonCode));

			morton3D_32_decode(morton, x, y, z);
			tNode.origin = gridOrigin + width * Eigen::Vector3f((float)x, (float)y, (float)z);
			tNode.width = width;
		}
	}
}

void SparseVoxelOctree::createOctree()
{
	assert(surfaceVoxelGridSize.x() >= 1 && surfaceVoxelGridSize.y() >= 1 && surfaceVoxelGridSize.z() >= 1);
	size_t gridCNodeSize = (size_t)mortonEncode_LUT((uint16_t)(surfaceVoxelGridSize.x() - 1), (uint16_t)(surfaceVoxelGridSize.y() - 1), (uint16_t)(surfaceVoxelGridSize.z() - 1)) + 1;
	//size_t gridCNodeSize = (size_t)((size_t)surfaceVoxelGridSize.x() * (size_t)surfaceVoxelGridSize.y() * (size_t)surfaceVoxelGridSize.z());
	size_t gridTreeNodeSize = gridCNodeSize % 8 ? gridCNodeSize + 8 - (gridCNodeSize % 8) : gridCNodeSize;
	///	TODO: 调成同样大小(只要把模型的bbox设置为立方体就可以了，具体可参考cuda_voxelizer中的createMeshBBCube方法)
	Eigen::Vector3f unitVoxelSize = Eigen::Vector3f(modelBBox.width.x() / surfaceVoxelGridSize.x(),
		modelBBox.width.y() / surfaceVoxelGridSize.y(),
		modelBBox.width.z() / surfaceVoxelGridSize.z());
	float unitNodeWidth = unitVoxelSize.x();

	Eigen::Vector3i* d_surfaceVoxelGridSize;
	CUDA_CHECK(cudaMalloc((void**)&d_surfaceVoxelGridSize, sizeof(Eigen::Vector3i)));
	CUDA_CHECK(cudaMemcpy(d_surfaceVoxelGridSize, &surfaceVoxelGridSize, sizeof(Eigen::Vector3i), cudaMemcpyHostToDevice));
	Eigen::Vector3f* d_gridOrigin;
	CUDA_CHECK(cudaMalloc((void**)&d_gridOrigin, sizeof(Eigen::Vector3f)));
	CUDA_CHECK(cudaMemcpy(d_gridOrigin, &modelBBox.min, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
	Eigen::Vector3f* d_unitVoxelSize;
	CUDA_CHECK(cudaMalloc((void**)&d_unitVoxelSize, sizeof(Eigen::Vector3f)));
	CUDA_CHECK(cudaMemcpy(d_unitVoxelSize, &unitVoxelSize, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
	float* d_unitNodeWidth;
	CUDA_CHECK(cudaMalloc((void**)&d_unitNodeWidth, sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_unitNodeWidth, &unitNodeWidth, sizeof(float), cudaMemcpyHostToDevice));

	/*CUDA_CHECK(cudaMemcpyToSymbol(d_surfaceVoxelGridSize, &surfaceVoxelGridSize, sizeof(Eigen::Vector3i)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_gridOrigin, &modelBBox.min, sizeof(Eigen::Vector3f)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_unitVoxelSize, &unitVoxelSize, sizeof(Eigen::Vector3f)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_unitNodeWidth, &unitNodeWidth, sizeof(float)));*/

	thrust::device_vector<uint32_t> d_CNodeMortonArray(gridCNodeSize, 0);
	thrust::device_vector<bool> d_isValidCNodeArray;
	thrust::device_vector<size_t> d_esumCNodesArray; // exclusive scan
	thrust::device_vector<uint32_t> d_pactCNodeArray;
	thrust::device_vector<size_t> d_numTreeNodesArray; // 节点数量记录数组
	thrust::device_vector<size_t> d_sumTreeNodesArray; // inlusive scan
	thrust::device_vector<size_t> d_esumTreeNodesArray; // 存储每一层节点数量的exclusive scan数组
	thrust::device_vector<uint32_t> d_begMortonArray;
	thrust::device_vector<SVONode> d_nodeArray; // 存储某一层的节点数组
	thrust::device_vector<SVONode> d_SVONodeArray; // save all sparse octree nodes

	// mesh voxelize
	//size_t voxelArraySize = (size_t)((size_t)(surfaceVoxelGridSize.x() + 1) * (size_t)(surfaceVoxelGridSize.y() + 1) * (size_t)(surfaceVoxelGridSize.z() + 1));
	// 不需要+1（莫顿码为0代表坐标位于原点的第一个八叉树节点，八个顶点坐标需要令算）
	resizeThrust(d_CNodeMortonArray, gridCNodeSize, (uint32_t)0);
	meshVoxelize(d_surfaceVoxelGridSize, d_unitVoxelSize, d_gridOrigin, d_CNodeMortonArray);

	//#ifndef NDEBUG
	//	vector<uint32_t> h_CNodeMortonArray(gridCNodeSize, 0);
	//	int cnodes = 0;
	//	CUDA_CHECK(cudaMemcpy(h_CNodeMortonArray.data(), d_CNodeMortonArray.data().get(), sizeof(uint32_t) * gridCNodeSize, cudaMemcpyDeviceToHost));
	//	for (int i = 0; i < h_CNodeMortonArray.size(); ++i)
	//		if (h_CNodeMortonArray[i] != 0) { ++cnodes; std::cout << (h_CNodeMortonArray[i] & D_MORTON_32_FLAG) << std::endl; }
	//	std::cout << "cnodes = " << cnodes << "\n--------\n";
	//#endif // !NDEBUG

		// create octree
		// 最后出来的树会比原始模型大7个格子, TODO: 到最顶层的时候只建立与模型bb相同的一个格子，它的周围7个格子不要建出来
	while (true)
	{
		// compute the number of 'coarse nodes'(eg: voxels)
		//size_t pactCNodeArraySize = 0;
		resizeThrust(d_isValidCNodeArray, gridCNodeSize);
		resizeThrust(d_esumCNodesArray, gridCNodeSize);
		thrust::transform(d_CNodeMortonArray.begin(), d_CNodeMortonArray.end(), d_isValidCNodeArray.begin(), scanMortonFlag<uint32_t>());
		thrust::exclusive_scan(d_isValidCNodeArray.begin(), d_isValidCNodeArray.end(), d_esumCNodesArray.begin(), 0); // 必须加init
		/*for (int i = 0; i < d_esumCNodesArray.size(); ++i)
			if (d_esumCNodesArray[i] != 0) std::cout << d_esumCNodesArray[i] << std::endl;*/
		size_t numCNodes = *(d_esumCNodesArray.rbegin()) + *(d_isValidCNodeArray.rbegin());
		//std::cout << "numCNodes = " << numCNodes << "\n--------\n";
		if (!numCNodes) { printf("Sparse Voxel Octree depth: %d\n", treeDepth); break; }

		treeDepth++;

		// compact coarse node array
		d_pactCNodeArray.clear(); resizeThrust(d_pactCNodeArray, numCNodes);
		getOccupancyMaxPotentialBlockSize(gridCNodeSize, minGridSize, blockSize, gridSize, compactArray, 0, 0);
		compactArray << <gridSize, blockSize >> > (gridCNodeSize, d_isValidCNodeArray.data().get(),
			d_CNodeMortonArray.data().get(), d_esumCNodesArray.data().get(), d_pactCNodeArray.data().get());
		getLastCudaError("Kernel 'compactArray' launch failed!\n");
		vector<uint32_t> h_pactCNodeArray(numCNodes, 0);
		CUDA_CHECK(cudaMemcpy(h_pactCNodeArray.data(), d_pactCNodeArray.data().get(), sizeof(uint32_t) * numCNodes, cudaMemcpyDeviceToHost));
		/*for (int i = 0; i < h_pactCNodeArray.size(); ++i)
			std::cout << (h_pactCNodeArray[i] & D_MORTON_32_FLAG) << std::endl;
		std::cout << "--------\n";*/

		// 验证体素
#ifdef NDEBUG
		if (treeDepth == 1)
		{
			vector<uint32_t> voxelArray;
			voxelArray.resize(numCNodes);
			CUDA_CHECK(cudaMemcpy(voxelArray.data(), d_pactCNodeArray.data().get(), sizeof(uint32_t) * numCNodes, cudaMemcpyDeviceToHost));
			writeVoxel(voxelArray, "bunny", unitNodeWidth);
		}
#endif // !NDEBUG

		// compute the number of (real)octree nodes by coarse node array
		// and set parent's morton code to 'd_CNodeMortonArray'
		size_t numNodes = 1;
		if (numCNodes > 1)
		{
			resizeThrust(d_numTreeNodesArray, numCNodes, (size_t)0);
			d_CNodeMortonArray.clear(); resizeThrust(d_CNodeMortonArray, gridTreeNodeSize, (uint32_t)0); // 此时用于记录父节点层的coarse node
			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, cpNumNodes, 0, 0);
			const uint32_t firstMortonCode = getParentMorton(d_pactCNodeArray[0]);
			d_CNodeMortonArray[firstMortonCode] = firstMortonCode | E_MORTON_32_FLAG;
			cpNumNodes << <gridSize, blockSize >> > (numCNodes, d_pactCNodeArray.data().get(), d_numTreeNodesArray.data().get(), d_CNodeMortonArray.data().get());
			getLastCudaError("Kernel 'cpNumNodes' launch failed!\n");
			resizeThrust(d_sumTreeNodesArray, numCNodes, (size_t)0); // inlusive scan
			thrust::inclusive_scan(d_numTreeNodesArray.begin(), d_numTreeNodesArray.end(), d_sumTreeNodesArray.begin());

			/*vector<uint32_t> h_CNodeMortonArray(numCNodes, 0);
			CUDA_CHECK(cudaMemcpy(h_CNodeMortonArray.data(), d_CNodeMortonArray.data().get(), sizeof(uint32_t) * numCNodes, cudaMemcpyDeviceToHost));
			for (int i = 0; i < h_CNodeMortonArray.size(); ++i)
				if (h_CNodeMortonArray[i] != 0) std::cout << "parent: " << (h_CNodeMortonArray[i] & D_MORTON_32_FLAG) << std::endl;
			std::cout << "--------\n";*/

			/*std::cout << "current coarse node morton:\n";
			vector<short> h_numTreeNodesArray(numCNodes, 0);
			vector<size_t> h_sumTreeNodesArray(numCNodes, 0);
			CUDA_CHECK(cudaMemcpy(h_numTreeNodesArray.data(), d_numTreeNodesArray.data().get(), sizeof(short) * numCNodes, cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(h_sumTreeNodesArray.data(), d_sumTreeNodesArray.data().get(), sizeof(size_t) * numCNodes, cudaMemcpyDeviceToHost));
			for (int i = 0; i < h_numTreeNodesArray.size(); ++i)
				if (i == 0)
					std::cout << (h_pactCNodeArray[i] & D_MORTON_32_FLAG) << ", " << h_numTreeNodesArray[i] << ", " << h_sumTreeNodesArray[i] << std::endl;
				else
					std::cout << (h_pactCNodeArray[i] & D_MORTON_32_FLAG) << ", " << (h_pactCNodeArray[i - 1] & D_MORTON_32_FLAG)
					<< ", " << h_numTreeNodesArray[i] << ", " << h_sumTreeNodesArray[i] << std::endl;
			std::cout << "--------\n";*/

			numNodes = *(d_sumTreeNodesArray.rbegin()) + 8;
			depthNumNodes.emplace_back(numNodes);
		}

		// set octree node array
		d_nodeArray.clear(); resizeThrust(d_nodeArray, numNodes, SVONode());
		uint32_t maxMortonCode = (*d_pactCNodeArray.rbegin()) & D_MORTON_32_FLAG;
		//resizeThrust(d_morton2Idx, gridTreeNodeSize);
		d_begMortonArray.clear(); resizeThrust(d_begMortonArray, numCNodes);
		if (treeDepth < 2)
		{
			//createNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize >> > (numNodes, numCNodes, d_sumTreeNodesArray.data().get(),
			//	d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_nodeArray.data().get()/*, d_morton2Idx.data().get()*/);

			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, createNode_1);
			createNode_1 << <gridSize, blockSize >> > (numCNodes, d_sumTreeNodesArray.data().get(),
				d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_begMortonArray.data().get(), d_nodeArray.data().get());
			getLastCudaError("Kernel 'createNode_1' launch failed!\n");
			printf("444\n");

			/*vector<SVONode> h_nodeArray(numNodes);
			CUDA_CHECK(cudaMemcpy(h_nodeArray.data(), d_nodeArray.data().get(), sizeof(SVONode) * numNodes, cudaMemcpyDeviceToHost));
			for (int i = 0; i < h_nodeArray.size(); ++i)
				std::cout << "node morton: " << (h_nodeArray[i].mortonCode) << std::endl;
			std::cout << "--------\n";*/

			d_esumTreeNodesArray.push_back(0);
		}
		else
		{
			//createNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize >> > (numNodes, numCNodes,
			//	*(d_esumTreeNodesArray.rbegin() + 1), *(d_esumTreeNodesArray.rbegin()),
			//	d_sumTreeNodesArray.data().get(), d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_nodeArray.data().get(),
			//	(d_SVONodeArray.data() + (*(d_esumTreeNodesArray.rbegin() + 1))).get()/*, d_morton2Idx.data().get()*/);

			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, createNode_2);
			createNode_2 << <gridSize, blockSize >> > (numCNodes, *(d_esumTreeNodesArray.rbegin() + 1), *(d_esumTreeNodesArray.rbegin()),
				d_sumTreeNodesArray.data().get(), d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_begMortonArray.data().get(),
				d_nodeArray.data().get(), (d_SVONodeArray.data() + (*(d_esumTreeNodesArray.rbegin() + 1))).get());
			getLastCudaError("Kernel 'createNode_2' launch failed!\n");

			/*vector<SVONode> h_nodeArray(numNodes);
			CUDA_CHECK(cudaMemcpy(h_nodeArray.data(), d_nodeArray.data().get(), sizeof(SVONode) * numNodes, cudaMemcpyDeviceToHost));
			for (int i = 0; i < h_nodeArray.size(); ++i)
				std::cout << "node morton: " << (h_nodeArray[i].mortonCode) << std::endl;
			std::cout << "--------\n";*/
		}
		auto newEndOfBegMorton = thrust::unique(d_begMortonArray.begin(), d_begMortonArray.end());
		resizeThrust(d_begMortonArray, newEndOfBegMorton - d_begMortonArray.begin());

		/*for (int i = 0; i < d_begMortonArray.size(); ++i)
			std::cout << "begMorton: " << d_begMortonArray[i] << std::endl;
		std::cout << "--------\n";*/

		blockSize = 256; gridSize = (numNodes + blockSize - 1) / blockSize;
		createRemainNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize / 8 >> > (numNodes, d_gridOrigin, d_unitNodeWidth,
			d_begMortonArray.data().get(), d_nodeArray.data().get());
		getLastCudaError("Kernel 'createRemainNode' launch failed!\n");

		/*vector<SVONode> h_nodeArray(numNodes);
		CUDA_CHECK(cudaMemcpy(h_nodeArray.data(), d_nodeArray.data().get(), sizeof(SVONode) * numNodes, cudaMemcpyDeviceToHost));
		for (int i = 0; i < h_nodeArray.size(); ++i)
			std::cout << "all node morton: " << (h_nodeArray[i].mortonCode) << std::endl;
		std::cout << "--------\n";*/

		d_SVONodeArray.insert(d_SVONodeArray.end(), d_nodeArray.begin(), d_nodeArray.end());
		//d_allSVONodeArray.push_back(d_nodeArray);
		//d_allMorton2Idx.push_back(d_morton2Idx);

		/*if (treeDepth >= 2)
		{
			vector<SVONode> h_SVONodeArray(d_SVONodeArray.size());
			CUDA_CHECK(cudaMemcpy(h_SVONodeArray.data(), d_SVONodeArray.data().get(), sizeof(SVONode) * d_SVONodeArray.size(), cudaMemcpyDeviceToHost));
			for (int i = 0; i < h_SVONodeArray.size(); ++i)
			{
				std::cout << (h_SVONodeArray[i].parent) << " -- ";
				for (int j = 0; j < 8; ++j)
					std::cout << (h_SVONodeArray[i].childs[j]) << ", ";
				std::cout << '\n';
			}
			std::cout << "--------\n";
		}*/

		d_esumTreeNodesArray.push_back(numNodes + (*d_esumTreeNodesArray.rbegin()));

		/*for (int i = 0; i < d_esumTreeNodesArray.size(); ++i)
			std::cout << d_esumTreeNodesArray[i] << std::endl;
		std::cout << "--------\n";*/

		//// special condition
		//if (treeDepth == 1 && gridCNodeSize == 1) { printf("Sparse Voxel Octree depth: %d\n", treeDepth); break; }
		// resize parent array 'd_CNodeMortonArray' to nexe loop
		uint32_t numParentCNodes = *thrust::max_element(d_CNodeMortonArray.begin(), d_CNodeMortonArray.end());
		bool isValidMorton = (numParentCNodes >> 31) & 1;
		// '+ isValidMorton' to prevent '(numParentNodes & D_MORTON_32_FLAG) = 0'同时正好可以让最后的大小能存储到最大的莫顿码
		numParentCNodes = (numParentCNodes & D_MORTON_32_FLAG) + isValidMorton;
		if (numParentCNodes != 0)
		{
			resizeThrust(d_CNodeMortonArray, numParentCNodes);
			unitNodeWidth *= 2.0; CUDA_CHECK(cudaMemcpy(d_unitNodeWidth, &unitNodeWidth, sizeof(float), cudaMemcpyHostToDevice));
			gridCNodeSize = numParentCNodes; gridTreeNodeSize = gridCNodeSize % 8 ? gridCNodeSize + 8 - (gridCNodeSize % 8) : gridCNodeSize;
			if (numNodes / 8 == 0) { printf("Sparse Voxel Octree depth: %d\n", treeDepth); break; }
		}
		else { printf("Sparse Voxel Octree depth: %d\n", treeDepth); break; }
	}
	numTreeNodes = d_esumTreeNodesArray[treeDepth];
	//TODO: copy to host
	svoNodeArray.resize(numTreeNodes);
	auto freeResOfCreateTree = [&]()
	{
		cleanupThrust(d_CNodeMortonArray);
		cleanupThrust(d_isValidCNodeArray);
		cleanupThrust(d_esumCNodesArray);
		cleanupThrust(d_pactCNodeArray);
		cleanupThrust(d_numTreeNodesArray);
		cleanupThrust(d_sumTreeNodesArray);
		cleanupThrust(d_nodeArray);

		CUDA_CHECK(cudaFree(d_surfaceVoxelGridSize));
		CUDA_CHECK(cudaFree(d_gridOrigin));
		CUDA_CHECK(cudaFree(d_unitNodeWidth));
		CUDA_CHECK(cudaFree(d_unitVoxelSize));
	};
	freeResOfCreateTree();

	constructNodeAtrributes(d_esumTreeNodesArray, d_SVONodeArray);
	CUDA_CHECK(cudaMemcpy(svoNodeArray.data(), d_SVONodeArray.data().get(), sizeof(SVONode) * numTreeNodes, cudaMemcpyDeviceToHost));
	cleanupThrust(d_numTreeNodesArray);
	cleanupThrust(d_SVONodeArray);
}

__device__ size_t d_topNodeIdx;
template<bool topFlag>
__global__ void findNeighbors(const size_t nNodes,
	const size_t preESumTreeNodes,
	SVONode* d_nodeArray)
{
	if (topFlag)
	{
		d_nodeArray[0].neighbors[13] = d_topNodeIdx;
	}
	else
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

}

void SparseVoxelOctree::constructNodeNeighbors(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
	thrust::device_vector<SVONode>& d_SVONodeArray)
{
	dim3 gridSize, blockSize;
	blockSize.x = 32, blockSize.y = 32;
	gridSize.y = 1;
	// find neighbors(up to bottom)
	if (treeDepth >= 2)
	{
		const size_t idx = d_SVONodeArray.size() - 1;
		CUDA_CHECK(cudaMemcpyToSymbol(d_topNodeIdx, &idx, sizeof(size_t)));
		findNeighbors<true> << <1, 1 >> > (1, 0, (d_SVONodeArray.data() + idx).get());
		for (int i = treeDepth - 2; i >= 0; --i)
		{
			const size_t nNodes = depthNumNodes[i];
			gridSize.x = (nNodes + blockSize.x - 1) / blockSize.x;
			findNeighbors<false> << <gridSize, blockSize >> > (nNodes, d_esumTreeNodesArray[i], d_SVONodeArray.data().get());
		}
	}
}

//template<typename T1, typename T2>
//struct cuPair
//{
//	T1 first;
//	T2 second;
//	CUDA_CALLABLE_MEMBER NodeVertexPair(const T1& _first, const T2& _second) :first(_first), second(_second) {}
//};
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

template <typename T>
struct uniqueEdge : public thrust::binary_function<T, T, T> {
	__host__ __device__
		bool operator()(const T& a, const T& b) {
		return ((a.first.first == b.first.first) && (a.first.second == b.first.second))||
				((a.first.first == b.first.second) && (a.first.second == b.first.first));
	}
};

void SparseVoxelOctree::constructNodeVertexAndEdge(thrust::device_vector<SVONode>& d_SVONodeArray)
{
	cudaStream_t streams[2];
	for (int i = 0; i < 2; ++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));

	thrust::device_vector < thrust::pair<Eigen::Vector3f, uint32_t>> d_nodeVertArray(numTreeNodes * 8);
	getOccupancyMaxPotentialBlockSize(numTreeNodes, minGridSize, blockSize, gridSize, determineNodeVertex, 0, 0);
	determineNodeVertex << <gridSize, blockSize, 0, streams[0] >> > (numTreeNodes, d_SVONodeArray.data().get(), d_nodeVertArray.data().get());

	thrust::device_vector < thrust::pair<thrust_edge, uint32_t>> d_nodeEdgeArray(numTreeNodes * 12);
	getOccupancyMaxPotentialBlockSize(numTreeNodes, minGridSize, blockSize, gridSize, determineNodeEdge, 0, 0);
	determineNodeEdge << <gridSize, blockSize, 0, streams[1] >> > (numTreeNodes, d_SVONodeArray.data().get(), d_nodeEdgeArray.data().get());

	cudaStreamSynchronize(streams[0]);
	auto vertNewEnd = thrust::unique(d_nodeVertArray.begin(), d_nodeVertArray.end(), uniqueVert<thrust::pair<Eigen::Vector3f, uint32_t>>());
	const size_t numVerts = vertNewEnd - d_nodeVertArray.begin();
	resizeThrust(d_nodeVertArray, numVerts);
	nodeVertexArray.resize(numVerts);
	CUDA_CHECK(cudaMemcpy(nodeVertexArray.data(), d_nodeVertArray.data().get(),
		sizeof(thrust::pair<Eigen::Vector3f, uint32_t>) * numVerts, cudaMemcpyDeviceToHost));

	cudaStreamSynchronize(streams[1]);
	auto edgeNewEnd = thrust::unique(d_nodeEdgeArray.begin(), d_nodeEdgeArray.end(), uniqueEdge<thrust::pair<thrust_edge, uint32_t>>()); // error
	const size_t numEdges = edgeNewEnd - d_nodeEdgeArray.begin();
	resizeThrust(d_nodeEdgeArray, numEdges);
	nodeEdgeArray.resize(numEdges);
	CUDA_CHECK(cudaMemcpy(nodeEdgeArray.data(), d_nodeEdgeArray.data().get(),
		sizeof(thrust::pair<thrust_edge, uint32_t>) * numEdges, cudaMemcpyDeviceToHost));

	for (int i = 0; i < 2; ++i) CUDA_CHECK(cudaStreamDestroy(streams[i]));
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
	fprintf(stdout, "[I/O] Writing octree data in obj format to file %s \n", filename_output.c_str());
	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = numTreeNodes / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");
#endif

	size_t faceBegIdx = 0;
	for (const auto& node : svoNodeArray)
	{
#ifndef SILENT			
		voxels_seen++;
		if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(3)) { fprintf(stdout, "75%%..."); }
#endif
		/*if (faceBegIdx == 440)
		{
			std::cout << node.origin.transpose() << std::endl;
		}*/
		//std::cout << "node.origin = " << node.origin.transpose() << ", node.width = " << node.width << ", node.morton = " << node.mortonCode << '\n';

		write_cube(node.origin, Eigen::Vector3f(node.width, node.width, node.width), output, faceBegIdx);
	}
#ifndef SILENT
	fprintf(stdout, "100%% \n");
#endif

	output.close();
}

void SparseVoxelOctree::writeVoxel(const vector<uint32_t>& voxelArray, const std::string& base_filename, const float& width)
{
	std::string filename_output = base_filename + std::string("_") + std::to_string(treeDepth) + std::string("_voxel.obj");
	std::ofstream output(filename_output.c_str(), std::ios::out);
	assert(output);

#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in obj voxels format to file %s \n", filename_output.c_str());
	// Write stats
	/*size_t voxels_seen = 0;
	const size_t write_stats_25 = numTreeNodes / 4.0f;*/
	//fprintf(stdout, "[I/O] Writing to file: 0%%...");
#endif

	size_t faceBegIdx = 0;
	for (size_t i = 0; i < voxelArray.size(); ++i)
	{
		//#ifndef SILENT			
		//		voxels_seen++;
		//		if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%..."); }
		//		else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%..."); }
		//		else if (voxels_seen == write_stats_25 * size_t(3)) { fprintf(stdout, "75%%..."); }
		//#endif

		const auto& morton = voxelArray[i];
		/*if (i == voxelArray.size() - 1)
		{
			std::cout << "last morton = " << morton << std::endl;
		}*/
		uint16_t x, y, z;
		morton3D_32_decode((morton & D_MORTON_32_FLAG), x, y, z);
		const Eigen::Vector3f nodeOrigin = modelBBox.min + width * Eigen::Vector3f((float)x, (float)y, (float)z);
		write_cube(nodeOrigin, Eigen::Vector3f(width, width, width), output, faceBegIdx);
	}
#ifndef SILENT
	//fprintf(stdout, "100%% \n");
#endif

	output.close();
}
