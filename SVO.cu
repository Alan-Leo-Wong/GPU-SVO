#include "SVO.h"
#include "CUDAUtil.h"
#include "MortonLUT.h"
#include <thrust/device_vector.h>

// Set a bit in the giant voxel table. This involves doing an atomic operation on a 32-bit word in memory.
// Blocking other threads writing to it for a very short time
__device__ __inline__ void setBit(uint32_t* voxel_table, size_t index) {
	size_t int_location = index / size_t(32); // voxels come in groups of 32
	// we count bit positions RtL, but bit array indices(i.e. group index) LtR (To reverse bit positions in RtL to LtR, we should perform '31 - index % 32')
	uint32_t bit_pos = size_t(31) - (index % size_t(32));
	uint32_t mask = 1 << bit_pos;
	atomicOr(&(voxel_table[int_location]), mask);
}

__global__ void surfaceVoxelize(const int nTris,
	const Eigen::Vector3i surfaceVoxelGridSize,
	const AABox<Eigen::Vector3f> modelBBox,
	const Eigen::Vector3f unitVoxelSize,
	float* d_triangle_data,
	uint32_t* d_voxelTable)
{
	size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	Eigen::Vector3f delta_p{ unitVoxelSize.x(), unitVoxelSize.y(), unitVoxelSize.z() };
	Eigen::Vector3i grid_max{ surfaceVoxelGridSize.x() - 1, surfaceVoxelGridSize.y() - 1, surfaceVoxelGridSize.z() - 1 }; // grid max (grid runs from 0 to gridsize-1)

	while (thread_id < nTris) { // every thread works on specific triangles in its stride
		size_t t = thread_id * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using bbox
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

					size_t location = mortonEncode_LUT(x, y, z);
					setBit(d_voxelTable, location);
				}
			}
		}
		thread_id += stride;
	}
}

void SparseVoxelOctree::constructFineNodes()
{
	uint32_t* d_voxelTabel;
	size_t voxelTabelize = surfaceVoxelGridSize.x() * surfaceVoxelGridSize.y() * surfaceVoxelGridSize.z() / 8;
	CUDA_CHECK(cudaMalloc((void**)&d_voxelTabel, sizeof(uint32_t) * voxelTabelize));
	CUDA_CHECK(cudaMemset(d_voxelTabel, 0, sizeof(uint32_t) * voxelTabelize));

	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the  maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 
	getOccupancyMaxPotentialBlockSize(nTris, minGridSize, blockSize, gridSize, surfaceVoxelize, 0, 0);

	Eigen::Vector3f unitVoxelSize{
		(modelBBox.max.x() - modelBBox.min.x()) / surfaceVoxelGridSize.x(),
		(modelBBox.max.y() - modelBBox.min.y()) / surfaceVoxelGridSize.y(),
		(modelBBox.max.z() - modelBBox.min.z()) / surfaceVoxelGridSize.z(),
	};
}
