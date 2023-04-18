#pragma once
#include "CUDAUtil.h"
#include <fstream>
#include <stdint.h>
#include <Eigen\Dense>

// An Axis Aligned Box (AAB) of a certain type - to be initialized with a min and max
template <typename Real>
struct AABox {
	Real min;
	Real max;
	Real width;

	CUDA_CALLABLE_MEMBER AABox() : min(Real()), max(Real()) {}
	CUDA_CALLABLE_MEMBER AABox(Real _min, Real _max) : min(_min), max(_max) {}
};

// This struct defines VoxelData for our voxelizer.
// This is the main memory hogger: the less data you store here, the better.
struct VoxelData {
	uint32_t morton;

	VoxelData() : morton(0) {}
	VoxelData(const uint_fast64_t& _morton) : morton(_morton) {}

	bool operator > (const VoxelData& a) const {
		return morton > a.morton;
	}

	bool operator < (const VoxelData& a) const {
		return morton < a.morton;
	}
};


class BaseModel
{
protected:
	uint32_t nTris;
	uint32_t nPoints;

	std::vector<Eigen::Vector3i> idx2Points;
	std::vector<Eigen::Vector3f> points;

	AABox<Eigen::Vector3f> modelBBox;

public:
	void loadOBJ(const std::string& in_file);
};

void BaseModel::loadOBJ(const std::string& in_file)
{
	std::ifstream in(in_file);
	if (!in) {
		std::cerr << "ERROR: loading obj:(" << in_file << ") file is not good" << std::endl;
		exit(1);
	}

	float x, y, z;
	int f0, _f0, f1, _f1, f2, _f2;
	char buffer[256] = { 0 };
	while (!in.getline(buffer, 255).eof()) {
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)) {
			if (sscanf_s(buffer, "v %f %f %f", &x, &y, &z) == 3)
				points.emplace_back(Eigen::Vector3f{ x, y, z });
		}
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32)) {
			if (sscanf_s(buffer, "f %d//%d %d//%d %d//%d", &f0, &_f0, &f1, &_f1, &f2, &_f2) == 6)
				idx2Points.emplace_back(Eigen::Vector3i{ f0 - 1, f1 - 1, f2 - 1 });
			else if (sscanf_s(buffer, "f %d %d %d", &f0, &f1, &f2) == 3)
				idx2Points.emplace_back(Eigen::Vector3i{ f0 - 1, f1 - 1, f2 - 1 });
		}
	}
}