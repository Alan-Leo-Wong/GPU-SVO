#pragma once
#include "CUDAUtil.h"
#include <Eigen\Dense>
#include <stdint.h>

// An Axis Aligned Box (AAB) of a certain type - to be initialized with a min and max
template <typename Real>
struct AABox {
	Real min;
	Real max;
	Real width;
	
	CUDA_CALLABLE_MEMBER AABox() : min(Real()), max(Real()) {}
	CUDA_CALLABLE_MEMBER AABox(Real _min, Real _max) : min(_min), max(_max) {}
};

class BaseModel
{
protected:
	uint32_t nTris;
	uint32_t nPoints;

	AABox<Eigen::Vector3f> modelBBox;
};