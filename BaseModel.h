#pragma once

#include "CUDAUtil.h"
#include <fstream>
#include <stdint.h>
#include <Eigen/Dense>

// An Axis Aligned Box (AAB) of a certain type - to be initialized with a min and max
template<typename Real>
struct AABox {
    Real min;
    Real max;
    Real width;

    CUDA_CALLABLE_MEMBER

    AABox() : min(Real()), max(Real()), width(Real()) {}

    CUDA_CALLABLE_MEMBER

    AABox(Real _min, Real _max) : min(_min), max(_max), width(_max - _min) {}
};

template<typename T>
AABox<T> createMeshBBCube(AABox<T> box) {
    AABox<T> answer(box.min, box.max); // initialize answer
    Eigen::Vector3f lengths = box.max - box.min; // check length of given bbox in every direction
    float max_length = fmaxf(lengths.x(), fmaxf(lengths.y(), lengths.z())); // find max length
    for (unsigned int i = 0; i < 3; i++) { // for every direction (X,Y,Z)
        if (max_length == lengths[i]) {
            continue;
        } else {
            float delta =
                    max_length - lengths[i]; // compute difference between largest length and current (X,Y or Z) length
            answer.min[i] = box.min[i] - (delta / 2.0f); // pad with half the difference before current min
            answer.max[i] = box.max[i] + (delta / 2.0f); // pad with half the difference behind current max
        }
    }

    // Next snippet adresses the problem reported here: https://github.com/Forceflow/cuda_voxelizer/issues/7
    // Suspected cause: If a triangle is axis-aligned and lies perfectly on a voxel edge, it sometimes gets counted / not counted
    // Probably due to a numerical instability (division by zero?)
    // Ugly fix: we pad the bounding box on all sides by 1/10001th of its total length, bringing all triangles ever so slightly off-grid
    Eigen::Vector3f epsilon = (answer.max - answer.min) / 10001.0f;
    answer.min -= epsilon;
    answer.max += epsilon;
    answer.width = answer.max - answer.min;
    return answer;
}

class BaseModel {
protected:
    size_t nModelTris;
    size_t nPoints;

    std::vector<Eigen::Vector3i> idx2Points;
    std::vector<Eigen::Vector3f> modelPoints;

    AABox<Eigen::Vector3f> modelBBox;

public:
    BaseModel() {}

    BaseModel(const std::string &filename) {
        loadOBJ(filename);
        nModelTris = idx2Points.size();
        nPoints = modelPoints.size();
        Eigen::Map <Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> pointsMat(
                reinterpret_cast<float *>(modelPoints.data()), modelPoints.size(), 3);
        Eigen::Vector3f minPoint = pointsMat.colwise().minCoeff();
        Eigen::Vector3f maxPoint = pointsMat.colwise().maxCoeff();
        modelBBox = createMeshBBCube(AABox<Eigen::Vector3f>(minPoint, maxPoint));
    }

    void loadOBJ(const std::string &in_file);
};