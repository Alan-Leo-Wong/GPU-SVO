#pragma once
#include <stdio.h>
#include <iostream>
#include <Eigen\Dense>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        const cudaError_t error_code = call;                                  \
        if (error_code != cudaSuccess)                                        \
        {                                                                     \
            fprintf(stderr, "CUDA Error:\n");                                          \
            fprintf(stderr, "    --File:       %s\n", __FILE__);                       \
            fprintf(stderr, "    --Line:       %d\n", __LINE__);                       \
            fprintf(stderr, "    --Error code: %d\n", error_code);                     \
            fprintf(stderr, "    --Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                     \
    } while (0);

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char* errorMessage, const char* file, const int line)
{
    const cudaError_t error_code = cudaGetLastError();

    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "%s(%d) : getLastCudaError() CUDA Error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(error_code), cudaGetErrorString(error_code));
        exit(EXIT_FAILURE);
    }
}

template<class Real>
static inline __host__ void getOccupancyMaxPotentialBlockSize(const uint32_t& dataSize,
    int&    minGridSize,
    int&    blockSize,
    int&    gridSize,
    Real       func,
    size_t dynamicSMemSize = 0,
    int    blockSizeLimit = 0)
{
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit);
    gridSize = (dataSize + blockSize - 1) / blockSize;
}

inline CUDA_CALLABLE_MEMBER float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline CUDA_CALLABLE_MEMBER float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline CUDA_CALLABLE_MEMBER int mini(int a, int b)
{
    return a < b ? a : b;
}

inline CUDA_CALLABLE_MEMBER int maxi(int a, int b)
{
    return a > b ? a : b;
}

inline CUDA_CALLABLE_MEMBER Eigen::Vector3f fminf(Eigen::Vector3f a, Eigen::Vector3f b)
{
    return Eigen::Vector3f(fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()));
}

inline CUDA_CALLABLE_MEMBER Eigen::Vector3f fmaxf(Eigen::Vector3f a, Eigen::Vector3f b)
{
    return Eigen::Vector3f(fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()));
}

inline CUDA_CALLABLE_MEMBER float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

inline CUDA_CALLABLE_MEMBER int clamp(int f, int a, int b)
{
    return maxi(a, mini(f, b));
}

inline CUDA_CALLABLE_MEMBER Eigen::Vector3f clamp(Eigen::Vector3f v, Eigen::Vector3f a, Eigen::Vector3f b)
{
    return Eigen::Vector3f(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()), clamp(v.z(), a.z(), b.z()));
}

inline CUDA_CALLABLE_MEMBER Eigen::Vector3i clamp(Eigen::Vector3i v, Eigen::Vector3i a, Eigen::Vector3i b)
{
    return Eigen::Vector3i(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()), clamp(v.z(), a.z(), b.z()));
}