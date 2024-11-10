# GPU Sparse Voxel Octree

This repository contains a GPU-accelerated implementation of a sparse voxel octree (SVO) based on the techniques described in the papers *"Fast Parallel Surface and Solid Voxelization on GPUs"* ([ACM](https://dl.acm.org/doi/10.1145/1882261.1866201)) and *"Data-Parallel Octrees for Surface Reconstruction"* ([IEEE](https://ieeexplore.ieee.org/abstract/document/5473223)). The project leverages CUDA for parallel processing to efficiently voxelized given 3D models and structure them into an SVO using Morton codes.

## Key Features

- **Surface Voxelization**: Efficient GPU-based surface voxelization using the methodology outlined in [cuda_voxelizer](https://github.com/Forceflow/cuda_voxelizer).
- **Sparse Voxel Octree Construction**: Constructs SVOs from voxelized data using a bottom-up approach with Morton codes for indexing and storage.
- **Geometry Primitives Storage**: Utilizes Morton codes to store non-redundant geometric primitives - vertices, edges, and faces of voxels.

## Prerequisites

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher.
- CUDA Toolkit 11.6 or higher.
- C++ compiler compatible with the CUDA version used. (C++17 at least)
- CMake(>=3.18) for building the project.

## Setup

```bash
git clone https://github.com/Alan-Leo-Wong/GPU-SVO.git
cd GPU-SVO
mkdir build && cd build
cmake ..
cmake --build . -j your-core-number
```

## Usage

After compiling the project, you can run the executable using:

```
./main <input-mesh.obj> <resolution>
```

- `<input-mesh.obj>`: Path to the input .obj file.
- `<resolution>`: Desired voxel resolution (e.g., 128).

The output will be an .obj file named `<mesh_name>_depth_svo.obj`, which represents the constructed sparse voxel octree at the specified resolution.