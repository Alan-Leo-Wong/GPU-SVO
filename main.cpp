#include <string>
#include <filesystem>
#include "src/SVO.h"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    const std::string inMesh = argc > 1 ? argv[1] : "bunny.obj";
    const int res = argc > 2 ? std::stoi(argv[2]) : 64;
    SparseVoxelOctree svo(inMesh, res, res, res);
    svo.create();

    const std::string inMeshName = fs::path(inMesh).stem().string();
    svo.writeTree(inMeshName);

    return 0;
}
