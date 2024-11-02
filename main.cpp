#include "SVO.h"

int main(int argc, char **argv) {
    SparseVoxelOctree svo("bunny.obj", 32, 32, 32);
    svo.create();
    svo.writeTree("bunny");

    return 0;
}
