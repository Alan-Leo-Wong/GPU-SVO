#include "SVO.h"

int main(int argc, char** argv)
{
    SparseVoxelOctree svo("bunny.obj", 16, 16, 16);
    svo.createOctree();
    svo.writeTree("bunny");
    
    return 0;
}
