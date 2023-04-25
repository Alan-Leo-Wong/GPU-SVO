#include "SVO.h"

int main(int argc, char** argv)
{
    SparseVoxelOctree svo("bunny.obj", 8, 8, 8);
    svo.createOctree();
    svo.writeTree("bunny");
    
    return 0;
}
