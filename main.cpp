#include "SVO.h"

int main(int argc, char** argv)
{
    SparseVoxelOctree svo("bunny.off", 32, 32, 32);
    //svo.createOctree();
    //svo.writeTree("bunny");
    
    return 0;
}
