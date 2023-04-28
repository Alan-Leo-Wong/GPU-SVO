#include "SVO.h"

int main(int argc, char** argv)
{
    SparseVoxelOctree svo("switchmec.obj", 128, 128, 128);
    svo.createOctree();
    svo.writeTree("switchmec");
    
    return 0;
}
