#include "define.h"

class Grid{
public:
    real boxLX, boxLY, cellLX, cellLY;
    int cellXN, cellYN;
    int *cellList, *cellListOffset;
    
    Grid(real boxLX, real boxLY, real cellLX, real cellLY, int maxParticlesPerCell);
    void initCellList();
    void GenerateCellList(dim3 blockNum, dim3 threadNum, int num, real* x, real* y, int maxParticlesPerCell);
    // void GenerateCellAroundList(int num, real* x,real* y);
};