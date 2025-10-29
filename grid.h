#include "define.h"

class Grid{
public:
    real boxLX, boxLY, cellLX, cellLY;
    real* CellAroundList;
    Grid(real boxLX, real boxLY, real cellLX, real cellLY);
    generateCellAroundList();
};