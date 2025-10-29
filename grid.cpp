#include "grid.h"
#include "define.h"
#include <cstdio>

Grid::Grid(real boxLX, real boxLY, real cellLX, real cellLY) 
    :boxLX(boxLX),boxLY(boxLY),cellLX(cellLX),cellLY(cellLY){
        printf("Grid initialized with boxLX: %f, boxLY: %f, cellLX: %f, cellLY: %f\n", boxLX, boxLY, cellLX, cellLY);
}