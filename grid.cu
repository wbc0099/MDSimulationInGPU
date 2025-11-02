#include "grid.h"
#include "define.h"
#include "cuda_runtime.h"
#include <cstdio>
#include <cmath>

using namespace std;

Grid::Grid(real boxLX, real boxLY, real cellLX, real cellLY, int maxParticlesPerCell) 
    :boxLX(boxLX),boxLY(boxLY),cellLX(cellLX),cellLY(cellLY){
        printf("Grid initialized with boxLX: %f, boxLY: %f, cellLX: %f, cellLY: %f\n", boxLX, boxLY, cellLX, cellLY);
    cellXN=boxLX/cellLX;
    cellYN=boxLY/cellLY;
    printf("cellXN: %d\n", cellXN);
    printf("cellYN: %d\n", cellYN);
    CHECK_CUDA(cudaMallocManaged(&cellList, cellXN*cellYN*maxParticlesPerCell*sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&cellListOffset, cellXN*cellYN*sizeof(int)));
    
}

void Grid::initCellList(){
    CHECK_CUDA(cudaMemset(cellListOffset, 0, cellXN*cellYN*sizeof(int)));
}

__global__ void GenerateCellList1(int num, real* x,real* y, real cellLX, real cellLY, int cellXN, int* cellList, int* offsets, int maxParticlesPerCell){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=num) return;
    int cellX, cellY, cellID;
    cellX=floor(x[id]/cellLX);
    cellY=floor(y[id]/cellLY);
    cellID=cellY*cellXN+cellX;
    int offset=atomicAdd(&offsets[cellID],1);
    if (offset<maxParticlesPerCell){
        cellList[cellID*maxParticlesPerCell+offset]=id;
    }else{
        printf("wrong! OffsetCL is greater than maxParticlePerCell");
    }
    printf("id: %d\n", cellID);
    // if ()
 
}

void Grid::GenerateCellList(dim3 blockNum, dim3 threadNum, int num, real* x, real* y, int maxParticlesPerCell){
    GenerateCellList1 <<<blockNum,threadNum>>> (num, x, y, cellLX, cellLY, cellXN, cellList, cellListOffset, maxParticlesPerCell);
}
