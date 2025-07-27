#pragma once
#include "define.h"
#include <curand_kernel.h>
#include <tuple>
#include <utility> 

class Particle{
public:
    using RealPtr = real*;
    using IntPtr = int*;
    RealPtr x,y,fx,fy,kBT,x_updateHL,y_updateHL;
    //because NL is a list, we can not alloc memroy to NL with getDeviceIntPointers;
    IntPtr NL,cell_nx,cell_ny,cell_n,senseNumber;
    int numParticles;
    curandState* states;
    int blockSize=256;
    int gridSize=(numParticles+255)/256;

    auto getDeviceRealPointers(){
        return std::tie(x,y,fx,fy,kBT,x_updateHL,y_updateHL,kBT);
    }

    auto getDeviceIntPointers(){
        return std::tie(cell_nx,cell_ny,cell_n,senseNumber);
    }

    auto getDeviceIntListPointers(){
        return std::tie(NL);
    }

    Particle(int numParticles);
    ~Particle();

    void generateParticles(real minX, real maxX, real minY, real maxY, 
        real Temperature, int generateMode);

    void allocateDeviceMemory();
    void freeDeviceMemory();
    void copyToHost();
    void initPosition(int mode, real boxLX, real boxLY, real cellLX, real cellLY);
    void initRandomStates(int numParticles, unsigned long long seed, dim3 gridSize, dim3 blockSize);
    void moveAllDataToGPU(int GPUID);
    void moveAllDataToCPU();
};