#pragma once
#include "define.h"
#include <curand_kernel.h>
#include <tuple>
#include <utility> 

class Particle{
    using RealPtr = real*;
    using IntPtr = int*;
    RealPtr x,y,fx,fy,kBT,x_updateHL,y_updateHL;
    IntPtr cell_nx,cell_ny,cell_n,NL,senseNumber;
    int numParticles;
    curandState* state;
    int blockSize=256;
    int gridSize=(numParticles+255)/256;

    auto getDeviceRealPointers(){
        return std::tie(x,y,fx,fy,kBT,x_updateHL,y_updateHL,kBT);
    }

    auto getDeviceIntPointers(){
        return std::tie(cell_nx,cell_ny,cell_n,NL,senseNumber);
    }

    Particle(int numParticles);
    ~Particle();

    void generateParticles(real minX, real maxX, real minY, real maxY, 
        real Temperature, int generateMode);

    void allocateDeviceMemory();
    void freeDeviceMemory();
    void copyToHost();  
    void initState();
};