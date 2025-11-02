#include "define.h"
#include "particle.h"
#include "cuda_runtime.h"
#include <iostream>
#include <tuple>
#include <utility> 
#include <random>
#include "tools.h"

void Particle::allocateDeviceMemory(){
    auto pointersReal=Particle::getDeviceRealPointers();
    std::apply([this](auto&&... ptrs){
        (CHECK_CUDA(cudaMallocManaged(&ptrs, numParticles * sizeof(real))), ...);
    }, pointersReal);
    auto pointersInt=Particle::getDeviceIntPointers();
    std::apply([this](auto&&... ptrs){
        (CHECK_CUDA(cudaMallocManaged(&ptrs, numParticles * sizeof(int))), ...);
    }, pointersInt);
    auto pointersIntList=Particle::getDeviceIntListPointers();
    std::apply([this](auto&&... ptrs){
        (CHECK_CUDA(cudaMallocManaged(&ptrs, numParticles * 1024 * sizeof(int))),...);
    }, pointersIntList);
    CHECK_CUDA(cudaMalloc(&states,numParticles * sizeof(curandState)));
}

__global__ void initRandomStatesGlobal(curandState* states, int numParticles, unsigned long long seed){
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles){
        curand_init(seed, idx, 0, &states[idx]);
        if (idx < 10) {
            printf("Thread %d initialized curandState with seed %llu\n", idx, seed);
        }
    }
}

void Particle::initRandomStates(int numParticles, unsigned long long seed, dim3 gridSize, dim3 blockSize){
    initRandomStatesGlobal<<<gridSize, blockSize>>>(states, numParticles, seed);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void Particle::freeDeviceMemory(){
    auto pointersReal=Particle::getDeviceRealPointers();
    std::apply([this](auto&&... ptrs){
        ((cudaFree(ptrs),ptrs=nullptr), ...);
    },pointersReal);

    auto pointersInt=Particle::getDeviceIntPointers();
    std::apply([this](auto&&... ptrs){
        ((cudaFree(ptrs),ptrs=nullptr), ...);
    },pointersInt);

    auto pointersIntList=Particle::getDeviceIntListPointers();
    std::apply([this](auto&&... ptrs){
        ((cudaFree(ptrs),ptrs=nullptr), ...);
    },pointersIntList);
}

Particle::Particle(int numParticles){
    this->numParticles = numParticles;
}

Particle::~Particle(){
    freeDeviceMemory();
}

void Particle::moveAllDataToCPU(){
    auto pointersReal=Particle::getDeviceIntListPointers();
    //printf("cudaCpuDeviceId is : %d\n", cudaCpuDeviceId); //every device can only have one host(cpu)
    std::apply([this](auto&&... ptrs){
        (CHECK_CUDA(cudaMemPrefetchAsync(ptrs, numParticles*sizeof(real), cudaCpuDeviceId)), ...);
    }, pointersReal);
    auto pointersInt=Particle::getDeviceIntPointers();
    std::apply([this](auto&&... ptrs){
        (CHECK_CUDA(cudaMemPrefetchAsync(ptrs, numParticles*sizeof(int), cudaCpuDeviceId)), ...);
    }, pointersInt);
    auto pointersIntList=Particle::getDeviceIntListPointers();
    std::apply([this](auto&&... ptrs){
        (CHECK_CUDA(cudaMemPrefetchAsync(ptrs, numParticles*1024*sizeof(int), cudaCpuDeviceId)), ...);
    }, pointersIntList);
}

void Particle::moveAllDataToGPU(int GPUID){
    auto pointersReal=Particle::getDeviceIntListPointers();
    std::apply([=](auto&&... ptrs){
        (CHECK_CUDA(cudaMemPrefetchAsync(ptrs, numParticles*sizeof(real), GPUID)), ...);
    }, pointersReal);
    auto pointersInt=Particle::getDeviceIntPointers();
    std::apply([=](auto&&... ptrs){
        (CHECK_CUDA(cudaMemPrefetchAsync(ptrs, numParticles*sizeof(int), GPUID)), ...);
    }, pointersInt);
    auto pointersIntList=Particle::getDeviceIntListPointers();
    std::apply([=](auto&&... ptrs){
        (CHECK_CUDA(cudaMemPrefetchAsync(ptrs, numParticles*1024*sizeof(int), GPUID)), ...);
    }, pointersIntList);
}

void Particle::initPosition(int* xAll, int* yAll, int mode, real boxLX, real boxLY, real cellLX, real cellLY, int cellXN, int cellYN, real r0){
    std::default_random_engine e;
    std::uniform_real_distribution <double> u(0.0,1.0);
    e.seed(time(0));
    real x0,y0,dr;
    int particleCellXN, particleCellYN, AroundCellXN,AroundCellYN,AroundCellN;
    int wrongFlag=0;
    int cellList[numParticles*cellXN*cellYN]={0};
    int cellListOffset[cellXN*cellYN]={0};

    for(int i=0;i<numParticles;i++){
        while(1){
            x0=u(e)*boxLX;
            y0=u(e)*boxLY;
            wrongFlag=0;
            particleCellXN=std::floor(x0/cellLX);
            particleCellYN=std::floor(y0/cellLY);
            for(int x=-1;x<=1;x++){
                for(int y=-1;y<=1;y++){
                    if(particleCellXN+x==-1){
                        AroundCellXN=cellXN-1;
                    }else if(particleCellXN+x==cellXN){
                        AroundCellXN=0;
                    }else{
                        AroundCellXN=particleCellXN+x;
                    }
                    if(particleCellYN+y==-1){
                        AroundCellYN=cellYN-1;
                    }else if(particleCellYN+x==cellYN){
                        AroundCellYN=0;
                    }else{
                        AroundCellYN=particleCellYN+y;
                    }
                    AroundCellN=AroundCellXN+AroundCellYN*cellXN;
                    for(int j=0; j<cellListOffset[AroundCellN]; j++){
                        dr=distancePBC(x0, y0, xAll[cellList[j]], yAll[cellList[j]],boxLX,boxLY);
                        if (dr<r0){
                            wrongFlag=1;
                            break;
                        }
                    }
                    if(wrongFlag==1){
                        break;
                    }
                }
                if(wrongFlag==1){
                    break;
                }
            }
            if(wrongFlag==0){
                break;
            }else continue;
        }
        xAll[i]=x0;
        yAll[i]=y0;
        cellList[(particleCellXN+particleCellYN*cellXN)*numParticles+cellListOffset[particleCellXN+particleCellYN*cellXN]]=i;
        cellListOffset[particleCellXN+particleCellYN*cellXN]++;
    }
}


// void Particle::initState(){
//     checkCudaError(cudaMalloc(&state,numParticles*sizeof(curandState)),__FILE__,__LINE__);
//     initRandomStates<<<gridSize,blockSize>>>(state,numParticles,time(NULL));
//     checkCudaError(cudaDeviceSynchronize(),__FILE__,__LINE__);
// }

// void generateParticles(real minX, real maxX, real minY, real maxY, 
//     real Temperature, int generateMode){
        
// };

