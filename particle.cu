#include "particle.h"
#include "cuda_runtime.h"
#include <iostream>
#include <tuple>
#include <utility> 

inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line 
                  << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::allocateDeviceMemory(){
    auto pointersReal=Particle::getDeviceRealPointers();
    std::apply([this](auto&&... ptrs){
        (checkCudaError(cudaMalloc(&ptrs, numParticles * sizeof(real)),__FILE__,__LINE__), ...);
    }, pointersReal);
    auto pointersInt=Particle::getDeviceIntPointers();
    std::apply([this](auto&&... ptrs){
        (checkCudaError(cudaMalloc(&ptrs, numParticles * sizeof(int)),__FILE__,__LINE__), ...);
    }, pointersInt);
}

__global__ void initRandomStates(curandState* states, int numParticles, unsigned long long seed){
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles){
        curand_init(seed, idx, 0, &states[idx]);
    }
}

void Particle::initState(){
    checkCudaError(cudaMalloc(&state,numParticles*sizeof(curandState)),__FILE__,__LINE__);
    initRandomStates<<<gridSize,blockSize>>>(state,numParticles,time(NULL));
    checkCudaError(cudaDeviceSynchronize(),__FILE__,__LINE__);
}

// void generateParticles(real minX, real maxX, real minY, real maxY, 
//     real Temperature, int generateMode){
        
// };

