#pragma once
#include "cuda_runtime.h"
#include "iostream"

#define PI 3.141592653589793
#define real double

#define CHECK_CUDA(err) checkCudaError(err, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line 
                  << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}