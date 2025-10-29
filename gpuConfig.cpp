#include "gpuConfig.h"
#include <iostream>
#include <vector>

GPUConfig::GPUConfig(int selectID, bool verbose_)
    : verbose(verbose_) // initial list, assign the passed verbose to the member variable verbose
{
    if (selectID == -1) {
        deviceID = autoSelectDevice();
    } else if (selectID < -1) {
        std::cerr << "[ERROR] Invalid device ID: " << selectID << std::endl;
        exit(EXIT_FAILURE);
    } else {
        deviceID = selectID;
    }

    setBlockSize();
}

void GPUConfig::initialize(){
    cudaSetDevice(deviceID);
    if(verbose){
        printDeviceInfo();
    }
}

int GPUConfig::autoSelectDevice(){
    int count=0;
    cudaGetDeviceCount(&count);

    size_t maxFreeMem=0;
    int bestDevice=0;

    for(int i=0; i<count; ++i){
        cudaSetDevice(i);

        size_t freeMem=0,totalMem=0;
        cudaMemGetInfo(&freeMem,&totalMem);

        if(verbose){
            std::cout << "[Device " << i << "] Free Memory:" << freeMem / (1024*1024) << " MB\n";
        }

        if(freeMem>maxFreeMem){
            maxFreeMem=freeMem;
            bestDevice=i;
        }
    }
    return bestDevice;
}

void GPUConfig::printDeviceInfo() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);

    std::cout << "[Using GPU] Device " << deviceID << ": " << prop.name << "\n";
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
}

void GPUConfig::checkCUDAError(const std::string& msg) const {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        std::cerr << "[CUDA ERROR]" << msg << ": " << cudaGetErrorString(err) << std::endl;
    }
}

int GPUConfig::getDeviceID() const{
    return deviceID;
}

void GPUConfig::setBlockSize(){
    blockSize=dim3(256);
}

dim3 GPUConfig::getBlockSize() const {
    return blockSize;
}

dim3 GPUConfig::getGridSize(int N) const {
    int blocks=(N+blockSize.x -1) / blockSize.x;
    return dim3(blocks);
}