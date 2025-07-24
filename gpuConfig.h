#pragma onece

#include "cuda_runtime.h"
#include <string>

class GPUConfig{
public:
    GPUConfig(int selectID = -1, bool verbose = true);

    void initialize();
    void printDeviceInfo() const;
    void checkCUDAError(const std::string& msg) const;

    int getDeviceID() const;
    dim3 getBlockSize() const;
    dim3 getGridSize(int N) const;

private:
    int deviceID;
    bool verbose;
    dim3 blockSize;

    void setBlockSize();
    int autoSelectDevice();
};