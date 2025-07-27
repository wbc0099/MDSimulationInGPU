#include "particle.h"
#include "parameters.h"
#include "gpuConfig.h"


int main(){
    printf("===========================read file test=====================================\n"); 
    auto& params=Parameters::getInstance();
    params.loadFromFile("../config.txt");
    int numParticles=params.getInt("particle_num",1000);
    real temperature=params.getFloat("temperature",0);
    printf("numPartciles:%d\n",numParticles);
    printf("temperature:%f\n",temperature);

    printf("=========================configurate gpu test=====================================\n");
    GPUConfig config(true,true);
    config.initialize();
    dim3 grid = config.getGridSize(numParticles);
    dim3 block = config.getBlockSize();

    printf("====================particle test=====================================\n"); 
    int seed=time(NULL);
    Particle particle(100);
    particle.allocateDeviceMemory();
    particle.initRandomStates(numParticles,seed,grid,block);
    particle.moveAllDataToCPU();
    particle.moveAllDataToGPU(0);
}