#include "particle.h"
#include "parameters.h"
#include "gpuConfig.h"
#include "grid.h"
#include "tools.h"


int main(){
    printf("===========================read file test=====================================\n"); 
    auto& params=Parameters::getInstance();
    params.loadFromFile("../config.txt");
    int numParticles=params.getInt("particle_num",1000);
    real temperature=params.getFloat("temperature",0);
    int maxParticlesPerCell=params.getInt("maxParticlesPerCell",100);
    printf("numPartciles:%d\n",numParticles);
    printf("temperature:%f\n",temperature);
    printf("maxParticlesPerCell:%d\n",maxParticlesPerCell);

    printf("=========================configurate gpu test=====================================\n");
    GPUConfig config(true,true);
    config.initialize();
    dim3 gridSize = config.getGridSize(numParticles);
    dim3 blockSize = config.getBlockSize();

    printf("====================particle test=====================================\n"); 
    int seed=time(NULL);
    Particle particle(100);
    particle.allocateDeviceMemory();
    particle.initRandomStates(numParticles,seed,gridSize,blockSize);
    particle.moveAllDataToCPU();
    particle.moveAllDataToGPU(0);
    //particle.initPosition(0,240,240,10,10);

    printf("============================== grid test ==============================\n");
    real boxLX=params.getFloat("boxLX");
    real boxLY=params.getFloat("boxLY");
    real cellLX=params.getFloat("cellLX");
    real cellLY=params.getFloat("cellLY");
    Grid grid(boxLX,boxLY,cellLX,cellLY,maxParticlesPerCell);
    grid.initCellList();
    // grid.GenerateCellList(gridSize,blockSize,numParticles, )

    printf("========================= tools test=====================================\n"); 
    real drTest=distancePBC(1,1,9,9,10,10);
    printf("DistanceTest return: %f\n",drTest);
    real xNew,yNew;
    movePositionPBC(1,1,-5,12,10,10,xNew,yNew);
    printf("xNew,yNew: %f,%f\n", xNew,yNew);
}