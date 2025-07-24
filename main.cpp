#include "particle.h"
#include "parameters.h"


int main(){
    auto& params=Parameters::getInstance();
    params.loadFromFile("../config.txt");
    int numParticles=params.getInt("particle_num",1000);
    real temperature=params.getFloat("temperature",0);
    printf("numPartciles:%d\n",numParticles);
    printf("temperature:%f\n",temperature);
}