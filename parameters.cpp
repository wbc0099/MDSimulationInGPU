#include "parameters.h"
#include <fstream>
#include <sstream>
#include "define.h"

Parameters& Parameters::getInstance(){
    static Parameters instance;
    return instance;
}

void Parameters::loadFromFile(const std::string& filename){
    std::ifstream file(filename);
    if(!file.is_open()){
        throw std::runtime_error("无法打开参数文件： " + filename);
    }
    std::string line;

    while (std::getline(file, line)){
        std::istringstream iss(line);
        std::string key, type,value;

        if(iss >> key >> type>>value){
            if(type == "float"){
                floatParams[key] = std::stof(value);
            }else if (type == "int"){
                intParams[key]=std::stoi(value);
            }
        }
    }
}

float Parameters::getFloat(const std::string& key, float defaultValue){
    return floatParams.count(key) ? floatParams[key] : defaultValue;
}

int Parameters::getInt(const std::string& key, int defaultValue){
    return intParams.count(key) ? intParams[key] : defaultValue;
}

//============================ test =======================================
// int main(){
//     auto& params=Parameters::getInstance();
//     params.loadFromFile("config.txt");
//     int numParticles=params.getInt("particle_num",1000);
//     real temperature=params.getFloat("temperature",0);
//     printf("numPartciles:%d\n",numParticles);
//     printf("temperature:%f\n",temperature);
// }