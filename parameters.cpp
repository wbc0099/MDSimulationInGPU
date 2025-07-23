#include "parameters.h"
#include <fstream>
#include <sstream>

Parameters& Parameters::getInstance(){
    static Parameters instance;
    return instance;
}

void Parameters::loadFromFile(const std::string& filename){
    std::ifstream file(filename);
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

int main(){
    auto& params=Parameters.getInstance();
    params.loadFromFile("config.txt");
}