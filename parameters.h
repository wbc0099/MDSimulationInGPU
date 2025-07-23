#pragma once
#include <string>
#include <map>

class Parameters{
private:
    std::map<std::string, float> floatParams;
    std::map<std::string, int> intParams;
public:
    static Parameters& getInstance();
    void loadFromFile(const std::string& filename);
    float getFloat(const std::string& key, float defaultValue = 0.0f);
    int getInt(const std::string& key, int defaultValue = 0);
private:
    Parameters() = default;
};