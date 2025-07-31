#include<vector>
#include<memory>
#include"lay.h"
#include"dense.h"
#pragma once

struct ModelConfig{
    std::vector<Lay> lay;
};


template<typename T>
class Model{

public:
Model(ModelConfig config){

}
};