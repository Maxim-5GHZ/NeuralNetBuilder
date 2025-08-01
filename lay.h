#include <functional>
#include <vector>
#pragma once

template<typename T>
class Lay{
public:
    Lay() = default;
    virtual ~Lay() = default;
    virtual std::vector<T> forward(const std::vector<T>& input) = 0;
};