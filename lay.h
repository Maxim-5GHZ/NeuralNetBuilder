#include <functional>
#include <vector>
#include <iostream>
#include <string>
#pragma once

template<typename T>
class Lay {
public:
    Lay() = default;
    virtual ~Lay() = default;
    virtual std::vector<T> forward(const std::vector<T>& input) = 0;
    virtual std::vector<T> backward(const std::vector<T>& output_gradient) = 0;
    virtual void update_weights(T learning_rate) {}
    
    virtual void save(std::ostream& out) const = 0;
    virtual void load(std::istream& in) = 0;
    virtual std::string getType() const = 0;
};