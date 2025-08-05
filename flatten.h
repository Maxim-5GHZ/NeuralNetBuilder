#include "lay.h"
#include <vector>
#include <stdexcept>
#include <iostream>

#pragma once

template<typename T>
class Flatten : public Lay<T> {
private:
    size_t m_input_size;

public:
    Flatten() = default;

    std::string getType() const override { return "Flatten"; }

    void save(std::ostream& out) const override {
        
    }

    void load(std::istream& in) override {
      
    }

    std::vector<T> forward(const std::vector<T>& input) override {
        m_input_size = input.size();
        return input;
    }

    std::vector<T> backward(const std::vector<T>& output_gradient) override {
        if (output_gradient.size() != m_input_size) {
            throw std::runtime_error("Flatten: output gradient size mismatch");
        }
        return output_gradient;
    }

    void update_weights(T learning_rate) override {}
};