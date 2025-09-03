#pragma once
#include "Tensor.h"
#include <memory>
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output_gradient) = 0;
    virtual std::unique_ptr<Layer> clone() const = 0;
};