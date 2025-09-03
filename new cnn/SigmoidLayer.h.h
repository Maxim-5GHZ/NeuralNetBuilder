#pragma once
#include "Layer.h"
#include <cmath>
class SigmoidLayer : public Layer {
private:
    Tensor last_output; 
public:
    Tensor forward(const Tensor& input) override {
        Tensor output = input;
        for (size_t i = 0; i < output.data.size(); ++i) {
            output.data[i] = 1.0f / (1.0f + exp(-input.data[i]));
        }
        last_output = output; 
        return output;
    }
    Tensor backward(const Tensor& output_gradient) override {
        Tensor input_gradient = output_gradient;
        for (size_t i = 0; i < last_output.data.size(); ++i) {
            float s = last_output.data[i];
            input_gradient.data[i] = output_gradient.data[i] * (s * (1.0f - s));
        }
        return input_gradient;
    }
};