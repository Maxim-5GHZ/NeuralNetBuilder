#pragma once
#include "Layer.h"
#include <random>
#include <vector>
#include <chrono>
class DropoutLayer : public Layer {
private:
    float rate;
    Tensor mask;
    bool is_training = true;
    std::default_random_engine generator;
public:
    DropoutLayer(float dropout_rate) : rate(dropout_rate) {
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    void set_training_mode(bool training) {
        is_training = training;
    }
    Tensor forward(const Tensor& input) override {
        if (!is_training || rate == 0.0f) {
            return input;
        }
        mask = Tensor(input.shape);
        std::bernoulli_distribution distribution(1.0f - rate);
        float scale = (rate < 1.0f) ? (1.0f / (1.0f - rate)) : 0.0f;
        Tensor output = input;
        for (size_t i = 0; i < mask.data.size(); ++i) {
            if (distribution(generator)) {
                mask.data[i] = 1.0f;
                output.data[i] *= scale; 
            } else {
                mask.data[i] = 0.0f;
                output.data[i] = 0.0f;
            }
        }
        return output;
    }
    Tensor backward(const Tensor& output_gradient) override {
        if (!is_training || rate == 0.0f) {
            return output_gradient;
        }
        Tensor input_gradient = output_gradient;
        float scale = (rate < 1.0f) ? (1.0f / (1.0f - rate)) : 0.0f;
        for (size_t i = 0; i < input_gradient.data.size(); ++i) {
            input_gradient.data[i] *= mask.data[i] * scale;
        }
        return input_gradient;
    }
};