#pragma once
#include "Layer.h"
#include <random>
class DenseLayer : public Layer {
public:
    Tensor weights;
    Tensor bias;
    Tensor grad_weights;
    Tensor grad_bias;
private:
    Tensor last_input; 
public:
    DenseLayer(int input_size, int output_size) {
        weights = Tensor({input_size, output_size});
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);
        for (auto& w : weights.data) {
            w = distribution(generator);
        }
        bias = Tensor({1, output_size}); 
    }
    Tensor forward(const Tensor& input) override {
        last_input = input;
        Tensor output = Tensor::dot(input, weights);
        for (int i = 0; i < output.shape[0]; ++i) {
            for (int j = 0; j < output.shape[1]; ++j) {
                output.at(i, j) += bias.data[j];
            }
        }
        return output;
    }
    Tensor backward(const Tensor& output_gradient) override {
        Tensor last_input_T({last_input.shape[1], last_input.shape[0]});
        for(int i = 0; i < last_input.shape[0]; ++i)
            for(int j = 0; j < last_input.shape[1]; ++j)
                last_input_T.at(j,i) = last_input.at(i,j);
        grad_weights = Tensor::dot(last_input_T, output_gradient);
        grad_bias = Tensor({1, output_gradient.shape[1]});
        for (int j = 0; j < output_gradient.shape[1]; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < output_gradient.shape[0]; ++i) {
                sum += output_gradient.at(i, j);
            }
            grad_bias.data[j] = sum;
        }
        Tensor weights_T({weights.shape[1], weights.shape[0]});
        for(int i = 0; i < weights.shape[0]; ++i)
            for(int j = 0; j < weights.shape[1]; ++j)
                weights_T.at(j,i) = weights.at(i,j);
        return Tensor::dot(output_gradient, weights_T);
    }
    std::unique_ptr<Layer> clone() const override {
        auto new_layer = std::make_unique<DenseLayer>(weights.shape[0], weights.shape[1]);
        new_layer->weights = this->weights;
        new_layer->bias = this->bias;
        return new_layer;
    }
};