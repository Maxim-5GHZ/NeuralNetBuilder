#pragma once
#include "Sequential.h"
#include "DenseLayer.h" 

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Sequential& model) = 0;
};

class SGD : public Optimizer {
private:
    float learning_rate;

public:
    SGD(float lr = 0.01f) : learning_rate(lr) {}

    void step(Sequential& model) override {
        for (const auto& layer_ptr : model.layers) {
           
            DenseLayer* dense_layer = dynamic_cast<DenseLayer*>(layer_ptr.get());
            if (dense_layer) {
                
                for (size_t i = 0; i < dense_layer->weights.data.size(); ++i) {
                    dense_layer->weights.data[i] -= learning_rate * dense_layer->grad_weights.data[i];
                }
                
                for (size_t i = 0; i < dense_layer->bias.data.size(); ++i) {
                    dense_layer->bias.data[i] -= learning_rate * dense_layer->grad_bias.data[i];
                }
            }
        }
    }
};