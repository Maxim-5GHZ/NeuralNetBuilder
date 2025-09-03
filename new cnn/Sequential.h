#pragma once
#include "Layer.h"
#include <vector>

class Sequential {
public:
    std::vector<std::unique_ptr<Layer>> layers;

    Sequential() = default;

    
    Sequential(const Sequential& other) {
        for (const auto& layer : other.layers) {
            layers.push_back(layer->clone());
        }
    }

    Sequential& operator=(const Sequential& other) {
        if (this == &other) return *this; // self-assignment check
        layers.clear();
        for (const auto& layer : other.layers) {
            layers.push_back(layer->clone());
        }
        return *this;
    }

    // Конструктор перемещения
    Sequential(Sequential&& other) noexcept = default;
    
    // Оператор присваивания перемещением
    Sequential& operator=(Sequential&& other) noexcept = default;

    void add(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Tensor forward(const Tensor& input) {
        Tensor current_output = input;
        for (const auto& layer : layers) {
            current_output = layer->forward(current_output);
        }
        return current_output;
    }

    void backward(const Tensor& initial_gradient) {
        Tensor current_gradient = initial_gradient;
        for (int i = layers.size() - 1; i >= 0; --i) {
            current_gradient = layers[i]->backward(current_gradient);
        }
    }
};