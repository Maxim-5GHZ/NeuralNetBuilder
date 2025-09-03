#pragma once
#include "Layer.h"

class FlattenLayer : public Layer {
private:
    std::vector<int> last_input_shape;

public:
    Tensor forward(const Tensor& input) override {
        last_input_shape = input.shape;
        int batch_size = input.shape[0];
        int features = 1;
        for (size_t i = 1; i < input.shape.size(); ++i) {
            features *= input.shape[i];
        }
        
        Tensor output({batch_size, features});
        output.data = input.data; // Данные уже лежат последовательно, просто меняем форму
        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        // Просто восстанавливаем исходную форму тензора
        Tensor input_gradient(last_input_shape);
        input_gradient.data = output_gradient.data;
        return input_gradient;
    }
};