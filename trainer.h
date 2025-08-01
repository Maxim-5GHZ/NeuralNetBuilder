#pragma once
#include "model.h"

template<typename T>
class BackwardTrainer {
    Model<T>& model;
    T learning_rate;

public:
    BackwardTrainer(Model<T>& model, T lr) : model(model), learning_rate(lr) {}

    void train_step(const std::vector<T>& input, 
                   const std::vector<T>& target,
                   std::function<T(const std::vector<T>&, const std::vector<T>&)> loss_func,
                   std::function<std::vector<T>(const std::vector<T>&, const std::vector<T>&)> loss_deriv) {
        

        auto output = model.forward(input);
        
        auto output_gradient = loss_deriv(output, target);
    
        model.backward(output_gradient);
        
       
        model.update_weights(learning_rate);
    }
};