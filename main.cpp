#include "activations.h"
#include <iostream>
#include <memory>
#include <vector>
#include "model.h"
#include "dense.h"
#include "trainer.h"
#include"maxpull.h"
#include"flutten.h"
#include"conv2d.h"

using namespace std;
using T = float;


const vector<pair<vector<T>, vector<T>>> train_data = {
    {{0, 0}, {0}},
    {{0, 1}, {1}},
    {{1, 0}, {1}},
    {{1, 1}, {0}}
};


T mse_loss(const vector<T>& pred, const vector<T>& target) {
    T loss = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
        T diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return loss / pred.size();
}

vector<T> mse_loss_deriv(const vector<T>& pred, const vector<T>& target) {
    vector<T> deriv(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        deriv[i] = 2 * (pred[i] - target[i]) / pred.size();
    }
    return deriv;
}

int main() {

    Model<T> model;
    model.add(make_unique<Dense<T>>(2, Activations<T>::relu, Activations<T>::relu_deriv));       
    model.add(make_unique<Dense<T>>(1, Activations<T>::relu, Activations<T>::relu_deriv));

    BackwardTrainer<T> trainer(model, 0.001);

    
    const T target_mse = 0.00001;
    const int max_epochs = 100000; 

    T epoch_loss = 1;
    for (int epoch = 0; epoch_loss > 0.000005; ++epoch) {
        
        epoch_loss = 0;
        for (const auto& [input, target] : train_data) {
            trainer.train_step(input, target, mse_loss, mse_loss_deriv);
            auto output = model.forward(input);
            epoch_loss += mse_loss(output, target);
        }
        
        epoch_loss /= train_data.size();
        
        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << ", MSE: " << epoch_loss << endl;
        }
        
      
        if (epoch_loss <= target_mse) {
            cout << "Target MSE reached at epoch " << epoch << " (MSE: " << epoch_loss << ")\n";
            break;
        }
    }
  
 
    cout << "\nTesting XOR network:\n";
    cout << fixed;
    cout.precision(4);
    
    for (const auto& [input, target] : train_data) {
        auto output = model.forward(input);
        cout << input[0] << " XOR " << input[1] << " = " << output[0] 
             << " (expected: " << target[0] << ")" << endl;
    }

    return 0;
}