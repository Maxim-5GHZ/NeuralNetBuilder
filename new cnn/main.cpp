#include "Tensor.h"
#include "Sequential.h"
#include "DenseLayer.h"
#include "ReLULayer.h"
#include "Loss.h"
#include "GeneticAlgorithmOptimizer.h" 
int main() {
    Tensor X_train({4, 2});
    X_train.data = {0,0, 0,1, 1,0, 1,1};
    Tensor y_train({4, 1});
    y_train.data = {0, 1, 1, 0};
    Sequential model_template;
    model_template.add(std::make_unique<DenseLayer>(2, 8));
    model_template.add(std::make_unique<ReLULayer>());
    model_template.add(std::make_unique<DenseLayer>(8, 1)); 
    MeanSquaredError loss_fn;
    GeneticAlgorithmOptimizer ga_optimizer(100, 0.05f, 0.1f, 2);
    Sequential best_model = ga_optimizer.train(model_template, X_train, y_train, loss_fn, 200);
    std::cout << "\n--- Testing Best Model from GA ---\n";
    Tensor test_output = best_model.forward(X_train);
    std::cout << "Predictions:" << std::endl;
    test_output.print();
    std::cout << "True values:" << std::endl;
    y_train.print();
    return 0;
}