#include "Tensor.h"
#include "Sequential.h"
#include "DenseLayer.h"
#include "ReLULayer.h"
#include "Loss.h"
// #include "Optimizer.h" // SGD больше не нужен
#include "GeneticAlgorithmOptimizer.h" // Подключаем наш ГА

int main() {
    
    Tensor X_train({4, 2});
    X_train.data = {0,0, 0,1, 1,0, 1,1};

    Tensor y_train({4, 1});
    y_train.data = {0, 1, 1, 0};

    // Создаем ШАБЛОН модели. ГА создаст из него популяцию.
    Sequential model_template;
    model_template.add(std::make_unique<DenseLayer>(2, 8));
    model_template.add(std::make_unique<ReLULayer>());
    model_template.add(std::make_unique<DenseLayer>(8, 1)); 
    
    MeanSquaredError loss_fn;
    
    // --- Обучение с помощью Генетического Алгоритма ---
    
    // 1. Создаем и настраиваем оптимизатор
    // GeneticAlgorithmOptimizer(размер популяции, шанс мутации, сила мутации, кол-во элиты)
    GeneticAlgorithmOptimizer ga_optimizer(100, 0.05f, 0.1f, 2);

    // 2. Запускаем "эволюцию"
    // Метод train возвращает лучшую модель, найденную за все поколения
    Sequential best_model = ga_optimizer.train(model_template, X_train, y_train, loss_fn, 200);

    /* --- Старый код с SGD для сравнения ---
    SGD optimizer(0.1f); 
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Tensor y_pred = model_template.forward(X_train);
        float current_loss = loss_fn.calculate(y_pred, y_train);
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << current_loss << std::endl;
        }
        Tensor loss_gradient = loss_fn.derivative(y_pred, y_train);
        model_template.backward(loss_gradient);
        optimizer.step(model_template);
    }
    */
    
    
    // --- Тестирование лучшей модели, найденной ГА ---
    std::cout << "\n--- Testing Best Model from GA ---\n";
    Tensor test_output = best_model.forward(X_train);
    
    std::cout << "Predictions:" << std::endl;
    test_output.print();
    
    std::cout << "True values:" << std::endl;
    y_train.print();

    return 0;
}