#pragma once
#include "Sequential.h"
#include "DenseLayer.h"
#include "Loss.h"
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

// Вспомогательная структура для хранения модели и её "приспособленности"
struct Individual {
    Sequential model;
    float fitness = 0.0f;

    // Сравнение для сортировки
    bool operator<(const Individual& other) const {
        return fitness > other.fitness; // Сортируем по убыванию приспособленности
    }
};

class GeneticAlgorithmOptimizer {
public:
    int population_size;
    float mutation_rate;
    float mutation_strength;
    int elitism_count; // Количество "элитных" особей, переходящих в следующее поколение без изменений

    std::vector<Individual> population;

private:
    // Генератор случайных чисел для мутаций
    std::default_random_engine generator;

public:
    GeneticAlgorithmOptimizer(int pop_size = 50, float mut_rate = 0.05f, float mut_strength = 0.1f, int elite_count = 2)
        : population_size(pop_size), mutation_rate(mut_rate), mutation_strength(mut_strength), elitism_count(elite_count) {
        std::random_device rd;
        generator.seed(rd());
    }

    // Главный метод обучения
    Sequential train(const Sequential& model_template, const Tensor& X_train, const Tensor& y_train, Loss& loss_fn, int generations) {
        // 1. Инициализация популяции
        initialize_population(model_template);

        for (int gen = 0; gen < generations; ++gen) {
            // 2. Оценка приспособленности каждой особи
            calculate_fitness(X_train, y_train, loss_fn);

            // 3. Сортировка популяции от лучших к худшим
            std::sort(population.begin(), population.end());

            if (gen % 10 == 0) {
                std::cout << "Generation " << gen << ", Best Fitness: " << population[0].fitness 
                          << ", Loss: " << (1.0f / population[0].fitness) << std::endl;
            }

            // 4. Создание нового поколения
            evolve_new_generation();
        }
        
        // Пересчитываем и сортируем в последний раз, чтобы вернуть лучшую модель
        calculate_fitness(X_train, y_train, loss_fn);
        std::sort(population.begin(), population.end());
        
        std::cout << "\n--- Training Finished ---" << std::endl;
        std::cout << "Final Best Loss: " << (1.0f / population[0].fitness) << std::endl;

        return population[0].model;
    }

private:
    void initialize_population(const Sequential& model_template) {
        population.clear();
        for (int i = 0; i < population_size; ++i) {
            // Каждый раз создается новая модель со случайными весами
            Individual individual;
            individual.model = Sequential(model_template);
            population.push_back(individual);
        }
    }

    void calculate_fitness(const Tensor& X, const Tensor& y, Loss& loss_fn) {
        for (auto& individual : population) {
            Tensor y_pred = individual.model.forward(X);
            float loss = loss_fn.calculate(y_pred, y);
            // Приспособленность обратно пропорциональна ошибке. Добавляем эпсилон для стабильности.
            individual.fitness = 1.0f / (loss + 1e-6f);
        }
    }

    void evolve_new_generation() {
        std::vector<Individual> next_generation;
        next_generation.reserve(population_size);

        // 1. Элитизм: лучшие особи переходят в новое поколение
        for (int i = 0; i < elitism_count; ++i) {
            next_generation.push_back(population[i]);
        }

        // 2. Заполняем оставшуюся часть популяции потомками
        while (next_generation.size() < static_cast<size_t>(population_size)) {
            // Выбираем двух родителей из лучшей половины популяции
            Individual& parent1 = select_parent();
            Individual& parent2 = select_parent();

            // Создаем потомка
            Individual offspring = crossover(parent1, parent2);
            mutate(offspring);
            
            next_generation.push_back(offspring);
        }
        population = std::move(next_generation);
    }
    
    // Простой выбор родителя: случайная особь из лучших 50%
    Individual& select_parent() {
        int index = std::uniform_int_distribution<int>(0, population_size / 2 - 1)(generator);
        return population[index];
    }

    Individual crossover(const Individual& parent1, const Individual& parent2) {
        Individual child;
        child.model = Sequential(parent1.model); // Начинаем с копии первого родителя
        
        // Проходим по всем слоям и "скрещиваем" веса
        for (size_t i = 0; i < child.model.layers.size(); ++i) {
            DenseLayer* child_dense = dynamic_cast<DenseLayer*>(child.model.layers[i].get());
            if (child_dense) {
                const DenseLayer* p1_dense = dynamic_cast<const DenseLayer*>(parent1.model.layers[i].get());
                const DenseLayer* p2_dense = dynamic_cast<const DenseLayer*>(parent2.model.layers[i].get());

                // Скрещиваем веса
                for(size_t j = 0; j < child_dense->weights.data.size(); ++j) {
                    if (std::uniform_real_distribution<float>(0, 1)(generator) < 0.5f) {
                        child_dense->weights.data[j] = p1_dense->weights.data[j];
                    } else {
                        child_dense->weights.data[j] = p2_dense->weights.data[j];
                    }
                }
                // Скрещиваем смещения
                for(size_t j = 0; j < child_dense->bias.data.size(); ++j) {
                     if (std::uniform_real_distribution<float>(0, 1)(generator) < 0.5f) {
                        child_dense->bias.data[j] = p1_dense->bias.data[j];
                    } else {
                        child_dense->bias.data[j] = p2_dense->bias.data[j];
                    }
                }
            }
        }
        return child;
    }

    void mutate(Individual& individual) {
        std::uniform_real_distribution<float> dist(-mutation_strength, mutation_strength);
        
        for (auto& layer : individual.model.layers) {
            DenseLayer* dense_layer = dynamic_cast<DenseLayer*>(layer.get());
            if (dense_layer) {
                // Мутация весов
                for (auto& weight : dense_layer->weights.data) {
                    if (std::uniform_real_distribution<float>(0, 1)(generator) < mutation_rate) {
                        weight += dist(generator);
                    }
                }
                // Мутация смещений
                for (auto& b : dense_layer->bias.data) {
                    if (std::uniform_real_distribution<float>(0, 1)(generator) < mutation_rate) {
                        b += dist(generator);
                    }
                }
            }
        }
    }
};