#include <vector>
#include <memory>
#include "lay.h"
#include"dense.h"
#include"conv2d.h"
#include"maxpool.h"
#include"flatten.h"
#include <fstream>
#include <string>
#include <stdexcept>
#pragma once

template<typename T>
class Model {
    std::vector<std::unique_ptr<Lay<T>>> m_layers;

public:
    void add(std::unique_ptr<Lay<T>> layer) {
        m_layers.push_back(std::move(layer));
    }

    std::vector<T> forward(const std::vector<T>& input) {
        std::vector<T> result = input;
        for (auto& layer : m_layers) {
            result = layer->forward(result);
        }
        return result;
    }

    std::vector<T> backward(const std::vector<T>& output_gradient) {
        std::vector<T> grad = output_gradient;
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }

    void update_weights(T learning_rate) {
        for (auto& layer : m_layers) {
            layer->update_weights(learning_rate);
        }
    }

    // Сохранение модели в файл
    void save(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out) throw std::runtime_error("Cannot open file for writing");
        
        for (const auto& layer : m_layers) {
            out << layer->getType() << "\n";
            layer->save(out);
        }
    }

    // Загрузка модели из файла
    void load(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) throw std::runtime_error("Cannot open file for reading");
        
        m_layers.clear();
        std::string layer_type;
        
        while (in >> layer_type) {
            std::unique_ptr<Lay<T>> layer;
            
            if (layer_type == "Dense") {
                layer = std::make_unique<Dense<T>>();
            } else if (layer_type == "Conv2D") {
                layer = std::make_unique<Conv2D<T>>();
            } else if (layer_type == "MaxPool") {
                layer = std::make_unique<MaxPool<T>>();
            } else if (layer_type == "Flatten") {
                layer = std::make_unique<Flatten<T>>();
            } else {
                throw std::runtime_error("Unknown layer type: " + layer_type);
            }
            
            layer->load(in);
            m_layers.push_back(std::move(layer));
        }
    }
};