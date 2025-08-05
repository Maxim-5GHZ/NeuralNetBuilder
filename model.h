#pragma once
#include <vector>
#include <memory>
#include "lay.h"
#include "dense.h"
#include "conv2d.h"
#include "maxpool.h"
#include "flatten.h"
#include <fstream>
#include <string>
#include <stdexcept>
#include <functional>
#include <unordered_map>

template<typename T>
std::unique_ptr<Lay<T>> create_layer(const std::string& type) {
    static const std::unordered_map<std::string, std::function<std::unique_ptr<Lay<T>>()>> creators = {
        {"Dense", []() { return std::make_unique<Dense<T>>(); }},
        {"Conv2D", []() { return std::make_unique<Conv2D<T>>(); }},
        {"MaxPool", []() { return std::make_unique<MaxPool<T>>(); }},
        {"Flatten", []() { return std::make_unique<Flatten<T>>(); }}
    };

    auto it = creators.find(type);
    if (it == creators.end()) {
        throw std::runtime_error("Unknown layer type: " + type);
    }
    return it->second();
}

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

    void save(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out) throw std::runtime_error("Cannot open file for writing");
        
        for (const auto& layer : m_layers) {
            out << layer->getType() << "\n";
            layer->save(out);
        }
    }

    void load(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) throw std::runtime_error("Cannot open file for reading");
        
        m_layers.clear();
        std::string layer_type;
        
        while (in >> layer_type) {
            auto layer = create_layer<T>(layer_type);
            
            try {
                layer->load(in);
            } catch (const std::exception& e) {
                throw std::runtime_error("Error loading layer '" + layer_type + "': " + e.what());
            }
            
            if (in.fail()) {
                throw std::runtime_error("File read error after layer: " + layer_type);
            }
            
            m_layers.push_back(std::move(layer));
        }
    }
};