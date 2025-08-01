#include <vector>
#include <memory>
#include "lay.h"
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
};