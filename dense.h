#include "lay.h"
#include "activations.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdlib>

#pragma once

template<typename T>
class Dense : public Lay<T> {
    size_t m_inputSize = 0;
    size_t m_outputSize;
    std::function<T(T)> m_activation;
    std::vector<T> m_weights;
    std::vector<T> m_biases;

    void initializeWeights() {
        m_weights.resize(m_inputSize * m_outputSize);
        m_biases.resize(m_outputSize);
        
        T range = sqrt(6.0 / (m_inputSize + m_outputSize));
        for (size_t i = 0; i < m_weights.size(); ++i) {
            m_weights[i] = static_cast<T>(rand()) / RAND_MAX * 2 * range - range;
        }
    }

public:
    Dense(size_t outputSize, std::function<T(T)> activation = [](T x) { return x; })
        : m_outputSize(outputSize), m_activation(activation) {}

    std::vector<T> forward(const std::vector<T>& input) override {
        if (m_weights.empty()) {
            m_inputSize = input.size();
            initializeWeights();
        } else if (input.size() != m_inputSize) {
            throw std::runtime_error("Input size mismatch in Dense layer");
        }

        std::vector<T> output(m_outputSize);
        for (size_t j = 0; j < m_outputSize; ++j) {
            T sum = m_biases[j];
            for (size_t i = 0; i < m_inputSize; ++i) {
                sum += input[i] * m_weights[j * m_inputSize + i];
            }
            output[j] = m_activation(sum);
        }
        return output;
    }
};