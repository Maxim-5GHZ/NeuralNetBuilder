#include "lay.h"
#include "activations.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <iostream>

#pragma once

template<typename T>
class Dense : public Lay<T> {
    size_t m_inputSize = 0;
    size_t m_outputSize;
    std::function<T(T)> m_activation;
    std::function<T(T)> m_activation_deriv;
    std::vector<T> m_weights;
    std::vector<T> m_biases;
    std::vector<T> m_last_input;
    std::vector<T> m_last_preactivation;
    std::vector<T> m_dweights;
    std::vector<T> m_dbiases;

    void initializeWeights() {
        m_weights.resize(m_inputSize * m_outputSize);
        m_biases.resize(m_outputSize);
        m_dweights.resize(m_inputSize * m_outputSize);
        m_dbiases.resize(m_outputSize);
        
        T range = sqrt(6.0 / (m_inputSize + m_outputSize));
        for (size_t i = 0; i < m_weights.size(); ++i) {
            m_weights[i] = static_cast<T>(rand()) / RAND_MAX * 2 * range - range;
        }
    }

public:
    Dense(size_t outputSize, 
          std::function<T(T)> activation = [](T x) { return x; },
          std::function<T(T)> activation_deriv = [](T) { return 1; })
        : m_outputSize(outputSize), 
          m_activation(activation),
          m_activation_deriv(activation_deriv) {}

    std::vector<T> forward(const std::vector<T>& input) override {
        if (m_weights.empty()) {
            m_inputSize = input.size();
            initializeWeights();
        } else if (input.size() != m_inputSize) {
            throw std::runtime_error("Input size mismatch in Dense layer");
        }

        m_last_input = input;
        std::vector<T> output(m_outputSize);
        m_last_preactivation.resize(m_outputSize);

        for (size_t j = 0; j < m_outputSize; ++j) {
            T sum = m_biases[j];
            for (size_t i = 0; i < m_inputSize; ++i) {
                sum += input[i] * m_weights[j * m_inputSize + i];
            }
            m_last_preactivation[j] = sum;
            output[j] = m_activation(sum);
        }
        return output;
    }

    std::vector<T> backward(const std::vector<T>& output_gradient) override {
        std::vector<T> input_gradient(m_inputSize, 0);
        std::vector<T> preact_gradient(m_outputSize);

       
        for (size_t j = 0; j < m_outputSize; ++j) {
            preact_gradient[j] = output_gradient[j] * m_activation_deriv(m_last_preactivation[j]);
        }

     
        for (size_t j = 0; j < m_outputSize; ++j) {
            for (size_t i = 0; i < m_inputSize; ++i) {
                size_t index = j * m_inputSize + i;
                m_dweights[index] += preact_gradient[j] * m_last_input[i];
                input_gradient[i] += m_weights[index] * preact_gradient[j];
            }
            m_dbiases[j] += preact_gradient[j];
        }

        return input_gradient;
    }

    void update_weights(T learning_rate) override {
        for (size_t i = 0; i < m_weights.size(); ++i) {
            m_weights[i] -= learning_rate * m_dweights[i];
            m_dweights[i] = 0;
        }
        for (size_t i = 0; i < m_biases.size(); ++i) {
            m_biases[i] -= learning_rate * m_dbiases[i];
            m_dbiases[i] = 0;
        }
    }
};