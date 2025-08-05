#pragma once
#include "lay.h"
#include "activations.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <map>
#include <cmath>

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
    std::string m_activation_name = "linear";

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
  
    void set_activation(const std::string& name) {
        m_activation_name = name;
        if (name == "sigmoid") {
            m_activation = Activations<T>::sigmoid;
            m_activation_deriv = Activations<T>::sigmoid_deriv;
        } else if (name == "relu") {
            m_activation = Activations<T>::relu;
            m_activation_deriv = Activations<T>::relu_deriv;
        } else if (name == "leakyRelu") {
            m_activation = Activations<T>::leakyRelu;
            m_activation_deriv = Activations<T>::leakyRelu_deriv;
        } else if (name == "tanh") {
            m_activation = Activations<T>::tanh;
            m_activation_deriv = Activations<T>::tanh_deriv;
        } else {
            m_activation = [](T x) { return x; };
            m_activation_deriv = [](T) { return 1; };
        }
    }

    Dense(size_t outputSize, const std::string& activation_name = "linear")
        : m_outputSize(outputSize) {
        set_activation(activation_name);
    }

    Dense() = default;

    std::string getType() const override { return "Dense"; }

    void save(std::ostream& out) const override {
        out << m_inputSize << " " << m_outputSize << "\n";
        out << m_activation_name << "\n";
        for (const auto& w : m_weights) out << w << " ";
        out << "\n";
        for (const auto& b : m_biases) out << b << " ";
        out << "\n";
    }

    void load(std::istream& in) override {
        in >> m_inputSize >> m_outputSize;
        in >> m_activation_name;
        set_activation(m_activation_name);  
        
        m_weights.resize(m_inputSize * m_outputSize);
        for (size_t i = 0; i < m_weights.size(); ++i) in >> m_weights[i];
        
        m_biases.resize(m_outputSize);
        for (size_t i = 0; i < m_biases.size(); ++i) in >> m_biases[i];
        
        m_dweights.resize(m_weights.size(), 0);
        m_dbiases.resize(m_biases.size(), 0);
    }

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