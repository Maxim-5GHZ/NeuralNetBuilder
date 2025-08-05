#pragma once
#include "lay.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>

template<typename T>
class Conv2D : public Lay<T> {
private:
    size_t m_input_channels;
    size_t m_kernel_size;
    size_t m_output_channels;
    size_t m_stride;
    size_t m_padding;
    
    size_t m_input_height;
    size_t m_input_width;
    size_t m_output_height;
    size_t m_output_width;
    
    std::vector<T> m_weights;
    std::vector<T> m_biases;
    std::vector<T> m_padded_input;
    std::vector<T> m_dweights;
    std::vector<T> m_dbiases;
    
    void initialize_weights() {
        size_t fan_in = m_input_channels * m_kernel_size * m_kernel_size;
        size_t fan_out = m_output_channels * m_kernel_size * m_kernel_size;
        T range = std::sqrt(6.0 / (fan_in + fan_out));
        
        m_weights.resize(m_output_channels * m_input_channels * m_kernel_size * m_kernel_size);
        for (size_t i = 0; i < m_weights.size(); ++i) {
            m_weights[i] = static_cast<T>(rand()) / RAND_MAX * 2 * range - range;
        }
        
        m_biases.resize(m_output_channels, 0);
        m_dweights.resize(m_weights.size(), 0);
        m_dbiases.resize(m_output_channels, 0);
    }
    
    void calculate_output_dimensions() {
        m_output_height = (m_input_height + 2 * m_padding - m_kernel_size) / m_stride + 1;
        m_output_width = (m_input_width + 2 * m_padding - m_kernel_size) / m_stride + 1;
        
        if (m_output_height <= 0 || m_output_width <= 0) {
            throw std::runtime_error("Invalid convolution parameters: output dimensions <= 0");
        }
    }
    
    void apply_padding(const std::vector<T>& input, std::vector<T>& padded_input) {
        size_t padded_height = m_input_height + 2 * m_padding;
        size_t padded_width = m_input_width + 2 * m_padding;
        padded_input.assign(m_input_channels * padded_height * padded_width, 0);
        
        for (size_t c = 0; c < m_input_channels; ++c) {
            for (size_t h = 0; h < m_input_height; ++h) {
                for (size_t w = 0; w < m_input_width; ++w) {
                    size_t input_idx = c * m_input_height * m_input_width + h * m_input_width + w;
                    size_t padded_idx = c * padded_height * padded_width + 
                                      (h + m_padding) * padded_width + 
                                      (w + m_padding);
                    padded_input[padded_idx] = input[input_idx];
                }
            }
        }
    }

public:
    Conv2D(size_t input_height, size_t input_width, size_t input_channels,
           size_t kernel_size, size_t output_channels,
           size_t stride = 1, size_t padding = 0)
        : m_input_height(input_height), m_input_width(input_width),
          m_input_channels(input_channels), m_kernel_size(kernel_size),
          m_output_channels(output_channels), m_stride(stride), m_padding(padding) {
        
        calculate_output_dimensions();
        initialize_weights();
    }

    Conv2D() = default; // Для загрузки

    std::string getType() const override { return "Conv2D"; }

    void save(std::ostream& out) const override {
        out << m_input_height << " " << m_input_width << " "
            << m_input_channels << " " << m_kernel_size << " "
            << m_output_channels << " " << m_stride << " " << m_padding << "\n";
            
        for (const auto& w : m_weights) out << w << " ";
        out << "\n";
        
        for (const auto& b : m_biases) out << b << " ";
        out << "\n";
    }

    void load(std::istream& in) override {
        in >> m_input_height >> m_input_width
           >> m_input_channels >> m_kernel_size
           >> m_output_channels >> m_stride >> m_padding;
        
        calculate_output_dimensions();
        
        size_t weights_size = m_output_channels * m_input_channels * 
                              m_kernel_size * m_kernel_size;
        m_weights.resize(weights_size);
        for (size_t i = 0; i < weights_size; ++i) in >> m_weights[i];
        
        m_biases.resize(m_output_channels);
        for (size_t i = 0; i < m_output_channels; ++i) in >> m_biases[i];
        
        m_dweights.resize(weights_size, 0);
        m_dbiases.resize(m_output_channels, 0);
    }

    std::vector<T> forward(const std::vector<T>& input) override {
        if (input.size() != m_input_height * m_input_width * m_input_channels) {
            throw std::runtime_error("Conv2D: input size mismatch");
        }
        
        apply_padding(input, m_padded_input);
        
        size_t padded_height = m_input_height + 2 * m_padding;
        size_t padded_width = m_input_width + 2 * m_padding;
        std::vector<T> output(m_output_height * m_output_width * m_output_channels, 0);
        
        for (size_t k = 0; k < m_output_channels; ++k) {
            for (size_t h = 0; h < m_output_height; ++h) {
                for (size_t w = 0; w < m_output_width; ++w) {
                    T sum = 0;
                    
                    for (size_t c = 0; c < m_input_channels; ++c) {
                        for (size_t kh = 0; kh < m_kernel_size; ++kh) {
                            for (size_t kw = 0; kw < m_kernel_size; ++kw) {
                                size_t h_in = h * m_stride + kh;
                                size_t w_in = w * m_stride + kw;
                                
                                size_t input_idx = c * padded_height * padded_width + 
                                                 h_in * padded_width + w_in;
                                size_t weight_idx = k * m_input_channels * m_kernel_size * m_kernel_size +
                                                  c * m_kernel_size * m_kernel_size +
                                                  kh * m_kernel_size + kw;
                                
                                sum += m_padded_input[input_idx] * m_weights[weight_idx];
                            }
                        }
                    }
                    
                    sum += m_biases[k];
                    size_t output_idx = k * m_output_height * m_output_width + 
                                      h * m_output_width + w;
                    output[output_idx] = sum;
                }
            }
        }
        
        return output;
    }

    std::vector<T> backward(const std::vector<T>& output_gradient) override {
        if (output_gradient.size() != m_output_height * m_output_width * m_output_channels) {
            throw std::runtime_error("Conv2D: output gradient size mismatch");
        }
        
        size_t padded_height = m_input_height + 2 * m_padding;
        size_t padded_width = m_input_width + 2 * m_padding;
        std::vector<T> padded_input_grad(m_input_channels * padded_height * padded_width, 0);
        
        std::fill(m_dweights.begin(), m_dweights.end(), 0);
        std::fill(m_dbiases.begin(), m_dbiases.end(), 0);
        

        for (size_t k = 0; k < m_output_channels; ++k) {
            for (size_t h = 0; h < m_output_height; ++h) {
                for (size_t w = 0; w < m_output_width; ++w) {
                    size_t out_idx = k * m_output_height * m_output_width + 
                                   h * m_output_width + w;
                    T grad = output_gradient[out_idx];
                    
                    m_dbiases[k] += grad;
                    
                    for (size_t c = 0; c < m_input_channels; ++c) {
                        for (size_t kh = 0; kh < m_kernel_size; ++kh) {
                            for (size_t kw = 0; kw < m_kernel_size; ++kw) {
                                size_t h_in = h * m_stride + kh;
                                size_t w_in = w * m_stride + kw;
                                
                                size_t input_idx = c * padded_height * padded_width + 
                                                h_in * padded_width + w_in;
                                size_t weight_idx = k * m_input_channels * m_kernel_size * m_kernel_size +
                                                  c * m_kernel_size * m_kernel_size +
                                                  kh * m_kernel_size + kw;
                                
                                m_dweights[weight_idx] += m_padded_input[input_idx] * grad;
                                padded_input_grad[input_idx] += m_weights[weight_idx] * grad;
                            }
                        }
                    }
                }
            }
        }
        
        std::vector<T> input_grad(m_input_channels * m_input_height * m_input_width, 0);
        for (size_t c = 0; c < m_input_channels; ++c) {
            for (size_t h = 0; h < m_input_height; ++h) {
                for (size_t w = 0; w < m_input_width; ++w) {
                    size_t input_idx = c * m_input_height * m_input_width + 
                                     h * m_input_width + w;
                    size_t padded_idx = c * padded_height * padded_width + 
                                     (h + m_padding) * padded_width + 
                                     (w + m_padding);
                    input_grad[input_idx] = padded_input_grad[padded_idx];
                }
            }
        }
        
        return input_grad;
    }

    void update_weights(T learning_rate) override {
        for (size_t i = 0; i < m_weights.size(); ++i) {
            m_weights[i] -= learning_rate * m_dweights[i];
        }
        
        for (size_t i = 0; i < m_biases.size(); ++i) {
            m_biases[i] -= learning_rate * m_dbiases[i];
        }
    }
};