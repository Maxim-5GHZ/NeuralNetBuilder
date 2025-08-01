#pragma once
#include "lay.h"
#include <vector>
#include <stdexcept>
#include <limits>

template<typename T>
class MaxPool : public Lay<T> {
private:
    size_t m_input_height;
    size_t m_input_width;
    size_t m_channels;
    size_t m_pool_size;
    size_t m_output_height;
    size_t m_output_width;
    std::vector<size_t> m_max_indices;

public:
    MaxPool(size_t input_height, size_t input_width, size_t channels, size_t pool_size)
        : m_input_height(input_height), m_input_width(input_width), 
          m_channels(channels), m_pool_size(pool_size),
          m_output_height(input_height / pool_size),
          m_output_width(input_width / pool_size) 
    {
        if (input_height % pool_size != 0 || input_width % pool_size != 0) {
            throw std::runtime_error("Input dimensions must be divisible by pool_size");
        }
    }

    std::vector<T> forward(const std::vector<T>& input) override {
        if (input.size() != m_input_height * m_input_width * m_channels) {
            throw std::runtime_error("MaxPool: input size does not match expected volume");
        }

        size_t output_size = m_output_height * m_output_width * m_channels;
        std::vector<T> output(output_size, T(0));
        m_max_indices.resize(output_size);

        for (size_t c = 0; c < m_channels; ++c) {
            for (size_t i = 0; i < m_output_height; ++i) {
                for (size_t j = 0; j < m_output_width; ++j) {
                    T max_val = std::numeric_limits<T>::lowest();
                    size_t max_index_in_region = 0;
                    size_t start_h = i * m_pool_size;
                    size_t start_w = j * m_pool_size;

                    for (size_t h = 0; h < m_pool_size; ++h) {
                        for (size_t w = 0; w < m_pool_size; ++w) {
                            size_t input_index = c * (m_input_height * m_input_width) 
                                              + (start_h + h) * m_input_width 
                                              + (start_w + w);
                            T val = input[input_index];
                            if (val > max_val) {
                                max_val = val;
                                max_index_in_region = h * m_pool_size + w;
                            }
                        }
                    }

                    size_t out_index = c * (m_output_height * m_output_width) 
                                    + i * m_output_width + j;
                    output[out_index] = max_val;
                    m_max_indices[out_index] = max_index_in_region;
                }
            }
        }

        return output;
    }

    std::vector<T> backward(const std::vector<T>& output_gradient) override {
        if (output_gradient.size() != m_output_height * m_output_width * m_channels) {
            throw std::runtime_error("MaxPool: output gradient size does not match");
        }

        std::vector<T> input_gradient(m_input_height * m_input_width * m_channels, T(0));

        for (size_t c = 0; c < m_channels; ++c) {
            for (size_t i = 0; i < m_output_height; ++i) {
                for (size_t j = 0; j < m_output_width; ++j) {
                    size_t out_index = c * (m_output_height * m_output_width) 
                                    + i * m_output_width + j;
                    T grad_val = output_gradient[out_index];
                    size_t max_index_in_region = m_max_indices[out_index];
                    size_t h_offset = max_index_in_region / m_pool_size;
                    size_t w_offset = max_index_in_region % m_pool_size;
                    size_t h = i * m_pool_size + h_offset;
                    size_t w = j * m_pool_size + w_offset;
                    size_t input_index = c * (m_input_height * m_input_width) 
                                      + h * m_input_width + w;
                    input_gradient[input_index] += grad_val;
                }
            }
        }

        return input_gradient;
    }

    void update_weights(T learning_rate) override {
        // Нет обучаемых параметров
    }
};
