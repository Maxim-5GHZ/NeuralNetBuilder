#pragma once
#include "Layer.h"
#include <vector>
#include <algorithm>
#include <limits>
class MaxPooling2DLayer : public Layer {
private:
    int pool_size;
    int stride;
    Tensor last_input;
    std::vector<int> max_indices; 
public:
    MaxPooling2DLayer(int pool_size, int stride = -1) : pool_size(pool_size) {
        this->stride = (stride == -1) ? pool_size : stride;
    }
    Tensor forward(const Tensor& input) override {
        last_input = input;
        int N = input.shape[0], C = input.shape[1], H_in = input.shape[2], W_in = input.shape[3];
        int H_out = (H_in - pool_size) / stride + 1;
        int W_out = (W_in - pool_size) / stride + 1;
        Tensor output({N, C, H_out, W_out});
        max_indices.resize(output.data.size());
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int max_idx = -1;
                        for (int ph = 0; ph < pool_size; ++ph) {
                            for (int pw = 0; pw < pool_size; ++pw) {
                                int h_in = h * stride + ph;
                                int w_in = w * stride + pw;
                                int input_idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                                if (input.data[input_idx] > max_val) {
                                    max_val = input.data[input_idx];
                                    max_idx = input_idx;
                                }
                            }
                        }
                        int output_idx = n * C * H_out * W_out + c * H_out * W_out + h * W_out + w;
                        output.data[output_idx] = max_val;
                        max_indices[output_idx] = max_idx;
                    }
                }
            }
        }
        return output;
    }
    Tensor backward(const Tensor& output_gradient) override {
        Tensor input_gradient(last_input.shape); 
        int N = last_input.shape[0], C = last_input.shape[1];
        int H_out = (last_input.shape[2] - pool_size) / stride + 1;
        int W_out = (last_input.shape[3] - pool_size) / stride + 1;
        for (int i = 0; i < max_indices.size(); ++i) {
            int input_idx = max_indices[i];
            input_gradient.data[input_idx] += output_gradient.data[i];
        }
        return input_gradient;
    }
};