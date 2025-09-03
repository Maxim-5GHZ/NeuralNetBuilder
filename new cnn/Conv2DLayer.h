#pragma once
#include "Layer.h"
#include <random>
class Conv2DLayer : public Layer {
public:
    Tensor kernels; 
    Tensor biases;  
    Tensor grad_kernels;
    Tensor grad_biases;
private:
    int in_channels, out_channels;
    int kernel_size, stride, padding;
    Tensor last_input;
public:
    Conv2DLayer(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0)
        : in_channels(input_channels), out_channels(output_channels), 
          kernel_size(kernel_size), stride(stride), padding(padding) {
        kernels = Tensor({out_channels, in_channels, kernel_size, kernel_size});
        biases = Tensor({1, out_channels}); 
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);
        for (auto& w : kernels.data) w = distribution(generator);
    }
    Tensor forward(const Tensor& input) override {
        last_input = input;
        int N = input.shape[0], C_in = input.shape[1], H_in = input.shape[2], W_in = input.shape[3];
        int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
        int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;
        Tensor output({N, out_channels, H_out, W_out});
        for (int n = 0; n < N; ++n) {
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float sum = biases.data[c_out];
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int h_in = h * stride + kh - padding;
                                    int w_in = w * stride + kw - padding;
                                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                        int input_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                                        int kernel_idx = c_out * C_in * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
                                        sum += input.data[input_idx] * kernels.data[kernel_idx];
                                    }
                                }
                            }
                        }
                        int output_idx = n * out_channels * H_out * W_out + c_out * H_out * W_out + h * W_out + w;
                        output.data[output_idx] = sum;
                    }
                }
            }
        }
        return output;
    }
    Tensor backward(const Tensor& output_gradient) override {
        grad_kernels = Tensor(kernels.shape); 
        grad_biases = Tensor(biases.shape);   
        return Tensor(last_input.shape);      
    }
};