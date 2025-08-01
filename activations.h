#pragma once
#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>

template<typename T>
struct Activations {
   
    static T linear(T x) { return x; }

  
    static T relu(T x) { return x > 0 ? x : 0; }

    static T leakyRelu(T x, T alpha = 0.01) { return x > 0 ? x : alpha * x; }


    static T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

    static T tanh(T x) { return std::tanh(x); }

    static std::vector<T> softmax(const std::vector<T>& x) {
        std::vector<T> result(x.size());
        T max_val = *std::max_element(x.begin(), x.end());
        T sum = 0;
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }
        
        for (auto& val : result) {
            val /= sum;
        }
        return result;
    }
};