#pragma once
#include "Tensor.h"

class Loss {
public:
    virtual ~Loss() = default;
    virtual float calculate(const Tensor& y_pred, const Tensor& y_true) = 0;
    virtual Tensor derivative(const Tensor& y_pred, const Tensor& y_true) = 0;
};

class MeanSquaredError : public Loss {
public:
    float calculate(const Tensor& y_pred, const Tensor& y_true) override {
        float sum = 0.0f;
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            float diff = y_pred.data[i] - y_true.data[i];
            sum += diff * diff;
        }
        return sum / y_pred.data.size();
    }

    Tensor derivative(const Tensor& y_pred, const Tensor& y_true) override {
        Tensor grad = y_pred;
        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            grad.data[i] = 2 * (y_pred.data[i] - y_true.data[i]) / y_pred.data.size();
        }
        return grad;
    }
};