#pragma once
#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <stdexcept>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<int> strides; // Добавлено для N-мерного доступа

    Tensor() = default;
    Tensor(const std::vector<int>& s) : shape(s) {
        int total_size = 1;
        for (int dim : s) {
            total_size *= dim;
        }
        data.resize(total_size, 0.0f);
        calculate_strides();
    }

private:
    // Вспомогательный метод для вычисления страйдов
    void calculate_strides() {
        strides.resize(shape.size());
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

public:
    // Универсальный метод для получения индекса в 1D-массиве
    int getIndex(const std::vector<int>& indices) const {
        assert(indices.size() == shape.size());
        int index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides[i];
        }
        return index;
    }

    // Универсальный доступ к элементу по N-мерным индексам
    float& at(const std::vector<int>& indices) {
        return data[getIndex(indices)];
    }
    const float& at(const std::vector<int>& indices) const {
        return data[getIndex(indices)];
    }

    // --- Удобные обертки для самых частых случаев ---

    // 2D доступ (как был)
    float& at(int row, int col) {
        assert(shape.size() == 2);
        return data[row * strides[0] + col * strides[1]];
    }
    const float& at(int row, int col) const {
        assert(shape.size() == 2);
        return data[row * strides[0] + col * strides[1]];
    }

    // 4D доступ (для CNN)
    float& at(int n, int c, int h, int w) {
        assert(shape.size() == 4);
        return data[n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
    }
    const float& at(int n, int c, int h, int w) const {
        assert(shape.size() == 4);
        return data[n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
    }
    
    // Статический метод для матричного умножения (без изменений)
    static Tensor dot(const Tensor& a, const Tensor& b) {
        assert(a.shape.size() == 2 && b.shape.size() == 2);
        assert(a.shape[1] == b.shape[0]);

        Tensor result({a.shape[0], b.shape[1]});
        for (int i = 0; i < a.shape[0]; ++i) {
            for (int j = 0; j < b.shape[1]; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < a.shape[1]; ++k) {
                    sum += a.at(i, k) * b.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        return result;
    }

    // Улучшенный метод вывода в консоль
    void print() const {
        std::cout << "Tensor Shape: [";
        for(size_t i=0; i<shape.size(); ++i) std::cout << shape[i] << (i == shape.size()-1 ? "" : ", ");
        std::cout << "]" << std::endl;
        
        if (shape.size() == 1) {
            for (int i = 0; i < shape[0]; ++i) std::cout << data[i] << " ";
            std::cout << std::endl;
        } else if (shape.size() == 2) {
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; ++j) {
                    std::cout << at(i, j) << " ";
                }
                std::cout << std::endl;
            }
        } else if (shape.size() == 4) {
             for (int n = 0; n < shape[0]; ++n) {
                std::cout << "Batch " << n << ":" << std::endl;
                for (int c = 0; c < shape[1]; ++c) {
                    std::cout << " Channel " << c << ":" << std::endl;
                    for (int h = 0; h < shape[2]; ++h) {
                        for (int w = 0; w < shape[3]; ++w) {
                            std::cout << at(n, c, h, w) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                std::cout << "---" << std::endl;
            }
        } else { // Для других размерностей - просто выводим все данные
            for(const auto& val : data) std::cout << val << " ";
            std::cout << std::endl;
        }
    }
};