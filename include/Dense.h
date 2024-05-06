#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../lib/Eigen/Core"

using namespace std;
using namespace Eigen;

typedef MatrixXd mat;

// class Dense: public Layer
class DenseLayer
{
private:
    mat weights;
    mat bias;
    mat input;
    mat output;
    int input_size;
    int output_size;

public:
    DenseLayer() {}

    // assigning random values b/w -1 & 1 at the initialization
    DenseLayer(int in_size, int out_size) : input_size(in_size), output_size(out_size)
    {
        srand((unsigned int)time(0));
        weights = mat::Random(output_size, input_size);
        bias = mat::Random(output_size, 1);
    }

    mat forward(const mat &input)
    {
        this->input = input;
        mat z = (weights * input) + bias;
        return z;
    }

    mat backward(const mat &output_gradient, const double learning_rate)
    {
        mat weights_gradient = (output_gradient * input.transpose());

        this->weights = this->weights - (learning_rate * weights_gradient);
        this->bias = this->bias - (learning_rate * output_gradient);

        return (weights.transpose() * output_gradient);
    }
    

    ~DenseLayer() {}

};
