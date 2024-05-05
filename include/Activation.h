#pragma once

#include <iostream>
#include <math.h>
// #include "Layer.h"
#include "../lib/Eigen/Core"

using namespace std;
using namespace Eigen;

typedef MatrixXd mat;

// class Activation : public Layer
class ActivationLayer
{
private:
    mat input;
    mat output;
    int input_size;
    int output_size;

public:
    ActivationLayer() {}

    // using tanh activation
    mat activation(mat inp)
    {
        inp = inp.unaryExpr([](double x)
                            { return tanh(x); });
        return inp;
    }

    mat activation_prime(mat inp)
    {
        inp = inp.unaryExpr([](double x)
                            { return (1 - tanh(x) * tanh(x)); });
        return inp;
    }

    mat forward(const mat &input)
    {
        this->input = input;
        return activation(this->input);
    }

    mat backward(const mat &output_gradient, const double learning_rate)
    {
        mat primed_inp = activation_prime(this->input);
        mat z = (output_gradient.cwiseProduct(primed_inp));
        return z;
    }
};
