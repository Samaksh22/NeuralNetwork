#pragma once

#include "../lib/Eigen/Core"

using namespace std;
using namespace Eigen;

typedef MatrixXd mat;
// typedef VectorXd vec;


class Layer
{
protected:
    mat input;
    mat output;
    int input_size;
    int output_size;

public:
    Layer(){}
    Layer(int in_size, int out_size) : input_size(in_size), output_size(out_size) {}

    // virtual vec forward() = 0;
    // virtual mat backward() = 0;

    ~Layer(){}
};