#pragma once

#include <iostream>
#include <math.h>
#include "../lib/Eigen/Core"

using namespace std;
using namespace Eigen;

typedef MatrixXd mat;

double mse(const mat &y_true, const mat &y_pred)
{
    mat err = (y_true - y_pred);
    err = err.unaryExpr([](auto x)
                        { return x * x; });

    return err.mean();
}

mat mse_prime(const mat &y_true, const mat &y_pred)
{
    mat diff = (y_pred - y_true);

    return (2 * diff / y_true.size());
}