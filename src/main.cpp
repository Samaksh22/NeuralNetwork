#include <iostream>
#include <conio.h>
#include <math.h>
#include "../include/Dense.h"
#include "../include/Activation.h"
#include "../include/Error.h"
#include "../lib/Eigen/Core"

using namespace std;
using namespace Eigen;

typedef MatrixXd mat;


auto makeNetwork()
{

}


int main(int argc, char *argv[])
{

    DenseLayer l1(2, 3);
    ActivationLayer l2;
    DenseLayer l3(3, 1);
    ActivationLayer l4;

    mat X(2, 4);
    X << 0, 0, 1, 1,
        0, 1, 0, 1;

    mat Y(1, 4);
    Y << 0, 1, 1, 0;

    cout << "Input:\n" << X << endl;
    cout << "Output:\n" << Y << endl;

    int epochs = 1000;
    double learning_rate = 0.1;

    for (int i = 0; i <= epochs; i++)
    {
        double error = 0;
        for (int j = 0; j < X.cols(); j++)
        {
            mat output = X.col(j);
            output = l1.forward(output);
            output = l2.forward(output);
            output = l3.forward(output);
            output = l4.forward(output);

            error += mse(Y.col(j), output);

            mat grad = mse_prime(Y.col(j), output);

            grad = l4.backward(grad, learning_rate);
            grad = l3.backward(grad, learning_rate);
            grad = l2.backward(grad, learning_rate);
            grad = l1.backward(grad, learning_rate);
        }

        error /= X.size();

        if (i % 200 == 0)
            cout << "Epoch: " << i << "  Error: " << error << endl;
    }
    // getch();
}
