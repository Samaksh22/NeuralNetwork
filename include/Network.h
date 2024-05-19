#pragma once

#include <vector>
#include "../lib/Eigen/Core"
#include "../include/Dense.h"
#include "../include/Activation.h"
#include "../include/Error.h"

using namespace std;
using namespace Eigen;

typedef MatrixXd mat;

class Network
{
private:
    vector<int> layers;
    vector<DenseLayer> dl;
    vector<ActivationLayer> al;

    void makeNetwork()
    {
        for(int i = 0; i < layers.size() - 1; i++)
        {
            dl.push_back({layers[i], layers[i+1]});
            al.push_back({});
        }
    }

    mat networkForward(mat output)
    {
        for(int i = 0; i < dl.size(); i++)
        {
            output = dl[i].forward(output);
            output = al[i].forward(output);
        }

        return output;
    }

    mat networkBackword(mat grad, double learningRate)
    {
        for(int i = dl.size() - 1; i >= 0; i--)
        {
            grad = al[i].backward(grad, learningRate);
            grad = dl[i].backward(grad, learningRate);
        }

        return grad;
    }

public:
    Network():layers({1}){}
    Network(vector<int> layers):layers(layers) {}

    void train(mat X, mat Y, int epochs = 1000, double learn_rate = 0.01, int show_at_iteration = 100)
    {
        makeNetwork();
        for (int i = 0; i <= epochs; i++)
        {
            double error = 0;
            for (int j = 0; j < X.cols(); j++)
            {
                mat output = X.col(j);
                output = networkForward(output);

                error += mse(Y.col(j), output);

                mat grad = mse_prime(Y.col(j), output);
                grad = networkBackword(grad, learn_rate);
            }

            error /= X.size();

            if (i % show_at_iteration == 0)
                cout << "Epoch: " << i << "  Error: " << error << endl;
        }
    }

    mat predict(mat input)
    {
        return networkForward(input);
    }


    ~Network(){}
};