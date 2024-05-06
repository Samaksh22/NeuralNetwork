#pragma once

#include <bits/stdc++.h>
#include "../include/Network.h"
#include "../lib/Eigen/Core"

using namespace std;
using namespace Eigen;

typedef MatrixXd mat;


void getInpMatrix(mat &input)
{
    cout << endl;
    for(int i = 0; i < input.rows(); i++)
    {
        for(int j = 0; j < input.cols(); j++)
        {
            printf("Enter (%d, %d) element: ", i,j);
            cin >> input(i, j);
        }
    }
    cout << endl;
}

