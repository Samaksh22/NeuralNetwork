#include <iostream>
#include <conio.h>
#include <math.h>
#include "../include/Network.h"
#include "../include/Extras.h"
#include "../lib/Eigen/Core"

using namespace std;
using namespace Eigen;

typedef MatrixXd mat;

int main()
{

    vector<int> layers{2, 3, 1};

    mat X(2, 4);
    X << 0, 0, 1, 1,
        0, 1, 0, 1;

    mat Y(1, 4);
    Y << 0, 1, 1, 0;

    Network n(layers);
    n.train(X, Y, 2000, 0.1);

    char exit = 0;
    while (exit != 'y')
    {
        mat pred(2, 1);
        
        cout << "\nEnter Input to be Predicted: ";
        getInpMatrix(pred);
        
        cout << "Prediction: " << n.predict(pred) << endl;
        
        cout << "Enter (y) for exit: ";
        exit = getch();
        cout << exit;
    }

    
}
