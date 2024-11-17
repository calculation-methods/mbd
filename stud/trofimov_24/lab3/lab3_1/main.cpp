#include <vector>
#include <iostream>
#include <numeric>
#include "lagrange.h"
#include <math.h>

#include "newton.h"

using namespace std;



double f(double x){
    return 1 / pow(x,2);
}



int main() {
    vector<double> X = {0.1, 0.5, 0.9, 1.3};
    double X0 = 0.8;
    vector<pair<double, double>> cord;

    for (double x : X)
        cord.emplace_back(x, f(x));
    Lagrange lagrange = Lagrange();
    cout << "\n a) The Lagrange polynomial\n";
    cout << "\tResult: " << lagrange.result(X0, cord) << endl;
    cout << "\tLoss: " << abs(lagrange.result(X0, cord) - f(X0)) << endl;


    Newton newton = Newton();

    cout << "\nThe Newton polynomial\n";
    cout << "\tResult: " << newton.result(X0, cord) << endl;
    cout << "\tLoss: " << abs(newton.result(X0, cord) - f(X0)) << endl;

    return 0;
}