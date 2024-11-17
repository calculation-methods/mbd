#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double df(vector<double> &x, vector<double> &y, double X) {
    auto lb = lower_bound(x.begin(), x.end(), X);
    int def = lb - x.begin() - 1;
    if (def == -1) def = 0;
    double a = (y[def + 1] - y[def]) / (x[def + 1] - x[def]),
    b = ((y[def + 2] - y[def + 1]) / (x[def + 2] - x[def + 1]) - (y[def + 1] - y[def]) / (x[def + 1] - x[def])) / (x[def + 2] - x[def]);
    return (a + b * (2 * X - x[def] - x[def + 1]));
}

double ddf(vector<double> &x, vector<double> &y, double X) {
    auto lb = lower_bound(x.begin(), x.end(), X);
    int def = lb - x.begin() - 1;
    if (def == -1) def = 0;
    return (2 * (((y[def + 2] - y[def + 1]) / (x[def + 2] - x[def + 1])) - ((y[def + 1] - y[def])) / (x[def + 1] - x[def]))) / (x[def + 2] - x[def]);
}

int main() {
    vector<double> x = {0.0, 0.2, 0.4, 0.6, 0.8};
    vector<double> y = {1.0, 1.4214, 1.8918, 2.4221, 3.0255};
    double X = 0.4;
    ofstream fout("answer.txt");
    fout << "Derivatives of f(x):\n";
    fout << "y`(X*) = " << df(x, y, X) << "\n";
    fout << "y``(X*) = " << ddf(x, y, X) << "\n";
    return 0;
}