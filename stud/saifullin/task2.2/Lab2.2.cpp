#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;


double f1(double x1, double x2) {
    return x1 * x1 - 2 * log10(x2) - 1;
}

double f2(double x1, double x2) {
    return x1 * x1 - x1 * x2 + 1;
}


vector<vector<double>> jacobian(double x1, double x2) {
    vector<vector<double>> J(2, vector<double>(2));
    J[0][0] = 2 * x1; // первое по x1
    J[0][1] = -2 / x2 / log(10); // первое по x2
    J[1][0] = 2 * x1 - x2; // второе по x1
    J[1][1] = -x1; // второе по x2
    return J;
}


pair<double, double> simpleIteration(double x1_0, double x2_0, double epsilon, int maxIterations, int &iterations) {
    double x1 = x1_0, x2 = x2_0;
    iterations = 0;
    while (iterations < maxIterations) {
        double x1Next = sqrt(2 * log10(x2) + 1); // x1 из первого уравнения
        double x2Next = (x1 * x1 + 1) / x1; // x2 из второго уравнения
        if (abs(x1Next - x1) < epsilon && abs(x2Next - x2) < epsilon && x1Next > 0 && x2Next > 0) {
            return make_pair(x1Next, x2Next);
        }
        x1 = x1Next;
        x2 = x2Next;
        iterations++;
    }
    cout << "cant solved with Simple iterations method" << endl;
    return make_pair(0, 0);
}


pair<double, double> newtonMethod(double x1_0, double x2_0, double epsilon, int maxIterations, int &iterations) {
    double x1 = x1_0, x2 = x2_0;
    iterations = 0;
    while (iterations < maxIterations) {
        vector<vector<double>> J = jacobian(x1, x2);
        double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        if (abs(detJ) < 1e-6) {
            cout << "cant solved with Newton method" << endl;
            return make_pair(0, 0);
        }
        double deltaX = -(f1(x1, x2) * J[1][1] - f2(x1, x2) * J[0][1]) / detJ;
        double deltaY = -(f2(x1, x2) * J[0][0] - f1(x1, x2) * J[1][0]) / detJ;
        double x1Next = x1 + deltaX;
        double x2Next = x2 + deltaY;
        if (abs(x1Next - x1) < epsilon && abs(x2Next - x2) < epsilon) {
            return make_pair(x1Next, x2Next);
        }
        x1 = x1Next;
        x2 = x2Next;
        iterations++;
    }
    cout << "cant solved with Newton method" << endl;
    return make_pair(0, 0);
}

int main() {
    double x1_0 = 1, x2_0 = 1; // начальные значения
    double eps = 0.000001; 
    int maxIter = 1000; 
    int currentIter = 0;
    ofstream fout("answer2.txt");
    fout << "eps = " << eps << endl;
    fout << "Simple iterations method:" << endl;
    pair<double, double> result1 = simpleIteration(x1_0, x2_0, eps, maxIter, currentIter);
    if (result1.first > 0 && result1.second > 0) {
        fout << "x1 = " << result1.first << ", x2 = " << result1.second << endl;
        fout << "Count of iterations: " << currentIter << endl;}
    else{
        pair<double, double> result1 = simpleIteration(result1.first, result1.second, eps, maxIter, currentIter);
        fout << "x1 = " << result1.first << ", x2 = " << result1.second << endl;
        fout << "Count of iterations: " << currentIter << endl;
    }
    fout << "Newton method:" << endl;
    pair<double, double> result2 = newtonMethod(x1_0, x2_0, eps, maxIter, currentIter);
    if (result2.first > 0 && result2.second > 0) {
        fout << "x1 = " << result2.first << ", x2 = " << result2.second << endl;
        fout << "Count of iterations: " << currentIter << endl;}
    else{
        pair<double, double> result1 = simpleIteration(result2.first, result2.second, eps, maxIter, currentIter);
        fout << "x1 = " << result2.first << ", x2 = " << result2.second << endl;
        fout << "Count of iterations: " << currentIter << endl;
    }
    return 0;
}
