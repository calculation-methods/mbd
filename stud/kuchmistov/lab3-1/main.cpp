#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double lagrangeInterpolation(vector<double> X, vector<double> Y, double x) {
    double result = 0;
    for (int i = 0; i < X.size(); ++i) {
        double term = Y[i];
        for (int j = 0; j < X.size(); ++j) {
            if (j != i) {
                term = term * (x - X[j]) / (X[i] - X[j]);
            }
        }
        result += term;
    }
    return result;
}

double newtonInterpolation(vector<double> X, vector<double> Y, double x) {
    int n = X.size();
    double result = 0;
    vector<vector<double>> f(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        f[i][0] = Y[i];
    }
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < n - i; ++j) {
            f[j][i] = (f[j + 1][i - 1] - f[j][i - 1]) / (X[i + j] - X[j]);
        }
    }
    for (int i = 0; i < n; ++i) {
        double term = f[0][i];
        for (int j = 0; j < i; ++j) {
            term *= (x - X[j]);
        }
        result += term;
    }
    return result;
}

int main() {
    vector<double> X_a = {0, M_PI / 8, 2 * M_PI / 8, 3 * M_PI / 8};
    vector<double> Y_a;
    for (auto x : X_a) {
        Y_a.push_back(tan(x) + x);
    }

    vector<double> X_b = {0, M_PI / 8, M_PI / 3, 3 * M_PI / 8};
    vector<double> Y_b;
    for (auto x : X_b) {
        Y_b.push_back(tan(x) + x);
    }
    
    double X_star = 3 * M_PI / 16;
    
    double trueValue_a = tan(X_star) + X_star;
    double lagrangeResult_a = lagrangeInterpolation(X_a, Y_a, X_star);
    double newtonResult_a = newtonInterpolation(X_a, Y_a, X_star);
    double trueValue_b = tan(X_star) + X_star;
    double lagrangeResult_b = lagrangeInterpolation(X_b, Y_b, X_star);
    double newtonResult_b = newtonInterpolation(X_b, Y_b, X_star);
    
    double error_lagrange_a = abs(lagrangeResult_a - trueValue_a);
    double error_newton_a = abs(newtonResult_a - trueValue_a);
    double error_lagrange_b = abs(lagrangeResult_b - trueValue_b);
    double error_newton_b = abs(newtonResult_b - trueValue_b);
    
    cout << "For a):" << endl;
    cout << "True value: " << trueValue_a << endl;
    cout << "Lagrange polynomial: " << lagrangeResult_a << ", Error: " << error_lagrange_a << endl;
    cout << "Newton polynomial: " << newtonResult_a << ", Error: " << error_newton_a << endl;
    cout << endl;
    cout << "For b):" << endl;
    cout << "True value: " << trueValue_b << endl;
    cout << "Lagrange polynomial: " << lagrangeResult_b << ", Error: " << error_lagrange_b << endl;
    cout << "Newton polynomial: " << newtonResult_b << ", Error: " << error_newton_b << endl;

    return 0;
}
