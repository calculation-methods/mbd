#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

double f(double x) {
    return 1/tan(x);
}

double pi = 2 * acos(0.0);

double Lagrange(double x, std::vector<double> X_i, std::vector<double> f_i) {
    int n = X_i.size();
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double composition = 1;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                composition *= (x - X_i[j]) / (X_i[i] - X_i[j]);
            }
        }
        sum += f_i[i] * composition;
    }
    return sum;
}

double Newton(double x, std::vector<double> X_i, std::vector<double> f_i) {
    int n = X_i.size();
    std::vector<std::vector<double>> differences;

    auto DividedDifferences = [&]() {
        differences = { f_i };
        for (int i = 1; i < n; i++) {
            std::vector<double> temp;
            for (int k = 0; k < n - i; k++) {
                temp.push_back((differences[i - 1][k] - differences[i - 1][k + 1]) / (X_i[k] - X_i[k + i]));
            }
            differences.push_back(temp);
        }
        };

    DividedDifferences();

    double sum = f_i[0];
    for (int i = 1; i < n; i++) {
        double composition = 1;
        for (int j = 0; j < i; j++) {
            composition *= x - X_i[j];
        }
        sum += composition * differences[i][0];
    }
    return sum;
}

int main() {
    std::vector<double> X1 = { pi/8, 2* pi / 8, 3* pi / 8, 4* pi / 8 };
    std::vector<double> f1;
    for (auto x : X1) {
        f1.push_back(f(x));
    }

    std::vector<double> X2 = { pi / 8, 5*pi / 16, 3*pi / 8, pi/2 };
    std::vector<double> f2;
    for (auto x : X2) {
        f2.push_back(f(x));
    }

    double X_variation = pi/3;
    std::cout << "Difference at point x = " << X_variation << ":\n"
        << "For Lagrange: " << std::abs(f(X_variation) - Lagrange(X_variation, X1, f1)) << "\n"
        << "For Newton: " << std::abs(f(X_variation) - Newton(X_variation, X1, f1)) << "\n";

    return 0;
}