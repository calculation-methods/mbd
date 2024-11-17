#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

double f(double x1, double x2) {
    return ((pow(1 / x1, 2) + pow(x1, 2)) - (pow(1 / x2, 2) + pow(x2, 2))) / (x1 - x2);
}

double f1(double x1, double x2, double x3) {
    return (f(x1, x2) - f(x2, x3)) / (x1 - x3);
}

double f2(double x1, double x2, double x3, double x4) {
    return (f1(x1, x2, x3) - f1(x2, x3, x4)) / (x1 - x4);
}

vector <double> omega_val(vector <double> x) {
    vector <double> omega(4, 0);
    omega[0] = (x[0] - x[1]) * (x[0] - x[2]) * (x[0] - x[3]);
    omega[1] = (x[1] - x[0]) * (x[1] - x[2]) * (x[1] - x[3]);
    omega[2] = (x[2] - x[0]) * (x[2] - x[1]) * (x[2] - x[3]);
    omega[3] = (x[3] - x[0]) * (x[3] - x[1]) * (x[3] - x[2]);
    return omega;
}

void Lagrange(vector <double> x, double X, vector <double>& L, vector <double>& y, vector <double>& delta) {
    vector <vector <double>> ans(5, vector<double>(4, 0));
    vector <double> omega = omega_val(x);
    for (int i = 0; i < x.size(); i++) {
        ans[0][i] = x[i];
        ans[1][i] = pow(1 / x[i], 2) + pow(x[i], 2);
        ans[2][i] = omega[i];
        ans[3][i] = (pow(1 / x[i], 2) + pow(x[i], 2)) / omega[i];
        ans[4][i] = X - x[i];
    }

    for (int i = 0; i < x.size(); i++) {
        L[i] = (ans[3][0] * (x[i] - ans[0][1]) * (x[i] - ans[0][2]) * (x[i] - ans[0][3]) \
            + ans[3][1] * (x[i] - ans[0][0]) * (x[i] - ans[0][2]) * (x[i] - ans[0][3]) \
            + ans[3][2] * (x[i] - ans[0][0]) * (x[i] - ans[0][1]) * (x[i] - ans[0][3]) \
            + ans[3][3] * (x[i] - ans[0][0]) * (x[i] - ans[0][1]) * (x[i] - ans[0][2]));
    }

    for (int i = 0; i < x.size(); i++) {
        y[i] = pow(1 / x[i], 2) + pow(x[i], 2);
    }

    for (int i = 0; i < x.size(); i++) {
        delta[i] = fabs(y[i] - L[i]);
    }
}


void Newton(vector <double> x, double X, vector <double>& P, vector <double>& y, vector <double>& delta) {
    vector <vector <double>> ans(5, vector<double>(4, 0));
    for (int i = 0; i < x.size(); i++) {
        ans[0][i] = x[i];
        ans[1][i] = pow(1 / x[i], 2) + pow(x[i], 2);
        if (i < 3) {
            ans[2][i] = f(x[i], x[i + 1]);
        }
        if (i < 2) {
            ans[3][i] = f1(x[i], x[i + 1], x[i + 2]);
        }
    }
    ans[4][0] = f2(x[0], x[1], x[2], x[3]);


    for (int i = 0; i < x.size(); i++) {
        P[i] = (ans[1][0] + ans[2][0] * (x[i] - ans[0][0]) + ans[3][0] * (x[i] - ans[0][0]) * (x[i] - ans[0][1]) \
            + ans[4][0] * (x[i] - ans[0][0]) * (x[i] - ans[0][1]) * (x[i] - ans[0][2]));
    }

    for (int i = 0; i < x.size(); i++) {
        y[i] = pow(1 / x[i], 2) + pow(x[i], 2);
    }

    for (int i = 0; i < x.size(); i++) {
        delta[i] = fabs(y[i] - P[i]);
    }
}

int main() {
    int n = 4;
    vector <double> x1 = { 0.1, 0.5, 0.9, 1.3 };
    vector <double> x2 = { 0.1, 0.5, 1.1, 1.3 };
    double root = 0.8;
    vector <double> L(n, 0), y(n, 0), delta(n, 0), P(n, 0);
    std::cout << "Lagrange method" << std::endl;
    Lagrange(x1, root, L, y, delta);
    std::cout << "for x1" << std::endl << "L(x)\n" << "[";
    for (int i = 0; i < L.size(); i++) std::cout << L[i] << "\t";
    std::cout << "]\n" << "y(x)\n" << "[";
    for (int i = 0; i < y.size(); i++) std::cout << y[i] << "\t";
    std::cout << "]\n" << "delta(x)\n" << "[";
    for (int i = 0; i < delta.size(); i++) std::cout << delta[i] << "\t";
    std::cout << "]\n";
    Lagrange(x2, root, P, y, delta);
    std::cout << "\n\nfor x2" << std::endl << "L(x)\n" << "[";
    for (int i = 0; i < L.size(); i++) std::cout << L[i] << "\t";
    std::cout << "]\n" << "y(x)\n" << "[";
    for (int i = 0; i < y.size(); i++) std::cout << y[i] << "\t";
    std::cout << "]\n" << "delta(x)\n" << "[";
    for (int i = 0; i < delta.size(); i++) std::cout << delta[i] << "\t";
    std::cout << "]\n";
    std::cout << "\n\nNewton method" << std::endl;
    Newton(x1, root, P, y, delta);
    std::cout << "for x1" << std::endl << "P(x)\n" << "[";
    for (int i = 0; i < P.size(); i++) std::cout << P[i] << "\t";
    std::cout << "]\n" << "y(x)\n" << "[";
    for (int i = 0; i < y.size(); i++) std::cout << y[i] << "\t";
    std::cout << "]\n" << "delta(x)\n" << "[";
    for (int i = 0; i < delta.size(); i++) std::cout << delta[i] << "\t";
    std::cout << "]\n";
    Newton(x2, root, P, y, delta);
    std::cout << "\n\nfor x2" << std::endl << "P(x)\n" << "[";
    for (int i = 0; i < P.size(); i++) std::cout << P[i] << "\t";
    std::cout << "]\n" << "y(x)\n" << "[";
    for (int i = 0; i < y.size(); i++) std::cout << y[i] << "\t";
    std::cout << "]\n" << "delta(x)\n" << "[";
    for (int i = 0; i < delta.size(); i++) std::cout << delta[i] << "\t";
    std::cout << "]\n";
}