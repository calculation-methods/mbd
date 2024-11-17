#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

double func(double x){
    return x / (2 * x + 5);
}

double Rect(double a, double b, double h){
    int n = (b - a) / h;
    double X[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }

    double F = 0;
    for (int i = 0; i < n; ++i){
        F += func((X[i] + X[i + 1]) / 2);
    }
    return F * h;
}

double Trap(double a, double b, double h){
    int n = (b - a) / h;
    double X[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }

    double F = 0;
    for (int i = 0; i < n; ++i){
        F += func(X[i]) + func(X[i + 1]);
    }
    return F * h / 2;
}

double S(double a, double b, double h){
    int n = (b - a) / 2 / h;
    double X[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + 2 * h * i;
    }

    double F = 0;
    for (int i = 0; i < n; ++i){
        F += func(X[i]) + 4 * func((X[i] + X[i + 1]) / 2) + func(X[i + 1]);
    }
    return F * h / 3;
}

double RR(double F1, double F2, double h1, double h2, int p){
    return F1 + (F1 - F2) / (pow(h2 / h1, p) - 1);
}

int main(){
    double xo = -1;
    double xk = 1;
    double h1 = 0.5;
    double h2 = 0.25;

    cout << Rect(xo, xk, h1) << " " << Rect(xo, xk, h2) << "\n";
    cout << Trap(xo, xk, h1) << " " << Trap(xo, xk, h2) << "\n";
    cout << S(xo, xk, h1) << " " << S(xo, xk, h2) << "\n";
    cout << "\n";
    cout << RR(Trap(xo, xk, h1), Rect(xo, xk, h2), h1, h2, 2) << " " << RR(Trap(xo, xk, h1), Trap(xo, xk, h2), h1, h2, 2) << " " << RR(S(xo, xk, h1), S(xo, xk, h2), h1, h2, 2) << "\n";
}
