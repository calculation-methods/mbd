#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

double f(double x){
    return x * pow(49 - pow(x, 2), 0.5);
}

// метод прямоугольников
double rectangle(double a, double b, double h){
    int n = (b - a) / h;
    double X[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }

    double F = 0;
    for (int i = 0; i < n; ++i){
        F += f((X[i] + X[i + 1]) / 2);
    }
    return F * h;
}

// метод трапеций
double trapezoid(double a, double b, double h){
    int n = (b - a) / h;
    double X[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }

    double F = 0;
    for (int i = 0; i < n; ++i){
        F += f(X[i]) + f(X[i + 1]);
    }
    return F * h / 2;
}

// метод Симпсона
double Simpson(double a, double b, double h){
    int n = (b - a) / 2 / h;
    double X[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + 2 * h * i;
    }

    double F = 0;
    for (int i = 0; i < n; ++i){
        F += f(X[i]) + 4 * f((X[i] + X[i + 1]) / 2) + f(X[i + 1]);
    }
    return F * h / 3;
}

// метод Рунге-Ромберга
double Runge_Romberg(double F1, double F2, double h1, double h2, int p){
    return F1 + (F1 - F2) / (pow(h2 / h1, p) - 1);
}

int main(){
    double a = -2;
    double b = 2;
    double h1 = 1;
    double h2 = 0.5;
    
    cout << rectangle(a, b, h1) << " " << rectangle(a, b, h2) << "\n";
    cout << trapezoid(a, b, h1) << " " << trapezoid(a, b, h2) << "\n";
    cout << Simpson(a, b, h1) << " " << Simpson(a, b, h2) << "\n";
    cout << "\n";
    cout << Runge_Romberg(rectangle(a, b, h1), rectangle(a, b, h2), h1, h2, 2) << " " << Runge_Romberg(trapezoid(a, b, h1), trapezoid(a, b, h2), h1, h2, 2) << " " << Runge_Romberg(Simpson(a, b, h1), Simpson(a, b, h2), h1, h2, 2) << "\n";
}


