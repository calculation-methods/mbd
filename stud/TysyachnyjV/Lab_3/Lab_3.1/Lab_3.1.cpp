#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

// исходная функция
double f(double x){
    return 1 / x + x;
}

// Лагранж
double Lagrange(double X[], double Y[], int n, double x){
    double s = 0;
    for (int i = 0; i < n; ++i){
        double p = 1;
        for (int j = 0; j < n; ++j){
            if (i != j){
                p *= (x - X[j]) / (X[i] - X[j]);
            }
        }
        s += Y[i] * p;
    }
    return s;
}

// Разделенная разность порядка n
double Split_difference(double X[], double Y[], int n){
    if (n == 0){
        return Y[0];
    }
    else if (n == 1){
        return (Y[0] - Y[1]) / (X[0] - X[1]);
    }
    else{
        double X_a[n];
        double X_b[n];
        double Y_a[n];
        double Y_b[n];
        for (int i = 0; i < n; ++i){
            X_a[i] = X[i];
            X_b[i] = X[i + 1];
            Y_a[i] = Y[i];
            Y_b[i] = Y[i + 1];
        }
        return (Split_difference(X_a, Y_a, n - 1) - Split_difference(X_b, Y_b, n - 1)) / (X[0] - X[n]);
    }
}

// Ньютон
double Newton(double X[], double Y[], int n, double x){
    double s = 0;
    double X_n[n];
    double Y_n[n];
    for (int i = 0; i < n; ++i){
        X_n[i] = X[i];
        Y_n[i] = Y[i];
        double p = Split_difference(X_n, Y_n, i);
        for (int j = 0; j < i; ++j){
            p *= (x - X[j]);
        }
        s += p;
    }
    return s;
}


int main(){
    int n = 4;
    double X_1[n] = {0.1, 0.5, 0.9, 1.3};
    double X_2[n] = {0.1, 0.5, 1.1, 1.3};
    double Y_1[n] = {f(X_1[0]), f(X_1[1]), f(X_1[2]), f(X_1[3])};
    double Y_2[n] = {f(X_2[0]), f(X_2[1]), f(X_2[2]), f(X_2[3])};
    double x = 0.8;

    // многочлен Лагранжа
    double L_1 = Lagrange(X_1, Y_1, n, x);
    cout << L_1 << " " << f(x) << " " << abs(L_1 - f(x)) << "\n"; 
    double L_2 = Lagrange(X_2, Y_2, n, x);
    cout << L_2 << " " << f(x) << " " << abs(L_2 - f(x)) << "\n"; 

    cout << "\n";

    // многочлен Ньютона
    double N_1 = Newton(X_1, Y_1, n, x);
    cout << N_1 << " " << f(x) << " " << abs(N_1 - f(x)) << "\n"; 
    double N_2 = Newton(X_2, Y_2, n, x);
    cout << N_2 << " " << f(x) << " " << abs(N_2 - f(x)) << "\n"; 
}