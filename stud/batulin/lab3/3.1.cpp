#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

double initFunc(double x){
    return sin(x);
}

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

double SplitDifference(double X[], double Y[], int n){
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
        return (SplitDifference(X_a, Y_a, n - 1) - SplitDifference(X_b, Y_b, n - 1)) / (X[0] - X[n]);
    }
}

double Newton(double X[], double Y[], int n, double x){
    double s = 0;
    double X_n[n];
    double Y_n[n];
    for (int i = 0; i < n; ++i){
        X_n[i] = X[i];
        Y_n[i] = Y[i];
        double p = SplitDifference(X_n, Y_n, i);
        for (int j = 0; j < i; ++j){
            p *= (x - X[j]);
        }
        s += p;
    }
    return s;
}


int main(){
    int n = 4;
    double X_1[n] = {0.1 * M_PI, 0.2 * M_PI, 0.3 * M_PI, 0.4 * M_PI};
    double X_2[n] = {0.1 * M_PI, M_PI / 6, 0.3 * M_PI, 0.4 * M_PI};
    double Y_1[n] = {initFunc(X_1[0]), initFunc(X_1[1]), initFunc(X_1[2]), initFunc(X_1[3])};
    double Y_2[n] = {initFunc(X_2[0]), initFunc(X_2[1]), initFunc(X_2[2]), initFunc(X_2[3])};
    double x = M_PI / 4;

    double L_1 = Lagrange(X_1, Y_1, n, x);
    cout << L_1 << " " << initFunc(x) << " " << abs(L_1 - initFunc(x)) << "\n"; 
    double L_2 = Lagrange(X_2, Y_2, n, x);
    cout << L_2 << " " << initFunc(x) << " " << abs(L_2 - initFunc(x)) << "\n"; 
    cout << "\n";

    double N_1 = Newton(X_1, Y_1, n, x);
    cout << N_1 << " " << initFunc(x) << " " << abs(N_1 - initFunc(x)) << "\n"; 
    double N_2 = Newton(X_2, Y_2, n, x);
    cout << N_2 << " " << initFunc(x) << " " << abs(N_2 - initFunc(x)) << "\n"; 
}
