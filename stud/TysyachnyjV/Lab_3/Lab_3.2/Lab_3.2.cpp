#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

class matrix
{
private:
    double **a;
    int n, m;
public:
    // матрица без элементов
    matrix (){
        a = 0;
        n = 0;
        m = 0;
    }

    // матрица NxM, если E, то единичная, иначе нулевая
    matrix (int N, int M, bool E = 0){
        n = N;
        m = M;
        a = new double *[n];
        for (int i = 0; i < n; ++ i){
            a[i] = new double[m];
            for (int j = 0; j < m; ++ j){
                a[i][j] = (i == j) * E;
            }
        }
    }

    double* operator [] (int index){
        if (index >= 0 && index < n){
            return a[index];
        }
        return 0;
    }
};

double spline(double X[], double Y[], int n, double x){
    double h[n];
    for (int i = 0; i < n; ++i){
        h[i] = X[i + 1] - X[i];
    }

    // решение трехдиагональной системы для нахождения с
    matrix A = matrix (n - 1, 3);
    double B[n];

    A[0][0] = 0;
    A[0][1] = 2 * (h[0] + h[1]);
    A[0][2] = h[1];
    B[0] = 3 * ((Y[2] - Y[1]) / h[1] - (Y[1] - Y[0]) / h[0]);
    for (int i = 1; i < n - 2; ++i){
        A[i][0] = h[i];
        A[i][1] = 2 * (h[i] + h[i + 1]);
        A[i][2] = h[i + 1];
        B[i] = 3 * ((Y[i + 2] - Y[i + 1]) / h[i + 1] - (Y[i + 1] - Y[i]) / h[i]);
    }
    A[n - 2][0] = h[n - 2];
    A[n - 2][1] = 2 * (h[n - 2] + h[n - 1]);
    A[n - 2][2] = 0;
    B[n - 2] = 3 * ((Y[n] - Y[n - 1]) / h[n - 1] - (Y[n - 1] - Y[n - 2]) / h[n - 2]);

    // прямой ход прогонки
    double P[n - 1];
    double Q[n - 1];
    double c[n];
    P[0] = -A[0][2] / A[0][1];
    Q[0] = B[0] / A[0][1];
    for (int i = 1; i < n - 1; ++i){
        P[i] = -A[i][2] / (A[i][1] + A[i][0] * P[i - 1]);
        Q[i] = (B[i] - A[i][0] * Q[i - 1]) / (A[i][1] + A[i][0] * P[i - 1]);
    }

    // обратный ход прогонки
    c[0] = 0;
    c[n - 1] = Q[n - 1];
    for (int i = n - 2; i > 0; --i){
        c[i] = P[i - 1] * c[i + 1] + Q[i - 1];
    }

    // нахождение остальных коэффициентов
    double a[n], b[n], d[n];
    for (int i = 0; i < n - 1; ++i){
        a[i] = Y[i];
        b[i] = (Y[i + 1] - Y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3;
        d[i] = (c[i + 1] - c[i]) / 3 / h[i];
    }
    a[n - 1] = Y[n - 1];
    b[n - 1] = (Y[n] - Y[n - 1]) / h[n - 1] - 2 / 3 * h[n - 1] * c[n - 1];
    d[n - 1] = - c[n - 1] / 3 / h[n - 1];
    
    int i = 0;
    while (X[i] < x and X[i + 1] < x){
        i += 1;
    }
    return a[i] + b[i] * (x - X[i]) + c[i] * pow(x - X[i], 2) + d[i] * pow(x - X[i], 3);
}

int main()
{
    int n = 4;
    double X[n + 1] = {0.1, 0.5, 0.9, 1.3, 1.7};
    double Y[n + 1] = {10.1, 2.5, 2.0111, 2.0692, 2.2882};
    double x = 0.8;

    cout << spline(X, Y, n, x);
}