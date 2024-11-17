#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;

vector<double> SolveSys(vector<vector<double>>& A, vector<double>& d, int n){
    vector<double> P (n, 0);
    vector<double> Q (n, 0);
    vector<double> x (n, 0);
    P[0] = - A[0][1] / A[0][0];
    Q[0] = d[0] / A[0][0];
    for (int i = 1; i < n; i++){
        P[i] = - A[i][i+1] / (A[i][i] + A[i][i-1] * P[i-1]);
        Q[i] = (d[i] - A[i][i-1] * Q[i-1]) / (A[i][i] + A[i][i-1] * P[i-1]);
    }
    for (int i = n - 1; i >= 0; i--){
        x[i] = P[i] * x[i+1] + Q[i];
    }
    return x;
}

double Spline(vector<double>& x, vector<double>& f, double x0, int n = 4){
    double res;
    vector<double> h (n+1,0);
    vector<double> a (n+1,0);
    vector<double> b (n+1,0);
    vector<double> c (n+1,0);
    vector<double> d (n+1,0);
    for (int i = 1; i <= n; i++){
        h[i] = x[i] - x[i-1];
    }
    vector<vector<double>> A = {
            {2*(h[1] + h[2]), h[2], 0},
            {h[2], 2*(h[2] + h[3]), h[3]},
            {0, h[3], 2*(h[3] + h[4])}
    };
    vector<double> B = {3*((f[2] - f[1])/h[2] - (f[1] - f[0])/h[1]),
                        3*((f[3] - f[2])/h[3] - (f[2] - f[1])/h[2]),
                        3*((f[4] - f[3])/h[4] - (f[3] - f[2])/h[3])};
    vector<double> c_0 = SolveSys(A, B, n-1);
    c[1] = 0;
    c[2] = c_0[0];
    c[3] = c_0[1];
    c[4] = c_0[2];
    for (int i = 1; i <= n; i++){
        a[i] = f[i-1];
    }
    for (int i = 1; i <= n-1; i++){
        b[i] = (f[i] - f[i-1])/h[i] - 1/3 * h[i]*(c[i+1] + 2*c[i]);
        d[i] = (c[i+1] - c[i])/(3*h[i]);
    }
    b[n] = (f[n] - f[n-1])/h[n] - 2/3*h[n]*c[n];
    d[n] = -c[n]/(3*h[n]);
    res = a[2] + b[2]*(x0 - x[1]) + c[2]*pow(x0 - x[1],2) + d[2]*pow(x0 - x[1],3);

    return res;
}

int main() {
    ofstream fout;
    fout.open("output.txt");
    vector<double> x = {-2, -1, 0, 1, 2};
    vector<double> f = {-1.8647, -0.63212, 1, 3.7183, 9.3891};
    double X0 = -0.5;
    fout << "Значение функции в точке X: " << Spline(x, f, X0) << endl;
    return 0;
}