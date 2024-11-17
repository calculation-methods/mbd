#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>


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



tuple<vector<double>, vector<double>, double, double> MNK(vector<double>& x, vector<double>& y, int n){
    double xsum = 0, ysum = 0, xsum2 = 0, ysum2 = 0, xysum = 0, xsum3 = 0, xsum4 = 0, xysum2 = 0;

    for (int i = 0; i <= n; i++){
        xsum += x[i];
        ysum += y[i];
        xsum2 += x[i] * x[i];
        ysum2 += y[i] * y[i];
        xysum += x[i] * y[i];
        xsum3 += pow(x[i], 3);
        xsum4 += pow(x[i], 4);
        xysum2 += x[i] * y[i] * x[i];
    }
    vector<vector<double>> A1= {
            {(double)(n+1), xsum},
            {xsum, xsum2}
    };
    vector<double> B1 = {ysum, xysum};
    vector<double> coef1 = SolveSys(A1, B1, 2);
    vector<vector<double>> A2= {
            {(double)(n+1), xsum, xsum2},
            {xsum, xsum2, xsum3},
            {xsum2, xsum3, xsum4}
    };

    vector<double> B2 = {ysum, xysum, xysum2};
    vector<double> coef2 = SolveSys(A2, B2, 3);

    vector<double> f (n+1, 0);
    double err1 = 0;
    for (int i = 0; i <= n; i++){
        f[i] = coef1[0] + coef1[1] * x[i];
    }
    for (int i = 0; i <= n; i++){
        err1 += pow(f[i] - y[i], 2);
    }

    double err2 = 0;
    for (int i = 0; i <= n; i++){
        f[i] = coef2[0] + coef2[1] * x[i] + coef2[2] * x[i] * x[i];
    }
    for (int i = 0; i <= n; i++){
        err2 += pow(f[i] - y[i], 2);
    }

    return make_tuple(coef1, coef2, err1, err2);
}

int main() {
    ofstream fout;
    fout.open("output.txt");
    int n = 5;
    vector<double> x = {-3, -2, -1, 0, 1, 2};
    vector<double> y = {-2.9502, -1.8647, -0.63212, 1, 3.7183, 9.3891};
    auto[coefs1, coefs2, err1, err2] = MNK(x, y, n);
    fout << "Приближающие многочлены 1 степени:" << endl;
    for (int i = 0; i < 2; i++){
        fout << coefs1[i] << " ";
    }
    fout << endl;
    fout << "Cумма квадратов ошибок: " << err1 << endl;
    fout << endl;
    fout << "Приближающие многочлены 2 степени:" << endl;
    for (int i = 0; i < 3; i++){
        fout << coefs2[i] << " ";
    }
    fout << endl;
    fout << "Cумма квадратов ошибок: " << err2 << endl;
    return 0;
}