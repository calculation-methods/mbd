#include <bits/stdc++.h>

using namespace std;
using matrix = vector<vector<double>>;

pair<matrix, matrix> lu_decomposition(matrix& coefficients, matrix& results) {
    int n1=coefficients.size(), m1=coefficients[0].size(), m2 = results[0].size();
    matrix L(n1), U = coefficients;
    for (int i=0; i<n1; i++)
        for (int j=0; j<m1; j++)
            L[i].push_back(0);

    for (int k=0; k<n1; k++) {
        if (U[k][k] == 0) {
            for (int i=k+1; i<n1; i++) {
                if (U[i][k] != 0) {
                    swap(U[k], U[i]);
                    swap(L[k], L[i]);
                    swap(coefficients[k], coefficients[i]);
                    swap(results[k], results[i]);
                    break;
                }
            }
        }
        L[k][k] = 1;
        for (int i=k+1; i<n1; i++) {
            L[i][k] = U[i][k]/U[k][k];
            if (U[i][k] == 0)
                continue;
            for(int j=k; j<m1; j++)
                U[i][j] -= L[i][k]*U[k][j];

        }
    }

    return make_pair(L, U);
}

matrix calculate_results(matrix& coefficients, matrix& results) {
    auto [L, U] = lu_decomposition(coefficients, results);
    matrix res = results;

    for (int k=0; k<res[0].size(); k++)
        for (int i=0; i<res.size(); i++)
            for (int j=0; j<i; j++)
                res[i][k] -= res[j][k]*L[i][j];
    for (int k=0; k<res[0].size(); k++) {
        for (int i=coefficients.size()-1; i>-1; i--) {
            for (int j=i+1; j<results.size(); j++) {
                res[i][k] -= res[j][k]*U[i][j];
            }
            res[i][k] /= U[i][i];
        }
    }

    return res;
}

int main() {
    vector<double> x = {-0.7, -0.4, -0.1, 0.2, 0.5, 0.8}, y = {-0.7754, -0.41152, -0.10017, 0.20136, 0.5236, 0.9273};

    double n = x.size();
    double ex = 0, ex2 = 0, ex3 = 0, ex4 = 0, ey = 0, eyx = 0, eyx2 = 0;
    for (int i=0; i<x.size(); i++){
        ex += x[i]; ex2 += pow(x[i], 2); ex3 += pow(x[i], 3); ex4 += pow(x[i], 4); ey += y[i]; eyx += y[i]*x[i]; eyx2 += y[i]*pow(x[i], 2);
    }

    matrix X = {{n, ex},{ex, ex2}};
    matrix Y = {{ey},{eyx}};
    matrix results = calculate_results(X, Y);

    cout << "Многочлен первого порядка " << results[0][0] << " + " << results[1][0] << " * x" << endl;
    double diff_f1 = 0;
    for (int i = 0; i < n; i++)
        diff_f1 += pow(results[0][0] + results[1][0] * x[i] - y[i], 2);
    cout << "Погрешность " << diff_f1 << endl << endl;

    X = {{n, ex, ex2},{ex, ex2, ex3},{ex2, ex3, ex4}};
    Y = {{ey},{eyx},{eyx2}};
    results = calculate_results(X, Y);
    cout << "Многочлен первого порядка " << results[0][0] << " + " << results[1][0] << " * x + " << results[2][0] << " * x^2" << endl;
    double diff_f2 = 0;
    for (int i = 0; i < n; i++)
        diff_f2 += pow(results[0][0] + results[1][0] * x[i] + results[2][0] * pow(x[i], 2) - y[i], 2);
    cout << "Погрешность " << diff_f2 << endl;

    return 0;
}