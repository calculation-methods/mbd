#include <vector>
#include <iostream>
#include "solve.h"

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

matrix solve(matrix& coefficients, matrix& results) {
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
vector<double> get_sums(vector<double> x,vector<double> y){
    double sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0, sum_y = 0, sum_yx = 0, sum_yx2 = 0;

    for (int i=0; i<x.size(); i++){
        sum_x += x[i];
        sum_x2 += pow(x[i], 2);
        sum_x3 += pow(x[i], 3);
        sum_x4 += pow(x[i], 4);
        sum_y += y[i];
        sum_yx += y[i]*x[i];
        sum_yx2 += y[i]*pow(x[i], 2);
    }
    return {sum_x, sum_x2, sum_x3, sum_x4, sum_y, sum_yx, sum_yx2};

}
