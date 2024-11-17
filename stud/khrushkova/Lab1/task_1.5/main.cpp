#include <bits/stdc++.h>

using namespace std;
using matrix = vector<vector<double> >;


void print_matrix(const matrix& matrix1) {
    for(const auto& vect: matrix1) {
        for (auto x: vect)
            cout << x << " ";
        cout << endl;
    }
}


matrix plus_matrix(const matrix& matrix1, const matrix& matrix2) {
    matrix res;
    int n = matrix1.size();
    for (int i = 0; i < n; i++) {
        vector<double> row;
        for (int j = 0; j < n; j++) {
            row.push_back(matrix1[i][j] + matrix2[i][j]);
        }
        res.push_back(row);
    }

    return res;
}


matrix transposed(const matrix& matrix1){
    int n = matrix1.size(), m = matrix1[0].size();
    matrix res(m, vector<double>(n));
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            res[j][i] = matrix1[i][j];
    return res;
}

matrix multiple_matrix(matrix& matrix1, matrix& matrix2) {
    int n1 = matrix1.size(), m1 = matrix1[0].size(), m2 = matrix2[0].size();
    matrix res(n1);
    for (int i=0; i<n1; i++)
        for (int j=0; j<m2; j++)
            res[i].push_back(0);

    for (int i=0; i<n1; i++) {
        for (int j=0; j<m2; j++) {
            double cntr = 0;
            for (int k=0; k<m1; k++)
               cntr += matrix1[i][k] * matrix2[k][j];
            res[i][j] = cntr;
        }
    }
    return res;
}


int sign(double a){
    return (a >= 0) ? 1 : -1;
}


double get_eps(const matrix& matrix1){
    double eps = 0;
    int n = matrix1.size();
    for(int i=0; i<n; i++)
        for(int j=0; j<i-1; j++)
            eps += matrix1[i][j]*matrix1[i][j];
    return sqrt(eps);
}


matrix get_E_matrix(int n){
    matrix E(n, vector<double>(n, 0));
    for(int i=0; i<n; i++)
        E[i][i] = 1;
    return E;
}


matrix get_H_matrix(const matrix& coefficients, int ind){
    int n = coefficients.size();
    matrix v(n, vector<double>(1));

    for(int i=0; i<n; i++){
        if (i < ind)
            v[i][0] = 0;
        else if (i == ind){
            double sum = 0;
            for (int j=ind; j < n; j++)
                sum += coefficients[j][i]*coefficients[j][i];
            v[i][0] = coefficients[i][i] + sign(coefficients[i][i]) * sqrt(sum);
        }
        else
            v[i][0] = coefficients[i][ind];   
    }

    matrix transposed_v = transposed(v);
    double k = -multiple_matrix(transposed_v, v)[0][0]/2;
    v = multiple_matrix(v, transposed_v);
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            v[i][j] /= k;

    matrix E = get_E_matrix(n);
    return plus_matrix(E, v);
}


pair<matrix, matrix> QR_decomposition(const matrix& coeff){
    matrix coefficients = coeff;
    matrix Q = get_H_matrix(coefficients, 0);
    coefficients = multiple_matrix(Q, coefficients);
    int n = coefficients.size();
    for (int i=1; i<n-1; i++){
        matrix H = get_H_matrix(coefficients, i);
        Q = multiple_matrix(Q, H);
        coefficients = multiple_matrix(H, coefficients);
    }
    return make_pair(Q, coefficients);
}


vector<double> get_eigenvalues(matrix& coefficients, double EPS){
    while (get_eps(coefficients) > EPS){
        pair<matrix, matrix> QR = QR_decomposition(coefficients);
        coefficients = multiple_matrix(QR.second, QR.first);
    }
    int n = coefficients.size();
    vector<double> result(n);
    for (int i=0; i<n; i++)
        result[i] = coefficients[i][i];
    return result;
}


int main() {
    matrix coefficient_matrix{
        {-5, -8, 4},
        {4, 2, 6},
        {-2, 5, -6}
    };


    vector<double> eigenvalues = get_eigenvalues(coefficient_matrix, 0.01);
    int n = eigenvalues.size();

    for (int i=0; i<n; i++)
        cout << "lambda_" << i+1 << " = " << eigenvalues[i] << endl;

    return 0;
}