#include <bits/stdc++.h>

using namespace std;
using matrix = vector<vector<double> >;

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

matrix calculate_decisions(matrix& coefficients, matrix& results) {
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

void print_matrix(const matrix& matrix1) {
    for(const auto& vect: matrix1) {
        for (auto x: vect)
            cout << x << " ";
        cout << endl;
    }
}

int main() {
    vector<double> x = {-0.7, -0.4, -0.1, 0.2, 0.5, 0.8}, y = {2.3462, 1.9823, 1.671, 1.3694, 1.0472, 0.6435};

    double n = x.size();
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

    matrix matr = {
        {n, sum_x},
        {sum_x, sum_x2}
    };

    matrix root = {
        {sum_y},
        {sum_yx}
    };

    matrix decisions = calculate_decisions(matr, root);

    cout << "F1(x) = (" << decisions[0][0] << ") + (" << decisions[1][0] << ")*x" << endl;
    double loss_f1 = 0;
    for (int i = 0; i < n; i++)
        loss_f1 += pow(decisions[0][0] + decisions[1][0] * x[i] - y[i], 2);
    cout << "Loss = " << loss_f1 << endl << endl;


    matr = {
        {n, sum_x, sum_x2},
        {sum_x, sum_x2, sum_x3},
        {sum_x2, sum_x3, sum_x4}
    };

    root = {
        {sum_y},
        {sum_yx},
        {sum_yx2}
    };

    decisions = calculate_decisions(matr, root);

    cout << "F2(x) = (" << decisions[0][0] << ") + (" << decisions[1][0] << ")*x + (" << decisions[2][0] << ")*x^2" << endl;
    double loss_f2 = 0;
    for (int i = 0; i < n; i++)
        loss_f2 += pow(decisions[0][0] + decisions[1][0] * x[i] + decisions[2][0] * pow(x[i], 2) - y[i], 2);
    cout << "Loss = " << loss_f2 << endl;

    return 0;
}