#include <iostream>
#include <vector>
#include <utility>

using namespace std;

using matrix = vector<vector<double>>;

matrix multiple_matrix(const matrix& matrix1, const matrix& matrix2) {
    int n1 = matrix1.size(), m1 = matrix1[0].size(), m2 = matrix2[0].size();
    matrix res(n1, vector<double>(m2, 0));

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < m2; j++) {
            double cntr = 0;
            for (int k = 0; k < m1; k++) {
                cntr += matrix1[i][k] * matrix2[k][j];
            }
            res[i][j] = cntr;
        }
    }
    return res;
}

pair<matrix, matrix> lu_decomposition(matrix& coefficients, matrix& results) {
    int n1 = coefficients.size(), m1 = coefficients[0].size(), m2 = results[0].size();
    matrix L(n1, vector<double>(n1, 0)), U = coefficients;

    for (int k = 0; k < n1; k++) {
        if (U[k][k] == 0) {
            for (int i = k + 1; i < n1; i++) {
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
        for (int i = k + 1; i < n1; i++) {
            L[i][k] = U[i][k] / U[k][k];
            if (U[i][k] == 0) continue;
            for (int j = k; j < m1; j++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    return make_pair(L, U);
}

double get_determinant(matrix& coefficients, matrix& results) {
    pair<matrix, matrix> LU = lu_decomposition(coefficients, results);
    double det = 1;
    matrix U = LU.second;
    for (int i = 0; i < coefficients.size(); i++) {
        det *= U[i][i];
    }
    return det;
}

matrix calculate_decisions(matrix& coefficients, matrix& results) {
    pair<matrix, matrix> LU = lu_decomposition(coefficients, results);
    matrix L = LU.first, U = LU.second;
    matrix res = results;

    for (int k = 0; k < res[0].size(); k++) {
        for (int i = 0; i < res.size(); i++) {
            for (int j = 0; j < i; j++) {
                res[i][k] -= res[j][k] * L[i][j];
            }
        }
        for (int i = coefficients.size() - 1; i >= 0; i--) {
            for (int j = i + 1; j < results.size(); j++) {
                res[i][k] -= res[j][k] * U[i][j];
            }
            res[i][k] /= U[i][i];
        }
    }

    return res;
}

matrix get_inverse_matrix(matrix& matrix1) {
    matrix E(matrix1.size(), vector<double>(matrix1.size(), 0));
    for (int i = 0; i < matrix1.size(); i++) {
        E[i][i] = 1;
    }
    return calculate_decisions(matrix1, E);
}

void print_matrix(const matrix& matrix1) {
    for (const auto& vect : matrix1) {
        for (auto x : vect) {
            cout << x << " ";
        }
        cout << endl;
    }
}

int main() {
    matrix coefficient_matrix{
            {-1, -3,  -4, -0},
            {3, 7, -8,  3},
            {1,  -6,  2, 5},
            {-8,  -4, -1, -1}
    };

    matrix equation_roots = {
            {-3},
            {30},
            {-90},
            {12}
    };

    pair<matrix, matrix> LU = lu_decomposition(coefficient_matrix, equation_roots);
    matrix l = LU.first;
    matrix u = LU.second;

    cout << "Decisions =" << endl;
    matrix decisions = calculate_decisions(coefficient_matrix, equation_roots);
    print_matrix(decisions);

    cout << endl;
    cout << "det of SLAU matrxi= " << get_determinant(coefficient_matrix, equation_roots) << endl;

    cout << endl << "Inversed matrix SLAU =" << endl;
    matrix inversed = get_inverse_matrix(coefficient_matrix);
    print_matrix(inversed);
    return 0;
}
