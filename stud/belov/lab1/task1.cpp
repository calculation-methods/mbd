#include <bits/stdc++.h>

using namespace std;
using matrix = vector<vector<double> >;



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


double get_determinant(matrix& coefficients, matrix& results) {
    auto [_, U] = lu_decomposition(coefficients, results);
    double det = 1;
    for (int i=0; i<coefficients.size(); i++)
        det *= U[i][i];
    return det;
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


matrix get_inverse_matrix(matrix& matrix1) {
    matrix E(matrix1.size());
    for (int i=0; i<matrix1.size(); i++)
        for (int j=0; j<matrix1.size(); j++)
            E[i].push_back((i == j) ? 1 : 0);
    return calculate_decisions(matrix1, E);
}

void print_matrix(const matrix& matrix1) {
    for(const auto& vect: matrix1) {
        for (auto x: vect)
            cout << x << " ";
        cout << endl;
    }
}

int main() {
    matrix coefficient_matrix{
            {2, 7,  -8, 6},
            {4, 4, 0,  -7},
            {-1,  -3,  6, 3},
            {9,  -7, -2, -8}
    };

    matrix equation_roots = {
            {-39},
            {41},
            {4},
            {113}
    };

    auto [l, u] = lu_decomposition(coefficient_matrix, equation_roots);

    cout << endl << "L =" << endl;
    print_matrix(l);

    cout << endl << endl;
    cout << "U =" << endl;
    print_matrix(u);

    cout << endl << endl;
    cout << "det = " << get_determinant(coefficient_matrix, equation_roots) << endl;

    cout << endl << endl << "Decisions =" << endl;
    matrix decisions = calculate_decisions(coefficient_matrix, equation_roots);
    print_matrix(decisions);

    cout << endl << endl << "Inversed matrix =" << endl;
    matrix inversed = get_inverse_matrix(coefficient_matrix);
    print_matrix(inversed);
    return 0;
}
