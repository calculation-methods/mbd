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


pair<int, int> get_indexes(const matrix& matrix1){
    double k = matrix1[0][1];
    pair<int, int> indexes = make_pair(0, 1);
    int n = matrix1.size(), m = matrix1[0].size();
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < m; j++)
            if (abs(matrix1[i][j]) > k) {
                k = abs(matrix1[i][j]);
                indexes = make_pair(i, j);
            }
    return indexes;
}


double get_phi(double a_ij, double a_ii, double a_jj){
    if (a_ii == a_jj)
        return 3,1415926535/4;
    return atan(2*a_ij/(a_ii-a_jj)) / 2;
}


matrix multiple_matrix(const matrix& matrix1, const matrix& matrix2) {
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


matrix get_E_matrix(int n){
    matrix E(n, vector<double>(n, 0));
    for(int i=0; i<n; i++)
        E[i][i] = 1;
    return E;
}


matrix get_U_matrix(const matrix& matrix1){
    matrix res = get_E_matrix(matrix1.size());
    int i_max, j_max;
    tie(i_max, j_max) = get_indexes(matrix1);
    double phi = get_phi(matrix1[i_max][j_max], matrix1[i_max][i_max], matrix1[j_max][j_max]);
    res[i_max][i_max] = cos(phi);
    res[j_max][j_max] = cos(phi);
    res[i_max][j_max] = -sin(phi);
    res[j_max][i_max] = sin(phi);
    return res;
}


double get_eps(const matrix& matrix1){
    double eps = 0;
    int n=matrix1.size(), m=matrix1[0].size();
    for (int i=0; i<n; i++)
        for (int j=i+1; j<m; j++)
            eps += matrix1[i][j]*matrix1[i][j];
    return sqrt(eps);
}


matrix transposed(const matrix& matrix1){
    int n = matrix1.size(), m = matrix1[0].size();
    matrix res(m, vector<double>(n));
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            res[j][i] = matrix1[i][j];
    return res;
}


pair<vector<double>, matrix> Jacobi_method(matrix& coeff_matrix, double EPS){
    int n = coeff_matrix.size();
    matrix eigenvectors = get_E_matrix(n);
    while (get_eps(coeff_matrix) > EPS){
        matrix U = get_U_matrix(coeff_matrix);
        eigenvectors = multiple_matrix(eigenvectors, U);
        matrix transposed_U = transposed(U);
        coeff_matrix = multiple_matrix(multiple_matrix(transposed_U, coeff_matrix), U);
    }

    vector<double> eigenvalues(n);
    for (int i=0; i<n; i++)
        eigenvalues[i] = coeff_matrix[i][i];
    return make_pair(eigenvalues, eigenvectors);
}


int main() {
    matrix coefficient_matrix{
        {4, 7, -1},
        {7, -9, -6},
        {-1, -6, -4}
    };


    vector<double> eigenvalues;
    matrix eigenvectors;
    tie(eigenvalues, eigenvectors) = Jacobi_method(coefficient_matrix, 0.01);
    int n = eigenvalues.size();

    cout << "Eigenvalues:" << endl;
    for (int i=0; i<n; i++)
        cout << "lambda_" << i+1 << " = " << eigenvalues[i] << endl;
    cout << endl;

    cout << "Eigenvectors:" << endl;
    for (int i=0; i<n; i++){
        cout << "x_" << i+1 << " = ";
        for(double element: eigenvectors[i])
            cout << element << " ";
        cout << endl;
    }

    return 0;
}