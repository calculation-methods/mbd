#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

vector<vector<double>> multiple_matrix(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2) {
    int n1 = matrix1.size(), m1 = matrix1[0].size(), m2 = matrix2[0].size();
    vector<vector<double>> res(n1);
    for (int i=0; i<n1; ++i)
        for (int j=0; j<m2; ++j)
            res[i].push_back(0);

    for (int i=0; i<n1; ++i) {
        for (int j=0; j<m2; ++j) {
            double cntr = 0;
            for (int k=0; k<m1; k++)
               cntr += matrix1[i][k] * matrix2[k][j];
            res[i][j] = cntr;
        }
    }
    return res;
}

vector<vector<double>> transp(const vector<vector<double>>& matrix1){
    int n = matrix1.size(), m = matrix1[0].size();
    vector<vector<double>> res(m, vector<double>(n));
    for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j)
            res[j][i] = matrix1[i][j];
    return res;
}

vector<vector<double>> E(int n){
    vector<vector<double>> E(n, vector<double>(n, 0));
    for(int i=0; i<n; ++i)
        E[i][i] = 1;
    return E;
}

vector<vector<double>> U(const vector<vector<double>>& matrix1){
    vector<vector<double>> u = E(matrix1.size());
    int i_max = 0, j_max = 1;
    double k = matrix1[0][1];
    int n = matrix1.size(), m = matrix1[0].size();
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < m; ++j)
            if (abs(matrix1[i][j]) > k) {
                k = abs(matrix1[i][j]);
                i_max = i;
                j_max = j;
            }
    double phi = (matrix1[i_max][i_max] == matrix1[j_max][j_max]) ? atan(2*matrix1[i_max][j_max]/(matrix1[i_max][i_max] - matrix1[j_max][j_max])) / 2 : 3.14/4;

    u[i_max][j_max] = -sin(phi);
    u[j_max][i_max] = sin(phi);
    u[i_max][i_max] = cos(phi);
    u[j_max][j_max] = cos(phi);
    return u;
}

double eps(const vector<vector<double>>& matrix1){
    int n=matrix1.size(), m=matrix1[0].size();
    double error = 0;
    for (int i=0; i<n; ++i)
        for (int j=i+1; j<m; ++j)
            error += matrix1[i][j]*matrix1[i][j];
    return sqrt(error);
}

pair<vector<double>, vector<vector<double>>> jacobi(vector<vector<double>>& coeff_matrix, double EPS){
    int n = coeff_matrix.size();
    vector<vector<double>> eigenvectors = E(n);
    while (eps(coeff_matrix) > EPS){
        vector<vector<double>> u = U(coeff_matrix);
        eigenvectors = multiple_matrix(eigenvectors, u);
        vector<vector<double>> U_T = transp(u);
        coeff_matrix = multiple_matrix(multiple_matrix(U_T, coeff_matrix), u);
    }

    vector<double> eigenvalues(n);
    for (int i=0; i<n; ++i)
        eigenvalues[i] = coeff_matrix[i][i];
    return make_pair(eigenvalues, eigenvectors);
}

int main() {
    ifstream in("input.txt");
    int n; // Количество неизвестных
    in >> n;
    vector<vector<double>> X(n), Y(n);

    //Чтение матрицы коэффицентов
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j) {
            int number;
            in >> number;
            X[i].push_back(number);
        }

    in.close();

    ofstream out("output.txt");

    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;
    tie(eigenvalues, eigenvectors) = jacobi(X, 0.01);

    out << "Собственные значения:";
    for (int i=0; i<n; ++i)
        out << " " << eigenvalues[i];
    out << endl << endl;

    out << "Собственные векторы:";
    for (int i=0; i<n; ++i){
        out << endl << i+1 << ": ";
        for(double element: eigenvectors[i])
            out << element << " ";
    }

    out.close();
    return 0;
}