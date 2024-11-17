#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

void qr_decomposition(vector<vector<double>>& A, vector<vector<double>>& Q, vector<vector<double>>& R, double eps, int n) {
    int m = A[0].size();
    R = vector<vector<double>>(m, vector<double>(m, 0.0));
    Q = A;

    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < j; ++k) {
            double dot_prod = 0.0;
            for (int i = 0; i < n; ++i) {
                dot_prod += Q[i][j] * Q[i][k];
            }
            for (int i = 0; i < n; ++i) {
                Q[i][j] -= dot_prod * Q[i][k];
            }
        }

        double norma = 0.0;
        for (int i = 0; i < n; ++i) {
            norma += Q[i][j] * Q[i][j];
        }
        norma = sqrt(norma);

        for (int i = 0; i < n; ++i) {
            Q[i][j] /= norma;
            R[j][j] = norma;
        }

        for (int k = j + 1; k < m; ++k) {
            double dot_prod = 0.0;
            for (int i = 0; i < n; ++i) {
                dot_prod += Q[i][j] * A[i][k];
            }
            R[j][k] = dot_prod;
        }
    }
}

vector<vector<double>> multiply(vector<vector<double>>& A, vector<vector<double>>& B, int n1, int n2) {
    int m = B[0].size();

    vector<vector<double>> res(n1, vector<double>(m, 0.0));

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n2; ++k) {
                res[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return res;
}

vector<double> comp_eigenvalues(vector<vector<double>>& A, int iter, double eps, int n) {

    vector<vector<double>> Ak = A;

    for (int i = 0; i < iter; ++i) {
        vector<vector<double>> Q, R;
        qr_decomposition(Ak, Q, R, eps, n);
        int n_1 = Q.size();
        int n_2 = R.size();
        Ak = multiply(R, Q, n_1, n_2);
    }

    vector<double> eigenvalues(n);

    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = Ak[i][i];
    }

    return eigenvalues;
}

int main(){
    int n = 3;
    vector<vector<double>> A(n, vector <double>(n, 0));
    double eps = 1e-4;

    ifstream in("matrix.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            in >> A[i][j];
    }

    vector<vector<double>> Q;
    vector<vector<double>> R;
    int iter = 45;
    qr_decomposition(A, Q, R, eps, n);
    vector<double> eigenvalues = comp_eigenvalues(A, iter, eps, n);

    ofstream out("output.txt");

    out << "Матрица Q:" << endl;
    for (const auto& row : Q) 
    {
        for (const auto& elem : row)
            out << elem << "\t";
        out << endl;
    }
    out << "Матрица R:" << endl;
    for (const auto& row : R) 
    {
        for (const auto& elem : row)
            out << elem << "\t";
        out << endl;
    }
    out << "Собственные значения:" << endl;
    for (double val : eigenvalues) {
        out << val << " ";
    }
    out << endl;

    return 0;
}