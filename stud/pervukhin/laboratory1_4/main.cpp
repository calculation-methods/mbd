#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

vector<vector<double>> SingleMatrix(int n) {
    vector<vector<double>> identity(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i)
        identity[i][i] = 1.0;
    return identity;
}

pair<vector<double>, vector<vector<double>>> Jacobi(vector<vector<double>>& A, double eps) {
    int n = A.size();
    vector<vector<double>> sv = SingleMatrix(n);
    for (int iter = 0; iter < 10000; ++iter) {
        double max_off_diag = 0;
        int p = 0, q = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (abs(A[i][j]) > max_off_diag) {
                    max_off_diag = abs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off_diag < eps)
            break;

        double phi;
        if (A[p][p] == A[q][q])
            phi = M_PI / 4;
        else
            phi = 0.5 * atan(2 * A[p][q] / (A[p][p] - A[q][q]));

        double c = cos(phi);
        double s = sin(phi);

        vector<vector<double>> U = SingleMatrix(n);
        U[p][p] = c;
        U[p][q] = -s;
        U[q][p] = s;
        U[q][q] = c;

        vector<vector<double>> U_T(n, vector<double>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                U_T[i][j] = U[j][i];
            }
        }

        vector<vector<double>> A_temp(n, vector<double>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0;
                for (int k = 0; k < n; ++k) {
                    sum += U_T[i][k] * A[k][j];
                }
                A_temp[i][j] = sum;
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0;
                for (int k = 0; k < n; ++k) {
                    sum += A_temp[i][k] * U[k][j];
                }
                A[i][j] = sum;
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0;
                for (int k = 0; k < n; ++k) {
                    sum += sv[i][k] * U[k][j];
                }
                sv[i][j] = sum;
            }
        }
    }

    vector<double> sz(n);
    for (int i = 0; i < n; ++i)
        sz[i] = A[i][i];

    return make_pair(sz, sv);
}

int main() {
    ofstream fout;
    fout.open("output.txt");
    double eps = 0.0001;
    vector<vector<double>> A = {{5, -3, -4},
                                {-3, -3, 4},
                                {-4, 4, 0}};
    auto result = Jacobi(A, eps);
    vector<double> sob_val = result.first;
    vector<vector<double>> sob_vec = result.second;

    fout << "Собственные значения:" << endl;
    for (double i : sob_val)
        fout << i << "  ";
    fout << endl;

    fout << "Собственные векторы:" << endl;
    for (int i = 0; i < sob_vec.size(); ++i) {
        for (int j = 0; j < sob_vec[i].size(); ++j) {
            fout << sob_vec[i][j] << " ";
        }
        fout << endl;
    }

    return 0;
}
