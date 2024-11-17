#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace std;

pair<vector<vector<double>>, vector<vector<double>>> QR_decompose(vector<vector<double>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    vector<vector<double>> q(m, vector<double>(n, 0));
    vector<vector<double>> r(n, vector<double>(n, 0));

    for (int i = 0; i < n; ++i) {
        vector<double> v;
        for (int k = 0; k < m; ++k) {
            v.push_back(matrix[k][i]);
        }

        for (int j = 0; j < i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < m; ++k) {
                sum += q[k][j] * matrix[k][i];
            }
            r[j][i] = sum;
            for (int k = 0; k < m; ++k) {
                v[k] -= r[j][i] * q[k][j];
            }
        }

        double norm_v = 0.0;
        for (double val : v) {
            norm_v += val * val;
        }
        r[i][i] = sqrt(norm_v);
        for (int k = 0; k < m; ++k) {
            q[k][i] = v[k] / r[i][i];
        }
    }
    return make_pair(q, r);
}

vector<vector<double>> Multiplication(vector<vector<double>>& A, vector<vector<double>>& B) {
    int m = A.size();
    int n = B[0].size();
    int p = B.size();
    vector<vector<double>> result(m, vector<double>(n, 0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

vector<double> QR_algorithm(vector<vector<double>>& matrix, double eps, ofstream& fout) {
    vector<vector<double>> current_matrix = matrix;
    int n = matrix.size();
    for (int iter = 0; iter < 1000; ++iter) {
        auto [q, r] = QR_decompose(current_matrix);
        current_matrix = Multiplication(r, q);

        double upper_triangular_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                upper_triangular_norm += pow(current_matrix[i][j], 2);
            }
        }
        if (sqrt(upper_triangular_norm) < eps) {
            break;
        }
    }
    auto[q1,r1] = QR_decompose(matrix);
    fout << "Матрица Q:"<< endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            fout <<  fixed << setprecision(2)<<q1[i][j] << " ";
        fout << endl;
    }
    fout << "Матрица R:"<< endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            fout <<  fixed << setprecision(2)<< r1[i][j] << " ";
        fout << endl;
    }

    vector<double> sv(n);
    for (int i = 0; i < n; ++i) {
        sv[i] = current_matrix[i][i];
    }
    return sv;
}


int main() {
    ofstream fout;
    fout.open("output.txt");
    vector<vector<double>> A =  {{-6, 1, -4},
                                {-6, 8, -2},
                                {2, -9, 5}};
    double eps = 0.0001;
    vector<double> sob_val = QR_algorithm(A, eps,fout);
    fout << "Собственные значения:\n";
    for (double value : sob_val) {
        fout << value << " ";
    }

    return 0;
}
