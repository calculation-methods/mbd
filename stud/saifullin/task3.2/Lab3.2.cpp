#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

vector<double> solve_linear_system(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();
    
    
    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(A[k][i]) > abs(A[max_row][i])) {
                max_row = k;
            }
        }
        swap(A[i], A[max_row]);
        swap(b[i], b[max_row]);
        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    return x;
}


double Cubic_spline(vector<double>& x, vector<double>& y, double X) {
    vector<double> A(3, 0);
    vector<vector<double>> B(3, vector<double>(3, 0));
    vector<vector<double>> results(4, vector<double>(4, 0));
    vector<double> roots(3, 0);
    double f = 0;

    for (int i = 2; i < 5; i++) {
        A[i - 2] = (3 * ((y[i] - y[i - 1]) / (x[i] - x[i - 1]) +
                         (y[i - 1] - y[i - 2]) / (x[i - 1] - x[i - 2])));
        for (int j = 0; j < B.size(); j++) {
            for (int k = 0; k < B[0].size(); k++) {
                if (j == k) {
                    B[j][k] = 2 * ((x[i - 1] - x[i - 2]) + (x[i] - x[i - 1]));
                } else if (j < k && (k != B.size() - 1 || j != 0)) {
                    B[j][k] = (x[i] - x[i - 1]);
                } else if (j > k && (j != B.size() - 1 || k != 0)) {
                    B[j][k] = (x[i - 1] - x[i - 2]);
                }
            }
        }
    }

    roots = solve_linear_system(B,A);
    roots.insert(roots.begin(), 0);

    for (int i = 0; i < results.size(); i++) {
        for (int j = 0; j < results[0].size(); j++) {
            if (i != results.size() - 1) {
                if (j == 0) {
                    results[i][j] = y[i];
                }
                if (j == 1) {
                    results[i][j] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (
                            (x[i + 1] - x[i]) * roots[i + 1] + 2 * roots[i]) / 3;
                }
                if (j == 2) {
                    if (i == 0) {
                        results[i][j] = 0;
                    } else {
                        results[i][j] = roots[i];
                    }
                }
                if (j == 3) {
                    results[i][j] = (roots[i + 1] - roots[i]) / (3 * (x[i + 1] - x[i]));
                }
            } else {
                if (j == 0) {
                    results[i][j] = y[i];
                }
                if (j == 1) {
                    results[i][j] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (2 * (x[i + 1] - x[i]) * roots[i]) / 3;
                }
                if (j == 2) {
                    results[i][j] = roots[i];
                }
                if (j == 3) {
                    results[i][j] = -roots[i] / (3 * (x[i + 1] - x[i]));
                }
            }
        }
    }

    f = results[1][0] + results[1][1] * (X - x[1]) + results[1][2] * pow((X - x[2]), 2) + results[1][3] * pow((X - x[3]), 3);

    return f;
}


int main() {
    ofstream fout("answer2.txt");
    vector<double> x = {-0.4, -0.1, 0.2, 0.5, 0.8};
    vector<double> y = {-0.81152, -0.20017, 0.40136, 1.0236, 1.7273};
    double X = 0.1;
    fout << "X = " << X << " in range: " << x[1] << " - " << x[2] << endl;
    fout << "Function value in this range: " << y[1] << " - " << y[2] << endl;
    fout << "Function value in X with cubix spline: " << Cubic_spline(x, y, X) << endl;

    return 0;
}