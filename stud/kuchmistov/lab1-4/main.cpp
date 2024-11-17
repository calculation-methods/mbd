#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

const double epsilon = 1e-6;

double getMaxOffDiagonal(const vector<vector<double>>& A, int& p, int& q) {
    int n = A.size();
    double maxVal = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (abs(A[i][j]) > maxVal) {
                maxVal = abs(A[i][j]);
                p = i;
                q = j;
            }
        }
    }

    return maxVal;
}

void rotateMatrix(vector<vector<double>>& A, int p, int q, vector<vector<double>>& V) {
    int n = A.size();
    double tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
    double t = (tau >= 0) ? 1.0 / (tau + sqrt(1.0 + tau * tau)) : -1.0 / (-tau + sqrt(1.0 + tau * tau));
    double c = 1.0 / sqrt(1.0 + t * t);
    double s = c * t;

    // Обновляем матрицу A
    double apq = A[p][q];
    A[p][q] = 0.0;
    A[q][p] = 0.0;
    A[p][p] = c * c * A[p][p] - 2.0 * c * s * apq + s * s * A[q][q];
    A[q][q] = s * s * A[p][p] + 2.0 * c * s * apq + c * c * A[q][q];

    for (int i = 0; i < n; ++i) {
        if (i != p && i != q) {
            double api = A[p][i];
            double aqi = A[q][i];
            A[p][i] = c * api - s * aqi;
            A[i][p] = A[p][i];
            A[q][i] = s * api + c * aqi;
            A[i][q] = A[q][i];
        }
    }

    for (int i = 0; i < n; ++i) {
        double vip = V[i][p];
        double viq = V[i][q];
        V[i][p] = c * vip - s * viq;
        V[i][q] = s * vip + c * viq;
    }
}

void findEigenvaluesAndEigenvectors(const vector<vector<double>>& A, vector<double>& eigenvalues, vector<vector<double>>& eigenvectors) {
    int n = A.size();
    eigenvectors = vector<vector<double>>(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        eigenvectors[i][i] = 1.0;
    }

    vector<vector<double>> B = A;

    int iterations = 0;
    while (true) {
        int p, q;
        double maxOffDiagonal = getMaxOffDiagonal(B, p, q);
        if (maxOffDiagonal < epsilon) // проверяем условие выхода из итераций
            break;

        rotateMatrix(B, p, q, eigenvectors);

        iterations++;
    }

    cout << endl << "Iterations " << iterations << endl << endl;

    eigenvalues.clear();
    for (int i = 0; i < n; ++i) {
        eigenvalues.push_back(B[i][i]);
    }
}

int main() {
    vector<vector<double>> A = {
            {-7, -5, -9},
            {-5, 5, 2},
            {-9, 2, 9}
    };

    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;
    findEigenvaluesAndEigenvectors(A, eigenvalues, eigenvectors);

    cout << "Values:" << endl;
    for (double eigenvalue : eigenvalues) {
        cout << eigenvalue << " " << endl;
    }
    cout << endl;

    cout << "Vectors:" << endl;
    for (int i = 0; i < eigenvectors.size(); ++i) {
        cout << "for lambda = " << eigenvalues[i] << ": " << endl;
        for (double component : eigenvectors[i]) {
            cout << component << endl;
        }
        cout << endl;
    }
    return 0;
}
