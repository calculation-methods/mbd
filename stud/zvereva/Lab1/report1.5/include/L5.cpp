#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

const double EPSILON = 0.00000001;

vector<vector<double>> multiplyMatrices(vector<vector<double>> firstMatrix, vector<vector<double>> secondMatrix) {
    vector<vector<double>> result(firstMatrix.size(), vector<double>(secondMatrix[0].size()));

    for (int row = 0; row < result.size(); row++) {
        for (int col = 0; col < result[row].size(); col++) {
            double cell = 0;
            for (int i = 0; i < secondMatrix.size(); i++) {
                cell += firstMatrix[row][i] * secondMatrix[i][col];
            }
            result[row][col] = cell;
        }
    }

    return result;
}

void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        cout << "|";
        for (const auto& d : row) {
            cout << "\t" << d << "\t";
        }
        cout << "|\n";
    }
}

vector<vector<double>> minusa(vector<vector<double>> a, vector<vector<double>> b) {
    vector<vector<double>> d(a.size(), vector<double>(a.size()));
    for (int i = 0; i < a.size(); ++i) {
        for (int j = 0; j < a.size(); ++j) {
            d[i][j] = a[i][j] - b[i][j];
        }
    }
    return d;
}

vector<vector<double>> transportV(vector<vector<double>> A) {
    int n = A.size();
    if (A[0].size() > n) n = A[0].size();

    vector<vector<double>> transportMatrix(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        transportMatrix[i][0] = A[0][i];
    }
    return transportMatrix;
}

vector<vector<double>> matrixE(int size) {
    vector<vector<double>> E(size, vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j) E[i][j] = 1;
            else E[i][j] = 0;
        }
    }
    return E;
}

double epsilon(vector<vector<double>> a) {
    double res = 0;
    for (int i = 0; i < a.size(); ++i) {
        for (int j = 0; j < i - 1; ++j) res += a[i][j] * a[i][j];
    }
    return sqrt(res);
}

vector<vector<double>> multa(vector<vector<double>> a, double n) {
    vector<vector<double>> A(a.size(), vector<double>(a.size()));
    for (int i = 0; i < a.size(); ++i) {
        for (int j = 0; j < a[i].size(); ++j) A[i][j] = a[i][j] * n;
    }
    return A;
}

vector<vector<double>> houseHold(vector<vector<double>> A, int column) {
    int n = A.size();
    vector<double> v(n, 0);

    double sum = 0;
    for (int j = column; j < n; ++j) sum += A[j][column] * A[j][column];
    sum = sqrt(sum);

    v[column] = A[column][column] + copysign(1.0, A[column][column]) * sum;

    for (int i = column + 1; i < n; ++i) {
        v[i] = A[i][column];
    }

    double a = multiplyMatrices({ v }, transportV({ v }))[0][0];
    a = 2.0 / a;

    vector<vector<double>> mult = multiplyMatrices(transportV({ v }), { v });

    vector<vector<double>> H = minusa(matrixE(n), multa(mult, a));

    return H;
}

vector<vector<vector<double>>> getQR(vector<vector<double>> A) {
    vector<vector<vector<double>>> ListH;

    vector<vector<double>> Ak = A;

    for (int i = 0; i < A.size() - 1; ++i) {
        vector<vector<double>> H = houseHold(Ak, i);
        Ak = multiplyMatrices(H, Ak);
        ListH.push_back(H);
    }

    vector<vector<double>> R = Ak;
    vector<vector<double>> Q;

    Q = multiplyMatrices(ListH[0], ListH[1]);
    for (int i = 2; i < ListH.size(); ++i) {
        Q = multiplyMatrices(Q, ListH[i]);
    }
    return { Q, R };
}


int main() {
    vector<vector<double>> A = {
        { -1, 4, -4 },
        { 2, -5, 0 },
        { -8, -2, 0 }
    };

    vector<vector<double>> Ak = A;
    while (epsilon(Ak) > EPSILON) {
        vector<vector<vector<double>>> QR = getQR(Ak);
        Ak = multiplyMatrices(QR[1], QR[0]);
    }
    cout << "Собственные значения\n[";
    for (int i = 0; i < Ak.size(); ++i) {
        if (i == Ak.size() - 1) cout << " " << Ak[i][i] << " ";
        else if (Ak[i][i + 1] > EPSILON * 10) {
            double k1 = ((Ak[i][i] + Ak[i + 1][i + 1])
                + sqrt(
                    pow((Ak[i][i] + Ak[i + 1][i + 1]), 2) - 4 * (Ak[i][i] * Ak[i + 1][i + 1] - Ak[i][i + 1] * Ak[i + 1][i]))) / 2;
            double k2 = ((Ak[i][i] + Ak[i + 1][i + 1])
                - sqrt(
                    pow((Ak[i][i] + Ak[i + 1][i + 1]), 2) - 4 * (Ak[i][i] * Ak[i + 1][i + 1] - Ak[i][i + 1] * Ak[i + 1][i]))) / 2;
            cout << " " << k1 << " " << k2 << " ";
            ++i;
        }
        else {
            cout << " " << Ak[i][i] << " ";
        }
    }
    cout << "]\n";


    return 0;
}