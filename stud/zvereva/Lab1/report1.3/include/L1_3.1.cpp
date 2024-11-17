#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

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

void printMatrix(vector<vector<double>> matrix) {
    for (auto& m : matrix) {
        cout << "|";
        for (auto& d : m) cout << "\t" << fixed << setprecision(2) << d << "\t";
        cout << "|\n";
    }
}


vector<double> sum(vector<double> a, vector<double> b) {
    vector<double> d(a.size());
    for (int i = 0; i < a.size(); ++i) {
        d[i] = a[i] + b[i];
    }
    return d;
}

vector<double> minusa(vector<double> a, vector<double> b) {
    vector<double> d(a.size());
    for (int i = 0; i < a.size(); ++i) {
        d[i] = a[i] - b[i];
    }
    return d;
}

double normal(vector<vector<double>> a) {
    double res = 0;
    for (int i = 0; i < a.size(); ++i) {
        for (int j = 0; j < a[i].size(); ++j) res += a[i][j] * a[i][j];
    }
    return sqrt(res);
}

vector<vector<double>> transportV(vector<double> v) {
    vector<vector<double>> t(v.size(), vector<double>(1));
    for (int i = 0; i < v.size(); ++i) {
        t[i][0] = v[i];
    }
    return t;
}

vector<double> transportM(vector<vector<double>> v) {
    vector<double> t(v.size());
    for (int i = 0; i < v.size(); ++i) {
        t[i] = v[i][0];
    }
    return t;
}

int main() {
    setlocale(LC_ALL, "Russian");
    vector<vector<double>> A = {
        {28.0, 9.0, -3.0, -7.0},
        {-5.0, 21.0, -5.0, -3.0},
        {-8.0, 1.0, -16.0, 5.0},
        {0.0, -2.0, 5.0, 8.0}
    };
    vector<double> b = { -159.0, 63.0, -45.0, 24.0 };
    int n = A.size();

    cout << "Исходная матрица" << endl;
    printMatrix(A);
    cout << endl;
    cout << "Вектор значений b" << endl;
    printMatrix(transportV(b));
    cout << endl;

    cout << "Преобразование матрицы A" << endl;
    vector<vector<double>> B(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) B[i][j] = 0.0;
            else B[i][j] = -A[i][j] / A[i][i];
        }
    }
    printMatrix(B);
    cout << endl;

    vector<double> d(n);
    for (int i = 0; i < n; ++i) {
        d[i] = b[i] / A[i][i];
    }
    cout << "Вектор преобразованных значений b" << endl;
    printMatrix(transportV(d));
    cout << endl;

    double eps = 0.0001;

    cout << "Метод Зейделя" << endl;
    vector<double> xold = { 1.0, 1.0, 1.0, 1.0 };
    vector<double> xnew = xold;
    double norm;
    int iter = 0;
    do {
        xold = xnew;
        xnew = vector<double>(n);
        for (int i = 0; i < n; ++i) {
            double sum1 = 0;
            for (int j = 0; j < i; ++j) {
                sum1 += B[i][j] * xnew[j];
            }

            double sum2 = 0;
            for (int j = i; j < n; ++j) {
                sum2 += B[i][j] * xold[j];
            }
            xnew[i] = d[i] + sum1 + sum2;
        }
        vector<vector<double>> temp = { minusa(xnew, xold) };
        norm = normal(temp);
        ++iter;
    } while (norm > eps);

    cout << "Вектор приближенных решений X" << endl;
    printMatrix(transportV(xnew));
    cout << "Число итераций: " << iter << endl;
    cout << endl;

    cout << "Проверка решений" << endl;
    cout << "Исходное b" << endl;
    printMatrix(transportV(b));
    cout << "Вычисленное b" << endl;
    printMatrix(multiplyMatrices(A, transportV(xnew)));
    cout << endl;

    return 0;
}