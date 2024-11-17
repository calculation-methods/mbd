#include <vector>
#include <iostream>
#include <locale.h>
#include "matrix.h"

using namespace std;


void compute_solution(const Matrix& U, const Matrix& L, const vector<double>& b, vector<double>& x) {
    x.resize(U.get_m());

    vector<double> z(L.get_m());
    for (int i = 0; i < L.get_m(); ++i) {
        z[i] = b[i];
        for (int j = 0; j < i; ++j) {
            z[i] -= (L[i][j] * z[j]);
        }
    }

    for (int i = U.get_m() - 1; i >= 0; --i) {
        x[i] = z[i];
        for (int j = i + 1; j < U.get_m(); ++j) {
            x[i] -= (x[j] * U[i][j]);
        }
        x[i] /= U[i][i];
    }
}

double gauss_completely(const Matrix& matrix, Matrix& L, Matrix& U, Matrix& X, const vector<double>& b, vector<double>& x) {
    if (!matrix.is_quadratic()) {
        throw "LU не существует!";
    }
    U = matrix;
    L = Matrix(matrix.get_n(), matrix.get_m());
    X = Matrix(matrix.get_n(), matrix.get_m());
    double determinant = 1.0;

    for (int j = 0; j < U.get_m(); ++j) {
        if (U[j][j] == 0.0) {
            throw "LU не существует!!";
        }
        L[j][j] = 1.0;
        for (int i = j + 1; i < U.get_n(); ++i) {
            double l_ij = U[i][j] / U[j][j];
            L[i][j] = l_ij;
            for (int k = j; k < U.get_m(); ++k) {
                U[i][k] -= U[j][k] * l_ij;
            }
        }
    }

    compute_solution(U, L, b, x);

    vector<double> b_1(b.size());
    for (int i = 0; i < U.get_n(); ++i) {
        b_1[i] = 1.0;
        compute_solution(U, L, b_1, X[i]);
        b_1[i] = 0.0;
    }
    X.transpose();

    for (int i = 0; i < U.get_m(); ++i) {
        determinant *= U[i][i];
    }

    return determinant;
}

int size_init() {
    int size;
    cin >> size;
    return size;
}

void matrix_init(Matrix& A, int size) {
    A = Matrix(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cin >> A[i][j];
        }
    }
}

void vector_init(vector<double>& b, int size) {
    b.resize(size);
    for (int i = 0; i < size; ++i) {
        cin >> b[i];
    }
}

void print_vector_x(const vector<double>& x) {
    for (unsigned i = 0; i < x.size(); ++i) {
        cout << 'x' << i + 1 << " = " << x[i] << " ";
    }
    cout << endl;
}


int main() {
    setlocale(0, "");
    Matrix L, U, B, A;
    vector<double> b, x;
    cout << "Введите порядок матрицы:";
    int size = size_init();
    cout << "Введите матрицу: \n";
    matrix_init(A, size);
    cout << "Введите вектор правых частей: \n";
    vector_init(b, size);
    double determinant = gauss_completely(A, L, U, B, b, x);

    cout << "Результат работы программы: \n";
    cout << "Обратная матрица: \n";
    cout << B ;
    cout << "Решение системы: \n\n";
    print_vector_x(x);
    cout << "Определитель: \n";
    cout << determinant;
    cout << "\n";
    return 0;
}