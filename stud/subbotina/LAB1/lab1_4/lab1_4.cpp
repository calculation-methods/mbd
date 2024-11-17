#include <vector>
#include <iostream>
#include <locale.h>
#include "matrix.h"
#include <cmath>
#define M_PI_4     0.785398163397448309616

using namespace std;

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

void print_vector_x(const vector<double>& x) {
    for (unsigned i = 0; i < x.size(); ++i) {
        cout << 'l' << i + 1 << " = " << x[i] << " ";
    }
    cout << endl;
}


int rotate_method(const Matrix& A, Matrix& U, vector<double>& x, double alfa) {
    if (!A.is_simmetric()) {
        throw "Матрица не симметрична!";
    }

    Matrix U_k(A.get_n(), A.get_m()), A_k = A;
    int i_max = 0, j_max = 0, itter = 0;
    double v_max = 0.0, fitta = 0.0, check = 0.0;

    U = Matrix(A.get_n(), A.get_m());
    U.make_ones();
    x.resize(A.get_m());

    do {
        U_k.make_ones();

        v_max = 0.0;
        i_max = j_max = 0;
        for (int i = 0; i < A_k.get_n(); ++i) {
            for (int j = i + 1; j < A_k.get_m(); ++j) {
                if (v_max < abs(A_k[i][j])) {
                    v_max = abs(A_k[i][j]);
                    i_max = i;
                    j_max = j;
                }
            }
        }

        fitta = A_k[i_max][i_max] == A_k[j_max][j_max] ?
            M_PI_4 :
            atan(2 * A_k[i_max][j_max] / (A_k[i_max][i_max] - A_k[j_max][j_max])) / 2;

        U_k[i_max][j_max] = -sin(fitta);
        U_k[i_max][i_max] = cos(fitta);
        U_k[j_max][j_max] = cos(fitta);
        U_k[j_max][i_max] = sin(fitta);

        U = U * U_k;

        A_k = A_k * U_k;
        U_k.transpose();
        A_k = U_k * A_k;

        check = 0.0;
        for (int i = 0; i < A_k.get_n(); ++i) {
            for (int j = i + 1; j < A_k.get_m(); ++j) {
                check += A_k[i][j] * A_k[i][j];
            }
        }
        check = sqrt(check);
        ++itter;
    } while (check > alfa);

    for (int i = 0; i < A_k.get_n(); ++i) {
        x[i] = A_k[i][i];
    }

    return itter;
}

int main() {
    setlocale(0, "");
    Matrix A, U;
    vector<double> x;
    double accuracy = 0.01;
    cout << "Введите порядок матрицы:";
    int size = size_init();
    cout << "Введите матрицу: \n";
    matrix_init(A, size);
    cout << "Введите точность вычислений: \n";
    cin >> accuracy;

    cout << "Метод вращений: \n" << endl;
    int itter = rotate_method(A, U, x, accuracy);
    cout << "Собственные ззначения: \n";
    print_vector_x(x);
    cout << "Собственные векторы: \n";
    for (int j = 0; j < U.get_m(); ++j) {
        cout << "x" << j + 1 << ": " << endl;
        for (int i = 0; i < U.get_n(); ++i) {
            cout.width(8);
            cout << U[i][j] << endl;
        }
        cout << endl;
    }
    cout << "Количество итераций: " << itter;

    return 0;
}