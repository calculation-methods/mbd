#include <vector>
#include <iostream>
#include <locale.h>
#include "matrix.h"
#include <cmath>

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

vector<double> vector_minus(const vector<double>& a, const vector<double>& b) {
    vector<double> minus = a;
    for (unsigned i = 0; i < minus.size(); ++i) {
        minus[i] -= b[i];
    }
    return minus;
}

vector<double> vector_plus(const vector<double>& a, const vector<double>& b) {
    vector<double> plus = a;
    for (unsigned i = 0; i < plus.size(); ++i) {
        plus[i] += b[i];
    }
    return plus;
}

double norm_of_vector(const vector<double>& vec) {
    double norm = 0.0;
    for (unsigned i = 0; i < vec.size(); ++i) {
        norm += vec[i] * vec[i];
    }
    return sqrt(norm);
}

int simple_itteration(const Matrix& A, const vector<double>& b, vector<double>& x, double alfa) {
    Matrix M = A;
    x.resize(b.size());
    vector<double> last(b.size(), 0.0), r = b;
    double coeff = 0.0;

    if (!M.is_quadratic()) {
        throw "Неправильная матрица!";
    }
    for (int i = 0; i < M.get_n(); ++i) {
        if (!A[i][i]) {
            throw "Неправильная матрица!";
        }
        for (int j = 0; j < M.get_m(); ++j) {
            M[i][j] = i == j ? 0.0 : -A[i][j] / A[i][i];
        }
        r[i] /= A[i][i];
    }

    x = r;
    coeff = M.get_norm();
    if (coeff < 1.0) {
        coeff /= 1 - coeff;
    }
    else {
        coeff = 1.0;
    }

    int itter = 0;

    for (itter = 0; coeff * norm_of_vector(vector_minus(x, last)) > alfa; ++itter) {
        x.swap(last);
        x = vector_plus(r, M * last);
    }

    return itter;
}

int zeidels_method(const Matrix& A, const vector<double>& b, vector<double>& x, double alfa) {
    Matrix M = A;
    x.resize(b.size());
    vector<double> last(b.size(), 0.0), r = b;
    double coeff = 0.0;

    if (!M.is_quadratic()) {
        throw "Неправильная матрица!";
    }
    for (int i = 0; i < M.get_n(); ++i) {
        if (!A[i][i]) {
            throw "Неправильная матрица!";
        }
        for (int j = 0; j < M.get_m(); ++j) {
            M[i][j] = i == j ? 0.0 : -A[i][j] / A[i][i];
        }
        r[i] /= A[i][i];
    }

    x = r;

    coeff = M.get_norm();
    if (coeff < 1) {
        coeff = M.get_upper_norm() / (1 - coeff);
    }
    else {
        coeff = 1.0;
    }

    int itter = 0;

    for (itter = 0; coeff * norm_of_vector(vector_minus(x, last)) > alfa; ++itter) {
        x.swap(last);
        x = r;
        for (int i = 0; i < M.get_n(); ++i) {
            for (int j = 0; j < i; ++j) {
                x[i] += x[j] * M[i][j];
            }
            for (int j = i; j < M.get_m(); ++j) {
                x[i] += last[j] * M[i][j];
            }
        }
    }

    return itter;
}

void print_solution(const vector<double>& x, int itter) {
    cout << "Решение системы:\n" << endl;
    print_vector_x(x);
    cout << "Количесво итераций: " << itter << endl;
    cout << "\n" << endl;
}

int main() {
    setlocale(0, "");
    Matrix A;
    vector<double> x, b;
    vector<vector<double>> vec;
    double accuracy = 0.001;
    cout << "Введите порядок матрицы:";
    int size = size_init();

    cout << "Введите матрицу: \n";
    matrix_init(A, size);
    cout << "Введите вектор правых частей: \n";
    vector_init(b, size);
    cout << "Введите точность вычислений: \n";
    cin >> accuracy;

    cout << "Метод простых итераций:" << endl;
    int itter = simple_itteration(A, b, x, accuracy);
    print_solution(x, itter);

    cout << "Метод Зейделя:" << endl;
    itter = zeidels_method(A, b, x, accuracy);
    print_solution(x, itter);

    return 0;
}