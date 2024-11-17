#include <vector>
#include <iostream>
#include <locale.h>
#include "matrix.h"

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

void matrix_to_vecs(const Matrix& matrix, vector<vector<double>>& vec) {
    vec.clear();
    vec.assign(matrix.get_n(), vector<double>(3, 0.0));

    for (int i = 0; i < matrix.get_n(); ++i) {
        vec[i][0] = i - 1 < 0 ? 0.0 : matrix[i][i - 1];
        vec[i][1] = matrix[i][i];
        vec[i][2] = i + 1 < matrix.get_m() ? matrix[i][i + 1] : 0.0;
    }
}

void race_method(vector<vector<double>>& vec, const vector<double>& b, vector<double>& x) {
    x.assign(b.size(), 0.0);
    vector<double> P(b.size()), Q(b.size());
    P[0] = -vec[0][2] / vec[0][1];
    Q[0] = b[0] / vec[0][1];

    for (int i = 1; i < (int)x.size(); ++i) {
        double z = (vec[i][1] + vec[i][0] * P[i - 1]);
        P[i] = -vec[i][2];
        P[i] /= z;
        Q[i] = (b[i] - vec[i][0] * Q[i - 1]);
        Q[i] /= z;
    }
    x.back() = Q.back();
    for (int i = x.size() - 2; i >= 0; --i) {
        x[i] = P[i] * x[i + 1] + Q[i];
    }
}


int main() {
    setlocale(0, "");
    Matrix A;
    vector<double> x, b;
    vector<vector<double>> vec;
    std::cout << "Пример трехдиагональной матрицы: \n";
    std::cout << " L = a b 0 0 0 \n";
    std::cout << "     c d e 0 0 \n";
    std::cout << "     0 f g h 0 \n";
    std::cout << "     0 0 i j k \n";
    std::cout << "     0 0 0 l m \n";
    cout << "Введите порядок матрицы:";
    int size = size_init();
    cout << "Введите матрицу: \n";
    matrix_init(A, size);
    cout << "Введите вектор правых частей: \n";
    vector_init(b, size);
    if (A.is_three_diagonal()) {
        matrix_to_vecs(A, vec);
    }
    else {
        cout << "Ошибка! Матрица не является трехдиагональной!" << endl;
        return 0;
    }
    race_method(vec, b, x);

    cout << "Решение системы: \n";
    print_vector_x(x);

}