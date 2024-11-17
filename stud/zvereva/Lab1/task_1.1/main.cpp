#include <iostream>
#include <cmath>
#include <vector>


using namespace std;

/*
4
-7 3 -4 7
8 -1 -7 6
9 9 3 -6
-7 -9 -8 -5
Введенная матрица:
-7 3 -4 7
8 -1 -7 6
9 9 3 -6
-7 -9 -8 -5
Введите столбец свободных членов b:
-126 29 27 34
Верхняя треугольная матрица U:
-7 3 -4 7
0 2.42857 -11.5714 14
0 0 59.1176 -71.1176
0 0 0 -16.4179
Нижняя треугольная матрица L:
1 0 0 0
-1.14286 1 0 0
-1.28571 5.29412 1 0
1 -4.94118 -1.03483 1
Решение системы:
x1 = 8
x2 = -9
x3 = 2
x4 = -5
*/

template<typename T> void printMatrix(T, int);

void LU_decomposition(double** A, int n, double** L, double** U) {
    for (int i = 0; i < n; i++) {
        L[i][i] = 1;

        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += L[i][j] * U[j][k];
            }

            U[i][k] = A[i][k] - sum;
        }

        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += L[k][j] * U[j][i];
            }

            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

double* solve_LU(double** L, double** U, vector<double> b, double* result, int n) {
    // Решение Ly = b
    double* y = new double[n];
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Решение Ux = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += U[i][j] * result[j];
        }
        result[i] = (y[i] - sum) / U[i][i];
    }

    delete[] y;

    return result;
}

void determinant(double** upper, int n) {
    int result = 1;
    for (int i = 0; i < n; i++) {
        // int flag = 0;
        for (int j = i; j < n; j++) {
            if (i == j) {
                result *= upper[i][j];
                break;
            }
        }
    }
    cout << "The determinant: \n" << result << '\n';
}

template<typename T> void printMatrix(T matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}


vector<vector<double>> trans(vector<vector<double>> mat, int n) {
    vector<vector<double>> trans_mat(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            trans_mat[j][i] = mat[i][j];
        }
    }

    return trans_mat;
}

void check_result(double** A, vector<vector<double>> inverse_matr, vector<vector<double>> E, int n) {
    int err = 0;
    vector<vector<double>> new_E(n, vector<double>(n));

    cout << "The inverse matrix A^(-1): \n";
    printMatrix(inverse_matr, n);
    cout << "The matrix A: \n";
    printMatrix(A, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * inverse_matr[k][j];
            }
            new_E[i][j] = sum;
        }
    }

    cout << "The new unit matrix: \n";
    printMatrix(new_E, n);

    cout << "The unit matrix: \n";
    printMatrix(E, n);
}

void inversed(double** A, int n, double** L, double** U, vector<double> b) {
    vector<vector<double>> E(n, vector<double>(n));

    vector<vector<double>> inversed_A(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) E[i][j] = 1;
            else E[i][j] = 0;
        }

        double* result = new double[n];
        solve_LU(L, U, E[i], result, n);
        for (int j = 0; j < n; j++) {
            inversed_A[i][j] = result[j];
        }
    }

    vector<vector<double>> inverse_matr(n, vector<double>(n));
    inverse_matr = trans(inversed_A, n);

    cout << "The inverse matrix A^(-1): \n";
    printMatrix(inverse_matr, n);

    check_result(A, inverse_matr, E, n);

}

void fullfill(int n, double** matrix) {
    for (int i = 0; i < n; i++) {
        matrix[i] = new double[n];
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 0;
            cin >> matrix[i][j];
        }
    }
}

void fullfill_second(int n, double** matrix) {
    for (int i = 0; i < n; i++) {
        matrix[i] = new double[n];
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 0;
        }
    }
}



void deleteMatrix(double** mat, int n, double** lower, double** upper) {
    for (int i = 0; i < n; i++) {
        delete[] mat[i];
    }
    delete[] mat;

    for (int i = 0; i < n; i++) {
        delete[] lower[i];
    }
    delete[] lower;

    for (int i = 0; i < n; i++) {
        delete[] upper[i];
    }
    delete[] upper;
}

int main() {

    int n;
    cin >> n;

    double** A = new double* [n];
    vector<double> b(n);
    fullfill(n, A);

    cout << "The introduced matrix:" << '\n';
    printMatrix(A, n);

    cout << "Enter the free members column b:" << '\n';
    for (int i = 0; i < n; i++) {
        cin >> b[i];
    }

    double** lower = new double* [n];
    double** upper = new double* [n];
    fullfill_second(n, lower);
    fullfill_second(n, upper);

    LU_decomposition(A, n, lower, upper);

    cout << "The upper triangular matrix U:" << '\n';
    printMatrix(upper, n);
    cout << "The lower triangular matrix L:" << '\n';
    printMatrix(lower, n);

    double* result = new double[n];
    solve_LU(lower, upper, b, result, n);

    determinant(upper, n);

    inversed(A, n, lower, upper, b);

    cout << "The solution of the system:" << '\n';
    for (int i = 0; i < n; i++) {
        cout << "x" << i + 1 << " = " << result[i] << endl;
    }

    deleteMatrix(A, n, upper, lower);
    // delete[] b;
    delete[] result;

    return 0;
}
