#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;



class matrix
{
private:
    double **a;
    int n, m;
public:
    // матрица без элементов
    matrix(){
        a = 0;
        n = 0;
        m = 0;
    }

    // матрица NxM, если E, то единичная, иначе нулевая
    matrix (int N, int M, bool E = 0){
        n = N;
        m = M;
        a = new double *[n];
        for (int i = 0; i < n; ++ i){
            a[i] = new double[m];
            for (int j = 0; j < m; ++ j){
                a[i][j] = (i == j) * E;
            }
        }
    }
   
    // получение количества строки и столбцов
    int get_n_rows(){
        return n;
    }
    int get_n_cols(){
        return m;
    }

    double* operator [] (int index){
        return getRow (index);
    }


    // получить строку матрицы
    double* getRow(int index){
        if (index >= 0 && index < n){
            return a[index];
        }
        return 0;
    }

    // получить столбец матрицы
    double* getColumn(int index){
        if (index < 0 || index >= m){
            return 0;
        }
        double * c = new double [n];
        for (int i = 0; i < n; ++ i){
            c[i] = a[i][index];
        }
        return c;
    }

    // поменять местами две строки
    void swapRows (int index1, int index2){
        if (index1 < 0 || index2 < 0 || index1 >= n || index2 >= n){
            return ;
        }
        for (int i = 0; i < m; ++ i){
            swap (a[index1][i], a[index2][i]);
        }
    }
};

// прочитать матрицу с консоли
matrix scanMatrix(int n, int m){
    matrix a = matrix (n, m);
    for (int i = 0; i < n; ++ i){
        for (int j = 0; j < m; ++ j){
            scanf ("%lf", & a[i][j]);
        }
    }
    return a;
}

// вывести матрицу на консоль
void printMatrix (matrix & a){
    for (int i = 0; i < a.get_n_rows (); ++ i){
        for (int j = 0; j < a.get_n_cols (); ++ j){
            printf ("%lf ", a[i][j]);
        }
        puts ("");
    }
}

// умножение матрицы на число
matrix mul (matrix & a, double k){
    matrix c = matrix (a.get_n_rows (), a.get_n_cols ());
    for (int i = 0; i < a.get_n_rows (); ++ i){
        for (int j = 0; j < a.get_n_cols (); ++ j){
            c[i][j] = a[i][j] * k;
        }
    }
    return c;
}

// умножение двух матриц
matrix mul (matrix & a, matrix & b){
    if (a.get_n_cols () != b.get_n_rows ()){
        throw "Error";
    }
    matrix c = matrix (a.get_n_rows (), b.get_n_cols ());
    for (int i = 0; i < a.get_n_rows (); ++ i){
        for (int j = 0; j < b.get_n_cols (); ++ j){
            for (int k = 0; k < a.get_n_cols (); ++ k){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

matrix operator+ (matrix A, matrix B){
    if (A.get_n_rows() != B.get_n_rows() || A.get_n_cols() != B.get_n_cols()){
        throw "Matrix size not matched";
    }
    matrix C = matrix(A.get_n_rows(), A.get_n_cols());
    for (int i = 0; i < A.get_n_rows(); ++i){
        for (int j = 0; j < A.get_n_cols(); ++j){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

matrix operator- (matrix A, matrix B){
    if (A.get_n_rows() != B.get_n_rows() || A.get_n_cols() != B.get_n_cols()){
        throw "Matrix size not matched";
    }
    matrix C = matrix(A.get_n_rows(), A.get_n_cols());
    for (int i = 0; i < A.get_n_rows(); ++i){
        for (int j = 0; j < A.get_n_cols(); ++j){
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

// угловой минор
matrix minor(matrix A, int a, int b){
    int n = A.get_n_rows();
    int m = A.get_n_cols();
    matrix C = matrix (n - 1, m - 1);
    int k = 0;
    for (int i = 0; i < n; ++i){
        if (i > a){
            k = 1;
        }
        for (int j = 0; j < m; ++j){
            if (i != a){
                if (j < b){
                    C[i - k][j] = A[i][j];
                }
                else if (j > b){
                    C[i - k][j - 1] = A[i][j];
                }
            }
        }
    }
    return C;
}

// определитель
double det(matrix A){
    double d = 0;
    int n = A.get_n_rows();
    if (n == 1){ 
        return A[0][0];  
    }
    else if (n == 2){
        return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    }
    else {
        for (int k = 0; k < n; ++k) {
            matrix M = matrix(n - 1, n - 1);
            for (int i = 1; i < n; ++i) {
                int t = 0;
                for (int j = 0; j < n; ++j) {
                if (j == k)
                    continue;
                M[i-1][t] = A[i][j];
                t += 1;
                }
            }
            d += pow(-1, k + 2) * A[0][k] * det(M);
        }
    }
    return d;
}

// норма матрицы
double Norm(matrix A){
    double norm = 0;
    for (int i = 0; i < A.get_n_rows(); ++i){
        for (int j = 0; j < A.get_n_cols(); ++j){
            norm += pow(A[i][j], 2);
        }
    }
    return pow(norm, 0.5);
}

// якобиан, обобщенный для вычисления матриц A
matrix Jacobian(matrix X, int k = -1){
    matrix J = matrix(2, 2);
    J[0][0] = 2 * X[0][0];
    J[0][1] = 2 * X[1][0] - 1;
    J[1][0] = 1;
    J[1][1] = -0.5 / pow(X[1][0] + 1, 0.5);
    if (k != -1){
        J[k][0] = pow(X[0][0], 2) - X[1][0] + pow(X[1][0], 2) - 1;
        J[k][1] = X[0][0] - pow(X[1][0] + 1, 0.5) + 1;
    }
    return J;
}

// матрица эквивалентныз функций
matrix Phi(matrix X){
    matrix phi = matrix(2,1);
    phi[0][0] = pow(X[1][0] + 1, 0.5) - 1;
    phi[1][0] = pow(X[1][0] + 1 - pow(X[0][0], 2), 0.5);
    return phi;
}


int main()
{
    // метод Ньютона
    double eps = 0.001;
    matrix X[2] = {matrix(2, 1), matrix(2, 1)};
    X[0][0][0] = 0.5; X[0][1][0] = 1.4;
    X[1][0][0] = 0.6; X[1][1][0] = 1.5;
    int k = 0;
    matrix detAJ = matrix(2, 1);
    while (Norm(X[1] - X[0]) > eps){
        X[0] = X[1];
        detAJ[0][0] = det(Jacobian(X[1], 0)) / det(Jacobian(X[1]));
        detAJ[1][0] = det(Jacobian(X[1], 1)) / det(Jacobian(X[1]));
        X[1] = X[1] - detAJ;
        k += 1;
    }
    printMatrix(X[1]);
    cout << "\n" << k << "\n\n";

    // метод простых итераций
    // начальный интервал, q и эквивалентная функция подобраны аналитически, исходя из условия сходимости
    k = 0;
    X[0][0][0] = 0.5; X[0][1][0] = 1.4;
    X[1][0][0] = 0.6; X[1][1][0] = 1.5;
    double q = 0.9;
    matrix phi = matrix(2, 1);
    while (q / (1 - q) * Norm(X[1] - X[0]) > eps){
        X[0] = X[1];
        X[1] = Phi(X[1]);
        k += 1;
    }
    printMatrix(X[1]);
    cout << k << "\n";
}