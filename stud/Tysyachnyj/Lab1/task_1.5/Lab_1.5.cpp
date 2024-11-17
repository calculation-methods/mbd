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
    matrix (){
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
            printf ("%5.3lf ", a[i][j]);
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

// умножение матриц, дающих в результате скаляр
double scal(matrix &A, matrix &B){
    if (A.get_n_cols () != B.get_n_rows () or A.get_n_rows () != 1 or B.get_n_cols () != 1){
        throw "Eroor";
    }
    double c;
    for (int i = 0; i < A.get_n_rows (); ++ i){
        for (int j = 0; j < B.get_n_cols (); ++ j){
            for (int k = 0; k < A.get_n_cols (); ++ k){
                c += A[i][k] * B[k][j];
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

// транспонирование матрицы
matrix transpose(matrix A){
    int n = A.get_n_rows();
    int m = A.get_n_cols();
    matrix C = matrix (m, n);
    for (int i = 0; i < n; ++ i){
        for (int j = 0; j < m; ++ j){
            C[j][i] = A[i][j];
        }
    }
    return C;
} 

// функция знака
double sign(double n){
    if (n < 0){
        return -1;
    }
    else if (n == 0){
        return 0;
    }
    else{
        return 1;
    }
}

// QR-разложение
pair<matrix, matrix> QR_decomposition(matrix A){
    int n = A.get_n_rows();
    matrix E = matrix(n, n, 1);
    matrix Q = matrix(n, n, 1);
    matrix H = matrix(n, n);
    for (int j = 0; j < n - 1; ++j){
        matrix v = matrix(n, 1);
        for (int i = j; i < n; ++i){
            v[i][0] = A[i][j];
            if (i == j){
                double s = 0;
                for (int k = j; k < n; ++k){
                    s += pow(A[k][j], 2);
                }
                v[i][0] += sign(A[i][j]) * pow(s, 0.5);
            }
        }
        matrix v_T = transpose(v);
        matrix v1 = mul(v, v_T);
        H = mul(v1, 2);
        H = E - mul(H, 1 / scal(v_T, v));
        Q = mul(Q, H);
        A = mul(H, A);
    }
    return pair<matrix, matrix> (Q, A);
}

// разложение корня квадратного уравнения на вещественную и мнимую части
pair<double, double> complex(double b, double c){
    if (pow(b, 2) - 4 * c < 0){
        return pair<double, double> (-b / 2, pow(- pow(b, 2) + 4 * c, 0.5) / 2);
    }
    else{
        return pair<double, double> (-b / 2 + pow(pow(b, 2) - 4 * c, 0.5) / 2, 0);
    }
}

// проверка окончания
bool check(matrix A[2], double eps){
    int n = A[1].get_n_rows();
    int j = 0;
    while (j < n - 1){
        pair<double, double> z0 = complex( - A[0][j][j] - A[0][j + 1][j + 1], A[0][j][j] * A[0][j + 1][j + 1] - A[0][j][j + 1] * A[0][j + 1][j]);
        pair<double, double> z1 = complex( - A[1][j][j] - A[1][j + 1][j + 1], A[1][j][j] * A[1][j + 1][j + 1] - A[1][j][j + 1] * A[1][j + 1][j]);
        if (z1.second == 0){
            double s = 0;
            for (int i = j + 1; i < n; ++i){
                s += pow(A[1][i][j], 2);
            }
            s = pow(s, 0.5);
            if (s > eps){
                return true;
            }
        }
        else{
            if (pow(pow(z1.first - z0.first, 2) + pow(z1.second - z0.second, 2), 0.5) > eps){
                return true;
            }
            else{
                j += 1;
            }
        }
        j += 1;
    }
    return false;
}





/*
-9 9 -7 -7 5 -1 -4 3 4
*/

int main()
{
    int n = 3;
    matrix A = matrix (n, n);
    A = scanMatrix(n, n);

    // начало QR-разложения
    double eps = 0.01;
    matrix AA[2] = {A, matrix(n, n)};
    auto [Q, R] = QR_decomposition(A);
    AA[1] = mul(R, Q);
    int k = 1;
    
    while (check(AA, eps)){
        auto [Q, R] = QR_decomposition(AA[1]);
        AA[0] = AA[1];
        AA[1] = mul(R, Q);
        k += 1;
    }
    
    // получение собственных значений
    string eigenvalue[n];
    int i = 0;
    while (i < n - 1){
        pair<double, double> z = complex(AA[1][i][i] + AA[1][i + 1][i + 1], AA[1][i][i] * AA[1][i + 1][i + 1] - AA[1][i][i + 1] * AA[1][i + 1][i]);
        if (z.second == 0){
            eigenvalue[i] = to_string(AA[1][i][i]);
        }
        else{
            eigenvalue[i] = to_string(z.first) + " + " + to_string(z.second) + "i";
            i += 1;
            eigenvalue[i] = to_string(z.first) + " - " + to_string(z.second) + "i";
        i += 1;
        }
        i += 1;
    }

    printMatrix(AA[1]);
    cout << k << "\n";
    for (int i = 0; i < n; ++i){
        cout << eigenvalue[i] << " ";
    }
}