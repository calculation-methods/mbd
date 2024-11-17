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

// LU-разложение
double* LU(matrix U, matrix B){
    int n = U.get_n_rows();
    
   matrix M[n - 1];
   matrix L = matrix(n, n, 1);
   int p = 0;

   // LU-разложение
   for (int k = 0; k < n - 1; ++k){
      M[k] = matrix(n, n, 1);
      for (int i = k + 1; i < n; ++i){
         if (U[k][k] == 0){
            int j = k + 1;
            while (U[j][j] == 0 and j < n){
               j += 1;
            }
            if (j == n){
               break;
            }
            U.swapRows(k, j);
            B.swapRows(k, j);
            p += 1;
         }
         M[k][i][k] = U[i][k] / U[k][k];
         for (int j = k; j < n; ++j){
            U[i][j] -= M[k][i][k] * U[k][j];
         }
      }
      L = mul(L, M[k]);
   }

   // решение первого СЛАУ
   double Z[n];
   for (int i = 0; i < n; ++i){
      double s = 0;
      for (int j = 0; j < i; ++j){
         s += L[i][j] * Z[j];
      }
      Z[i] = B[i][0] - s;
   }

   // решение второго СЛАУ
   double* X = new double[n];
   for (int i = n - 1; i > -1; --i){
      double s = 0;
      for (int j = i + 1; j < n; ++j){
         s += U[i][j] * X[j];
      }
      X[i] = (Z[i] - s) / U[i][i];
   }
   return X;
}

// сумма n-ых степеней чисел
double* sum_n(double X[], int n, int k){
    double* S = new double[k];
    for (int j = 0; j < k; ++j){
        double s = 0;
        for (int i = 0; i < n; ++i){
            s += pow(X[i], j);
        }
        S[j] = s;
    }
    return S;
} 

// метод наименьших квадратов
double* MLS(double X[], double Y[], int n, int k){
    matrix A = matrix(k + 1, k + 1);
    matrix B = matrix(k + 1, 1);
    double* S = sum_n(X, n, 2 * k + 1);
    for (int i = 0; i < k + 1; ++i){
        for (int j = i; j < k + 1; ++j){
            A[i][j] = S[i + j];
            A[j][i] = S[i + j];
        }
        double s = 0;
        for (int j = 0; j < n; ++j){
            s += Y[j] * pow(X[j], i);
        }
        B[i][0] = s;
    }
    return LU(A, B);
}

// функция наименьших квадратов
double* F_MLS(double a[], double X[], int n, int k){
    double* S = new double[n];
    for (int i = 0; i < n; ++i){
        double s = 0;
        for (int j = 0; j < k + 1; ++j){
            s += a[j] * pow(X[i], j);
        }
        S[i] = s;
    }
    return S;
}

// квадраты ошибок
double SE(double F[], double Y[], int n){
    double s = 0;
    for (int i = 0; i < n; ++i){
        s += pow(F[i] - Y[i], 2);
    }
    return s;
}

int main()
{
    int n = 6;
    double X[n] = {0.1, 0.5, 0.9, 1.3, 1.7, 2.1};
    double Y[n] = {10.1, 2.5, 2.011, 2.0692, 2.2882, 2.5762};

    int k = 3;
    for (int i = 1; i <= k; ++i){
        cout << SE(F_MLS(MLS(X, Y, n, i), X, n, i), Y, n) << " ";
    }
}