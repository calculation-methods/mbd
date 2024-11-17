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



/*
-5 -6 4 -2 0 3 -4 -6 2 4 -4 2 1 -8 2 8 
64 -55 -48 68
*/

int main()
{
   int n = 4;
   matrix A = matrix (n, n);
   matrix U = matrix (n, n);

   matrix B = matrix (n, 1);
   A, U = scanMatrix(n, n);
   B = scanMatrix(n, 1);
    
   matrix M[n - 1];
   matrix L = matrix(n, n, 1);
   int p = 0;
   double det = 1;

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
      det *= U[k][k];
      L = mul(L, M[k]);
   }
   det *= pow(-1, p) * U[n - 1][n - 1];
   cout << "\n";

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
   double X[n];
   for (int i = n - 1; i > -1; --i){
      double s = 0;
      for (int j = i + 1; j < n; ++j){
         s += U[i][j] * X[j];
      }
      X[i] = (Z[i] - s) / U[i][i];
   }

   for (int i = 0; i < n; ++i){
      cout << X[i] << " ";
   }
   cout << "\n\n";

   // определитель
   cout << det << "\n\n";

   // обратная матрица
   matrix A_inv = matrix(n, n);
   matrix E = matrix(n, n, 1);
   for (int k = 0; k < n; ++k){
      for (int i = 0; i < n; ++i){
         double s = 0;
         for (int j = 0; j < i; ++j){
            s += L[i][j] * Z[j];
         }
         Z[i] = E[i][k] - s;
      }

      for (int i = n - 1; i > -1; --i){
         double s = 0;
         for (int j = i + 1; j < n; ++j){
            s += U[i][j] * A_inv[j][k];
         }
         A_inv[i][k] = (Z[i] - s) / U[i][i];
      }
   }
   printMatrix(A_inv);
}