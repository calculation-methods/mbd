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



/*
0 12 -5 -3 -18 -8 -2 -16 -9 -4 18 -7 4 -9 0
*/

int main()
{

   int n = 5;
   matrix A = matrix (n, 3);
   double B[n] = {148, 45, -155, 11, 3};
   A = scanMatrix(n, 3);
   
   // прямой ход прогонки
   double P[n], Q[n], X[n];
   P[0] = -A[0][2] / A[0][1];
   Q[0] = B[0] / A[0][1];
   for (int i = 1; i < n; ++i){
      P[i] = -A[i][2] / (A[i][1] + A[i][0] * P[i - 1]);
      Q[i] = (B[i] - A[i][0] * Q[i - 1]) / (A[i][1] + A[i][0] * P[i - 1]);
   }

   // обратный ход прогонки
   X[-1] = Q[-1];
   for (int i = n - 2; i > -1; --i){
      X[i] = P[i] * X[i + 1] + Q[i];
   }
   for (int i = 0; i < n; ++i){
      cout << X[i] << " ";
   }
}