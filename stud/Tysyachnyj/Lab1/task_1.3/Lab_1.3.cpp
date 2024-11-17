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

// обратная матрица
matrix inv(matrix A){
   double d = det(A);
   int n = A.get_n_rows();
   matrix inv_A = matrix(n, n);
   for(int i = 0; i < n; ++i){
      for(int j = 0; j < n; ++j){
         matrix M = matrix(n - 1, n - 1);
         M = minor(A, i, j);
         inv_A[i][j] = pow(-1.0, i + j + 2) * det(M) / d;
      }
   }
   inv_A = transpose(inv_A);
   return inv_A;
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



/*
15 -4 -6 5 4 -14 -1 4 7 -7 27 -8 -3 -3 2 -14
104 70 170 48
*/

int main()
{

   int n = 4;
   matrix A = matrix (n, n);
   matrix Alpha = matrix (n, n);
   matrix Down = matrix (n, n);
   matrix E = matrix (n, n, 1);
   matrix B = matrix (n, 1);
   matrix Beta = matrix(n, 1);
   A = scanMatrix(n, n);
   B = scanMatrix(n, 1);
   
   // приведение СЛАУ у эквивалентному виду
   for (int i = 0; i < n; ++i){
      for (int j = 0; j < n; ++j){
         double s = 0;
         if (i != j){
            Alpha[i][j] = -A[i][j] / A[i][i];
            if (i > j){
               Down[i][j] = Alpha[i][j];
            }
            s += abs(A[i][j]);
         }
         else{
            Alpha[i][j] = 0;
         }
      }
      Beta[i][0] = B[i][0] / A[i][i];
   }

   // решение СЛАУ
   matrix X[2] = {Beta, Beta + mul(Alpha, Beta)};
   int k = 1;
   double eps = 0.01;
   while (Norm(X[1] - X[0]) > eps){
      X[0] = X[1];
      X[1] = Beta + mul(Alpha, X[1]);
      k += 1;
   }
   printMatrix(X[1]);
   cout << "\n";
   cout << k << "\n";

   // решение СЛАУ методом Зейделя
   k = 1;
   matrix C = E - Down;
   C = inv(C);
   C = mul(C, Beta);
   X[0] = Beta;
   X[1] = mul(Alpha, Beta) + C;
   while (Norm(X[1] - X[0]) > eps){
      X[0] = X[1];
      X[1] = mul(Alpha, X[1]) + C;
      k += 1;
   }
   
   cout << "\n";
   printMatrix(X[1]);
   cout << k << "\n";
}