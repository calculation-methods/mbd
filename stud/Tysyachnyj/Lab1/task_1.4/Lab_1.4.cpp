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

// нахождение максимального наддиагонального элемента
pair<int, int> find_max(matrix A){
   int n = A.get_n_rows();
   int i_max = 0;
   int j_max = 1;
   for (int i = 0; i < n - 1; ++i){
      for (int j = i + 1; j < n; ++j){
         if (abs(A[i][j]) > abs(A[i_max][j_max])){
            i_max = i;
            j_max = j;
         }
      }
   }
   return pair<int, int>(i_max, j_max);
}

// критерий сходимости
double crit(matrix A){
   double s = 0;
   for (int i = 0; i < A.get_n_rows() - 1; ++i){
      for (int j = i + 1; j < A.get_n_cols(); ++j){
         s += pow(A[i][j], 2);
      }
   }
   return pow(s, 0.5);
}



/*
5 -4 7 -4 -3 4 7 4 -1
*/

int main()
{
   int n = 3;
   matrix A = matrix (n, n);
   A = scanMatrix(n, n);
    
   matrix X = matrix (n, n, 1);
   double eps = 0.01;
   int k = 0;

   // метод вращений Якоби
   while (crit(A) > eps){
      matrix U = matrix(n, n, 1);
      matrix U_T = matrix(n, n, 1);
      auto [i_max, j_max] = find_max(A);
      double phi = atan(2 * A[i_max][j_max] / (A[i_max][i_max] - A[j_max][j_max])) / 2;
      U[i_max][i_max] = cos(phi);
      U[j_max][j_max] = cos(phi);
      U[i_max][j_max] = -sin(phi);
      U[j_max][i_max] = sin(phi);
      U_T = transpose(U);
      A = mul(U_T, A);
      A = mul(A, U);
      X = mul(X, U);
      k += 1;
   }

   matrix Lambda = matrix(1, n);
   for (int i = 0; i < n; ++i){
      Lambda[0][i] = A[i][i];
   }
    
   printMatrix(Lambda);
   cout << k << "\n";
}