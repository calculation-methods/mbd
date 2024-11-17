#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

class Matrix {
	public:
	int rows, cols;
   double **a;

   Matrix (){
      rows = 0;
      cols = 0;
      a = new double*[rows];
      for(int i = 0; i < rows; i++)
         a[i] = new double[cols];
   }

	Matrix(int n, int m, bool identity = 0)
	{
      rows = n;
      cols = m;
		a = new double *[rows];
      for (int i = 0; i < rows; i++){
         a[i] = new double[cols];
         for (int j = 0; j < cols; j++){
            a[i][j] = identity * (i == j);
         }
      }
	}

    double* operator[](int i)
	{
      return a[i];
   }

   void RowSwap(int row1, int row2)
	{
		swap(a[row1], a[row2]);
	}

   void scan()
   {
      for(int i = 0; i < rows; i++)
      {
         for(int j = 0; j < cols; j++)
         {
            scanf ("%lf", &a[i][j]);
         }
      }
   }

   void print()
   {
      for(int i = 0; i < rows; i++)
      {
         for(int j = 0; j < cols; j++)
         {
            cout <<"\t"<< a[i][j] << "\t";
         }
         cout << endl;
      }
   }

   Matrix& operator*=(Matrix& m)
   {    
      if (cols != m.rows)
      {
      throw "Wait. That's illegal.";
      }
      Matrix temp(m.rows, m.cols);
      for (int i = 0; i < temp.rows; ++i) 
      {
         for (int j = 0; j < temp.cols; ++j) 
         {
               for (int k = 0; k < cols; ++k) 
               {
                  temp.a[i][j] += (a[i][k] * m.a[k][j]);
               }
         }
      }
      return (*this = temp);
   }
};


Matrix operator* (Matrix & A, double k){
   Matrix C = Matrix (A.rows, A.cols);
   for (int i = 0; i < A.rows; ++ i){
      for (int j = 0; j < A.cols; ++ j){
         C[i][j] = A[i][j] * k;
      }
   }
   return C;
}

Matrix operator* (Matrix & A, Matrix & B){
   Matrix C = Matrix (A.rows, B.cols);
   for (int i = 0; i < A.rows; ++ i){
      for (int j = 0; j < B.cols; ++ j){
         for (int k = 0; k < A.cols; ++ k){
            C[i][j] += A[i][k] * B[k][j];
         }
      }
   }
   return C;
}


double* LU(Matrix U, Matrix B){
    int n = U.rows;

   Matrix M[n - 1];
   Matrix L = Matrix(n, n, 1);
   int p = 0;

   for (int k = 0; k < n - 1; ++k){
      M[k] = Matrix(n, n, 1);
      for (int i = k + 1; i < n; ++i){
         if (U[k][k] == 0){
            int j = k + 1;
            while (U[j][j] == 0 and j < n){
               j += 1;
            }
            if (j == n){
               break;
            }
            U.RowSwap(k, j);
            B.RowSwap(k, j);
            p += 1;
         }
         M[k][i][k] = U[i][k] / U[k][k];
         for (int j = k; j < n; ++j){
            U[i][j] -= M[k][i][k] * U[k][j];
         }
      }
      L *=M[k];
   }

   double Z[n];
   for (int i = 0; i < n; ++i){
      double s = 0;
      for (int j = 0; j < i; ++j){
         s += L[i][j] * Z[j];
      }
      Z[i] = B[i][0] - s;
   }

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

double* SumN(double X[], int n, int k){
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

double* LeastSQ(double X[], double Y[], int n, int k){
    Matrix A = Matrix(k + 1, k + 1);
    Matrix B = Matrix(k + 1, 1);
    double* S = SumN(X, n, 2 * k + 1);
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

double* LeastSQF(double a[], double X[], int n, int k){
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

double ErrSQ(double F[], double Y[], int n){
    double s = 0;
    for (int i = 0; i < n; ++i){
        s += pow(F[i] - Y[i], 2);
    }
    return s;
}

int main()
{
    int n = 6;
    double X[n] = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
    double Y[n] = {-0.5, 0.0, 0.5, 0.86603, 1.0, 0.86603};

    int k = 3;
    for (int i = 1; i <= k; ++i){
        cout << ErrSQ(LeastSQF(LeastSQ(X, Y, n, i), X, n, i), Y, n) << " ";
    }
}
