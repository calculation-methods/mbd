#include <iostream>
#include<iomanip>
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

// вывести матрицу на консоль
void printMatrix (matrix & a){
   for (int i = 0; i < a.get_n_rows (); ++ i){
      for (int j = 0; j < a.get_n_cols (); ++ j){
         printf ("%5.3lf ", a[i][j]);
      }
      puts ("");
   }
}

// исходная функция
double f(double x, double y, double z){
    return (- y - (2 - x) * z) / (2 * x * (x + 2));
}

double y(double x){
    return pow(abs(x),  0.5) + x - 2;
}

double Phi(double y, double z){
    return z + y - 5.25;
}

double p(double x){
    return (2 - x) / 2 / x / (x + 2);
}

double q(double x){
    return 0.5 / x / (x + 2);
}

// вспомогательны функции для метода Рунге-Кутты
double K(double x, double y, double z, double h, int i);

double L(double x, double y, double z, double h, int i){
    if (i == 0){
        return h * z;
    }
    else if (i == 3){
        return h * (z + K(x, y, z, h, i - 1));
    }
    else{
        return h * (z + K(x, y, z, h, i - 1) / 2);
    }
}

double K(double x, double y, double z, double h, int i){
    if (i == 0){
        return h * f(x, y, z);
    }
    else if (i == 3){
        return h * f(x + h, y + L(x, y, z, h, i - 1), z + K(x, y, z, h, i - 1));
    }
    else{
        return h * f(x + h / 2, y + L(x, y, z, h, i - 1) / 2, z + K(x, y, z, h, i - 1) / 2);
    }
}

double delta_z(double x, double y, double z, double h){
    double d = 0;
    for (int i = 0; i < 4; ++i){
        if (i == 0 or i == 3){
            d += K(x, y, z, h, i);
        }
        else{
            d += 2 * K(x, y, z, h, i);
        }
    }
    return d / 6;
}

double delta_y(double x, double y, double z, double h){
    double d = 0;
    for (int i = 0; i < 4; ++i){
        if (i == 0 or i == 3){
            d += L(x, y, z, h, i);
        }
        else{
            d += 2 *L(x, y, z, h, i);
        }
    }
    return d / 6;
}

// метод Рунге-Кутты 4 порядка
pair<double*, double*> Runge_Kutta_4(double a, double b, double h, double y0, double z0){
    int n = (b - a) / h;
    double x[n + 1];
    double* y = new double[n + 1];
    double* z = new double[n + 1];
    x[0] = a;
    y[0] = y0;
    z[0] = z0;
    for (int i = 1; i <= n; ++i){
        y[i] = y[i - 1] + delta_y(x[i - 1], y[i - 1], z[i - 1], h);
        z[i] = z[i - 1] + delta_z(x[i - 1], y[i - 1], z[i - 1], h);
        x[i] = x[i - 1] + h;
    }
    return pair<double*, double*>(y, z);
}

// метод стрельбы
double* shooting(double a, double b, double h, double y0, double eps){
    double nu[3] = {1, 0.8, 0};
    double phi[3] = {0, 0, 0};
    int n = (b - a) / h;

    pair<double*, double*> R;
    R = Runge_Kutta_4(a, b, h, y0, nu[0]);
    phi[0] = Phi(R.first[n], R.second[n]);
    R = Runge_Kutta_4(a, b, h, y0, nu[1]);
    phi[1] = Phi(R.first[n], R.second[n]);
    phi[2] = phi[1];
    while (abs(phi[1]) > eps){
        nu[2] = nu[1] - (nu[1] - nu[0]) / (phi[1] - phi[0]) * phi[1];
        R = Runge_Kutta_4(a, b, h, y0, nu[2]);
        phi[2] = Phi(R.first[n], R.second[n]);
        nu[0] = nu[1];
        nu[1] = nu[2];
        phi[0] = phi[1];
        phi[1] = phi[2];
    }
    return R.first;
}

// конечно-разностный метод
double* finite_difference(double a, double b, double h, double y0, double eps){
    int n = (b - a) / h;
    double X[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }
    matrix A = matrix(n, 3);
    double B[n];
    A[0][0] = 0;
    A[0][1] = -2 + pow(h, 2) * q(X[1]);
    A[0][2] = 1 + p(X[1]) * h / 2;
    B[0] = 0;
    for (int i = 1; i < n - 1; ++i){
        A[i][0] = 1 - p(X[i]) * h / 2;
        A[i][1] = -2 + pow(h, 2) * q(X[i]);
        A[i][2] = 1 + p(X[i]) * h / 2;
        B[i] = 0;
    }
    A[n - 1][0] = -1;
    A[n - 1][1] = 1 + h;
    A[n - 1][2] = 0;
    B[n - 1] = 5.25 * h;

    double P[n], Q[n];
    double* Y = new double[n + 1];
    P[0] = -A[0][2] / A[0][1];
    Q[0] = B[0] / A[0][1];
    for (int i = 1; i < n; ++i){
        P[i] = -A[i][2] / (A[i][1] + A[i][0] * P[i - 1]);
        Q[i] = (B[i] - A[i][0] * Q[i - 1]) / (A[i][1] + A[i][0] * P[i - 1]);
    }

    Y[n] = Q[n - 1];
    for (int i = n - 2; i > -1; --i){
        Y[i + 1] = P[i] * Y[i + 2] + Q[i];
    }
    Y[0] = 0;
    return Y;
}

double* Error(double Y_t[], double Y[], int n){
    double* eps = new double[n];
    for (int i = 0; i < n; ++i){
        eps[i] = abs(Y_t[i] - Y[i]);
    }
    return eps;
}


int main(){
    double a = 1;
    double b = 4;
    double y0 = 0;
    double h = 0.3;
    double eps = 0.001;

    double* Ans[5];
    int n = (b - a) / h;

    double* X = new double[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }
    Ans[0] = X;

    // метод стрельбы
    double* Y_S = shooting(a, b, h, y0, eps);
    Ans[1] = Y_S;

    // конечно-разностный метод
    double* Y_FD = finite_difference(a, b, h, y0, eps);
    Ans[2] = Y_FD;

    double Y_t[n + 1];
    for (int i = 0; i <= n; ++i){
        Y_t[i] = y(X[i]);
    }

    // погрещность вычислений методом стрельбы
    double* err = Error(Y_t, Y_S, n + 1);
    Ans[3] = err;

    // погрещность вычислений методом стрельбы
    err = Error(Y_t, Y_FD, n + 1);
    Ans[4] = err;

    for (int i = 0; i < 5; ++i){
        for (int j = 0; j <= n; ++j){
            cout << fixed << Ans[i][j] << " ";
        }
        cout << "\n";
    }
}