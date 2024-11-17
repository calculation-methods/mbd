#include <iostream>
#include<iomanip>
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
};

double init(double x, double y, double z){
    return -2 * z / x + y;
}

double exsol(double x){
    return exp(x) / x;
}

double Phi(double y, double z){
    return 1.5 * y + z - exp(2);
}

double p(double x){
    return 2 / x;
}

double q(double x){
    return -1;
}

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
        return h * init(x, y, z);
    }
    else if (i == 3){
        return h * init(x + h, y + L(x, y, z, h, i - 1), z + K(x, y, z, h, i - 1));
    }
    else{
        return h * init(x + h / 2, y + L(x, y, z, h, i - 1) / 2, z + K(x, y, z, h, i - 1) / 2);
    }
}

double dz(double x, double y, double z, double h){
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

double dy(double x, double y, double z, double h){
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

pair<double*, double*> RK4(double a, double b, double h, double y0, double z0){
    int n = (b - a) / h;
    double x[n + 1];
    double* y = new double[n + 1];
    double* z = new double[n + 1];
    x[0] = a;
    y[0] = y0;
    z[0] = z0;
    for (int i = 1; i <= n; ++i){
        y[i] = y[i - 1] + dy(x[i - 1], y[i - 1], z[i - 1], h);
        z[i] = z[i - 1] + dz(x[i - 1], y[i - 1], z[i - 1], h);
        x[i] = x[i - 1] + h;
    }
    return pair<double*, double*>(y, z);
}

double* S(double a, double b, double h, double z0, double eps){
    double nu[3] = {1, 0.8, 0};
    double phi[3] = {0, 0, 0};
    int n = (b - a) / h;

    pair<double*, double*> R;
    R = RK4(a, b, h, nu[0], z0);
    phi[0] = Phi(R.first[n], R.second[n]);
    R = RK4(a, b, h, nu[1], z0);
    phi[1] = Phi(R.first[n], R.second[n]);
    phi[2] = phi[1];
    while (abs(phi[1]) > eps){
        nu[2] = nu[1] - (nu[1] - nu[0]) / (phi[1] - phi[0]) * phi[1];
        R = RK4(a, b, h, nu[2], z0);
        phi[2] = Phi(R.first[n], R.second[n]);
        nu[0] = nu[1];
        nu[1] = nu[2];
        phi[0] = phi[1];
        phi[1] = phi[2];
    }
    return R.first;
}

double* FD(double a, double b, double h, double z0, double eps){
    int n = (b - a) / h;
    double X[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }
    Matrix A = Matrix(n + 1, 3);
    double B[n + 1];
    A[0][0] = 0;
    A[0][1] = -1;
    A[0][2] = 1;
    B[0] = h * z0;
    for (int i = 1; i < n; ++i){
        A[i][0] = 1 - p(X[i]) * h / 2;
        A[i][1] = -2 + pow(h, 2) * q(X[i]);
        A[i][2] = 1 + p(X[i]) * h / 2;
        B[i] = 0;
    }
    A[n][0] = -1;
    A[n][1] = 1 + 1.5 * h;
    A[n][2] = 0;
    B[n] = exp(2) * h;

    double P[n + 1], Q[n + 1];
    double* Y = new double[n + 1];
    P[0] = -A[0][2] / A[0][1];
    Q[0] = B[0] / A[0][1];
    for (int i = 1; i < n + 1; ++i){
        P[i] = -A[i][2] / (A[i][1] + A[i][0] * P[i - 1]);
        Q[i] = (B[i] - A[i][0] * Q[i - 1]) / (A[i][1] + A[i][0] * P[i - 1]);
    }

    Y[n] = Q[n];
    for (int i = n - 1; i > -1; --i){
        Y[i] = P[i] * Y[i + 1] + Q[i];
    }
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
    double b = 2;
    double z0 = 0;
    double h = 0.1;
    double eps = 0.001;

    double* Ans[5];
    int n = (b - a) / h;

    double* X = new double[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }
    Ans[0] = X;

    double* Y_S = S(a, b, h, z0, eps);
    Ans[1] = Y_S;

    double* Y_FD = FD(a, b, h, z0, eps);
    Ans[2] = Y_FD;

    double Y_t[n + 1];
    for (int i = 0; i <= n; ++i){
        Y_t[i] = exsol(X[i]);
    }

    double* err = Error(Y_t, Y_S, n + 1);
    Ans[3] = err;

    err = Error(Y_t, Y_FD, n + 1);
    Ans[4] = err;

    for (int i = 0; i < 5; ++i){
        for (int j = 0; j <= n; ++j){
            cout << fixed << Ans[i][j] << " ";
        }
        cout << "\n";
    }
}
