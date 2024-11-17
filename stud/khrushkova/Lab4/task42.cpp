#include <bits/stdc++.h>

using namespace std;


double f(double x, double y, double z){
    return ((2*x+1)*y + (2*x-1)*z - x*x - x)/2;
}

double y(double x){
    return 2*x - 1 + exp(-x) + (x*x + 1)/2;
}


class double_matrix
{
private:
   double **start_pos;
   int n, m;
public:
    int m_shape(){
        return n;
    }

   int n_shape(){
      return m;
   }

   double* operator [] (int ind){
      return get_n (ind);
   }

   double_matrix(){
      start_pos = 0;
      n = 0;
      m = 0;
   }

   double_matrix(int N, int M, bool E = 0){
        n = N;
        m = M;
        start_pos = new double *[n];
        for (int i = 0; i < n; ++ i){
            start_pos[i] = new double[m];
            for (int j = 0; j < m; ++ j)
                start_pos[i][j] = (i == j) * E;
      }
   }

   double* get_n(int ind){
    if (ind >= 0 && ind < n)
        return start_pos[ind];
    return 0;
   }

   void swapRows (int ind1, int ind2){
      if (ind1 < 0 || ind2 < 0 || ind1 >= n || ind2 >= n)
        return ;
      for (int i = 0; i < m; ++ i)
         swap (start_pos[ind1][i], start_pos[ind2][i]);
   }
};

void cout_matrix(double_matrix & start_pos){
   for (int i = 0; i < start_pos.m_shape (); ++ i){
      for (int j = 0; j < start_pos.n_shape (); ++ j)
         printf ("%5.3lf ", start_pos[i][j]);
      puts ("");
   }
}

double ugol(double y, double z){
    return z + y - 5.25;
}

double p(double x){
    return (2 - x) / 2 / x / (x + 2);
}

double q(double x){
    return 0.5 / x / (x + 2);
}

double K(double x, double y, double z, double precision, int i);

double L(double x, double y, double z, double precision, int i){
    if (i == 0)
        return precision * z;
    else if (i == 3)
        return precision * (z + K(x, y, z, precision, i - 1));
    else
        return precision * (z + K(x, y, z, precision, i - 1) / 2);
}

double K(double x, double y, double z, double precision, int i){
    if (i == 0)
        return precision * f(x, y, z);
    else if (i == 3)
        return precision * f(x + precision, y + L(x, y, z, precision, i - 1), z + K(x, y, z, precision, i - 1));
    else
        return precision * f(x + precision / 2, y + L(x, y, z, precision, i - 1) / 2, z + K(x, y, z, precision, i - 1) / 2);
}

double dz(double x, double y, double z, double precision){
    double d = 0;
    for (int i = 0; i < 4; ++i)
        if (i == 0 || i == 3)
            d += K(x, y, z, precision, i);
        else
            d += 2 * K(x, y, z, precision, i);
    return d / 6;
}

double dy(double x, double y, double z, double precision){
    double d = 0;
    for (int i = 0; i < 4; ++i)
        if (i == 0 || i == 3)
            d += L(x, y, z, precision, i);
        else
            d += 2 *L(x, y, z, precision, i);
    return d / 6;
}

pair<double*, double*> RK_method(double start_pos, double end_pos, double precision, double y0, double z0){
    int n = (end_pos - start_pos) / precision;
    double x[n + 1];
    double* y = new double[n + 1];
    double* z = new double[n + 1];
    x[0] = start_pos;
    y[0] = y0;
    z[0] = z0;
    for (int i = 1; i <= n; ++i){
        y[i] = y[i - 1] + dy(x[i - 1], y[i - 1], z[i - 1], precision);
        z[i] = z[i - 1] + dz(x[i - 1], y[i - 1], z[i - 1], precision);
        x[i] = x[i - 1] + precision;
    }
    return pair<double*, double*>(y, z);
}

double* shooting(double start_pos, double end_pos, double precision, double y0, double eps){
    double nu[3] = {1, 0.8, 0};
    double phi[3] = {0, 0, 0};
    int n = (end_pos - start_pos) / precision;
    pair<double*, double*> R;
    R = RK_method(start_pos, end_pos, precision, y0, nu[0]);
    phi[0] = ugol(R.first[n], R.second[n]);
    R = RK_method(start_pos, end_pos, precision, y0, nu[1]);
    phi[1] = ugol(R.first[n], R.second[n]);
    phi[2] = phi[1];
    while (abs(phi[1]) > eps){
        nu[2] = nu[1] - (nu[1] - nu[0]) / (phi[1] - phi[0]) * phi[1];
        R = RK_method(start_pos, end_pos, precision, y0, nu[2]);
        phi[2] = ugol(R.first[n], R.second[n]);
        nu[0] = nu[1];
        nu[1] = nu[2];
        phi[0] = phi[1];
        phi[1] = phi[2];
    }
    return R.first;
}

double* f_diff(double start_pos, double end_pos, double precision, double y0, double eps){
    int n = (end_pos - start_pos) / precision;
    double X[n + 1];
    for (int i = 0; i <= n; ++i)
        X[i] = start_pos + precision * i;
    double_matrix A_matr = double_matrix(n, 3);
    double B_vect[n];
    A_matr[0][0] = 0;
    A_matr[0][1] = -2 + pow(precision, 2) * q(X[1]);
    A_matr[0][2] = 1 + p(X[1]) * precision / 2;
    B_vect[0] = 0;
    for (int i = 1; i < n - 1; ++i){
        A_matr[i][0] = 1 - p(X[i]) * precision / 2;
        A_matr[i][1] = -2 + pow(precision, 2) * q(X[i]);
        A_matr[i][2] = 1 + p(X[i]) * precision / 2;
        B_vect[i] = 0;
    }
    A_matr[n - 1][0] = -1;
    A_matr[n - 1][1] = 1 + precision;
    A_matr[n - 1][2] = 0;
    B_vect[n - 1] = 5.25 * precision;
    double P[n], Q[n];
    double* Y = new double[n + 1];
    P[0] = -A_matr[0][2] / A_matr[0][1];
    Q[0] = B_vect[0] / A_matr[0][1];
    for (int i = 1; i < n; ++i){
        P[i] = -A_matr[i][2] / (A_matr[i][1] + A_matr[i][0] * P[i - 1]);
        Q[i] = (B_vect[i] - A_matr[i][0] * Q[i - 1]) / (A_matr[i][1] + A_matr[i][0] * P[i - 1]);
    }
    Y[n] = Q[n - 1];
    for (int i = n - 2; i > -1; --i)
        Y[i + 1] = P[i] * Y[i + 2] + Q[i];
    Y[0] = 0;
    return Y;
}

double* deviation(double yt[], double Y[], int n){
    double* eps = new double[n];
    for (int i = 0; i < n; ++i)
        eps[i] = abs(yt[i] - Y[i]);
    return eps;
}

int main(){
    double start_pos = 1, end_pos = 4, y0 = 1, precision = 0.01, eps = 0.001;
    double* answer[5];
    int n = (end_pos - start_pos) / precision;
    double* X = new double[n + 1];
    for (int i = 0; i <= n; ++i)
        X[i] = start_pos + precision * i;
    answer[0] = X;
    double* Y_S = shooting(start_pos, end_pos, precision, y0, eps);
    answer[1] = Y_S;
    double* yfd = f_diff(start_pos, end_pos, precision, y0, eps);
    answer[2] = yfd;
    double yt[n + 1];
    for (int i = 0; i <= n; ++i)
        yt[i] = y(X[i]);
    double* err = deviation(yt, Y_S, n + 1);
    answer[3] = err;
    err = deviation(yt, yfd, n + 1);
    answer[4] = err;
    for (int i = 0; i < 5; ++i){
        for (int j = 0; j <= n; ++j)
            cout << fixed << answer[i][j] << " ";
        cout << "\n";
    }
}