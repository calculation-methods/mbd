#include <iostream>
#include<iomanip>
#include <cmath>

using namespace std;

// исходная функция
double f(double x, double y, double z){
    return z / (x - 2) + 3 * y / pow(x - 2, 2);
}

double y(double x){
    return pow(x - 2, 3) + 1 / (x - 2);
}

// метод Эйлера
double* Euler(double a, double b, double h, double y0, double z0){
    int n = (b - a) / h;
    double x[n + 1];
    double* y = new double[n + 1];
    double z[n + 1];
    x[0] = a;
    y[0] = y0;
    z[0] = z0;
    for (int i = 1; i <= n; ++i){
        y[i] = y[i - 1] + h * z[i - 1];
        z[i] = z[i - 1] + h * f(x[i - 1], y[i - 1], z[i - 1]);
        x[i] = x[i - 1] + h;
    }
    return y;
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


// метод Адамса
double* Adams(double a, double b, double h, double y0, double z0){
    int n = (b - a) / h;
    double x[n + 1];
    double* y = new double[n + 1];
    double* z = new double[n + 1];
    pair<double*, double*> yz = Runge_Kutta_4(a, b, h, y0, z0);
    for (int i = 0; i < 4; ++i){
        x[i] = a + h * i;
        y[i] = yz.first[i];
        z[i] = yz.second[i];
    }
    for (int i = 4; i <= n; ++i){
        y[i] = y[i - 1] + h / 24 * (55 * z[i - 1] - 59 * z[i - 2] + 37 * z[i - 3] - 9 * z[i - 4]);
        z[i] = z[i - 1] + h / 24 * (55 * f(x[i - 1], y[i - 1], z[i - 1]) - 59 * f(x[i - 2], y[i - 2], z[i - 2]) + 37 * f(x[i - 3], y[i - 3], z[i - 3]) - 9 * f(x[i - 4], y[i - 4], z[i - 4]));
        x[i] = x[i - 1] + h;
    }
    return y;
}

double* Runge_Romberg(double Y_1[], double Y_2[], int n){
    double* R = new double[n];
    for (int i = 0; i < n; ++i){
        R[i] = (Y_1[i * 2] - Y_2[i]) / (pow(2, 4) - 1);
    }
    return R;
}

double* Error(double Y_t[], double Y[], int n){
    double* eps = new double[n];
    for (int i = 0; i < n; ++i){
        eps[i] = abs(Y_t[i] - Y[i]);
    }
    return eps;
}


int main(){
    double a = 3;
    double b = 4;
    double y0 = 2;
    double z0 = 2;
    double h = 0.1;

    double* Ans[5];
    int n = (b - a) / h;

    double* X = new double[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }
    Ans[0] = X;

    // метод Эйлера
    double* Y_E = Euler(a, b, h, y0, z0);
    Ans[1] = Y_E;

    // метод Рунге-Кутты 4 порядка
    auto [Y_R, Z1] = Runge_Kutta_4(a, b, h, y0, z0);
    auto [Y_R2, Z2] = Runge_Kutta_4(a, b, h * 2, y0, z0);
    Ans[2] = Y_R;

    // метод Адамса
    double* Y_A = Adams(a, b, h, y0, z0);
    Ans[3] = Y_A;

    double Y_t[n + 1];
    for (int i = 0; i <= n; ++i){
        Y_t[i] = y(X[i]);
    }

    // погрещность вычислений Адамса
    double* eps = Error(Y_t, Y_A, n + 1);
    Ans[4] = eps;

    for (int i = 0; i < 5; ++i){
        for (int j = 0; j <= n; ++j){
            cout << fixed << Ans[i][j] << " ";
        }
        cout << "\n";
    }

    // погрешность Рунге-Ромберга
    double* RR = Runge_Romberg(Y_R, Y_R2, n / 2 + 1);
    for (int i = 0; i <= n / 2; ++i){
        cout << RR[i] << " ";
    }
    
}