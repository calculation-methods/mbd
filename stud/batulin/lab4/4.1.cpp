#include <iostream>
#include<iomanip>
#include <cmath>

using namespace std;

double init(double x, double y, double z){
    return -y + sin(3 * x);
}

double exsol(double x){
    return cos(x) + 11. / 8. * sin(x) - sin(3 * x) / 8.;
}

double* E(double a, double b, double h, double y0, double z0){
    int n = (b - a) / h;
    double x[n + 1];
    double* y = new double[n + 1];
    double z[n + 1];
    x[0] = a;
    y[0] = y0;
    z[0] = z0;
    for (int i = 1; i <= n; ++i){
        y[i] = y[i - 1] + h * z[i - 1];
        z[i] = z[i - 1] + h * init(x[i - 1], y[i - 1], z[i - 1]);
        x[i] = x[i - 1] + h;
    }
    return y;
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

double* A(double a, double b, double h, double y0, double z0){
    int n = (b - a) / h;
    double x[n + 1];
    double* y = new double[n + 1];
    double* z = new double[n + 1];
    pair<double*, double*> yz = RK4(a, b, h, y0, z0);
    for (int i = 0; i < 4; ++i){
        x[i] = a + h * i;
        y[i] = yz.first[i];
        z[i] = yz.second[i];
    }
    for (int i = 4; i <= n; ++i){
        y[i] = y[i - 1] + h / 24 * (55 * z[i - 1] - 59 * z[i - 2] + 37 * z[i - 3] - 9 * z[i - 4]);
        z[i] = z[i - 1] + h / 24 * (55 * init(x[i - 1], y[i - 1], z[i - 1]) - 59 * init(x[i - 2], y[i - 2], z[i - 2]) + 37 * init(x[i - 3], y[i - 3], z[i - 3]) - 9 * init(x[i - 4], y[i - 4], z[i - 4]));
        x[i] = x[i - 1] + h;
    }
    return y;
}

double* RR(double Y_1[], double Y_2[], int n){
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
    double a = 0;
    double b = 1;
    double y0 = 1;
    double z0 = 1;
    double h = 0.1;

    double* Ans[5];
    int n = (b - a) / h;

    double* X = new double[n + 1];
    for (int i = 0; i <= n; ++i){
        X[i] = a + h * i;
    }
    Ans[0] = X;

    double* Y_E = E(a, b, h, y0, z0);
    Ans[1] = Y_E;

    auto [Y_R, Z1] = RK4(a, b, h, y0, z0);
    Ans[2] = Y_R;

    double* Y_A = A(a, b, h, y0, z0);
    Ans[3] = Y_A;

    double Y_t[n + 1];
    for (int i = 0; i <= n; ++i){
        Y_t[i] = exsol(X[i]);
    }

    double* eps = Error(Y_t, Y_R, n + 1);
    Ans[4] = eps;

    for (int i = 0; i < 5; ++i){
        for (int j = 0; j <= n; ++j){
            cout << fixed << Ans[i][j] << " ";
        }
        cout << "\n";
    }
}
