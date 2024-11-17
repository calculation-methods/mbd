#include <bits/stdc++.h>

using namespace std;

const double PI = 3.1415926535; 

double y(double x){
    return x*cos(x) + x*sin(x);
}

double f(double x, double y, double z){
    return (2*x*z - x*x*y)/(x*x + 2);
}

double* method_eiler(double start_pos, double end_pos, double precision, double y_0, double z_0){
    int n = (end_pos - start_pos) / precision;
    double x[n + 1];
    double* y = new double[n + 1];
    double z[n + 1];
    x[0] = start_pos;
    y[0] = y_0;
    z[0] = z_0;
    for (int i = 1; i <= n; i++){
        y[i] = y[i - 1] + precision * z[i - 1];
        z[i] = z[i - 1] + precision * f(x[i - 1], y[i - 1], z[i - 1]);
        x[i] = x[i - 1] + precision;
    }
    return y;
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
    for (int i = 0; i < 4; i++){
        if (i == 0 || i == 3)
            d += K(x, y, z, precision, i);
        else
            d += 2 * K(x, y, z, precision, i);
    }
    return d / 6;
}

double dy(double x, double y, double z, double precision){
    double d = 0;
    for (int i = 0; i < 4; i++)
        if (i == 0 || i == 3)
            d += L(x, y, z, precision, i);
        else
            d += 2 *L(x, y, z, precision, i);
    return d / 6;
}

pair<double*, double*> RK_method(double start_pos, double end_pos, double precision, double y_0, double z_0){
    int n = (end_pos - start_pos) / precision;
    double x[n + 1];
    double* y = new double[n + 1];
    double* z = new double[n + 1];
    x[0] = start_pos;
    y[0] = y_0;
    z[0] = z_0;
    for (int i = 1; i <= n; i++){
        y[i] = y[i - 1] + dy(x[i - 1], y[i - 1], z[i - 1], precision);
        z[i] = z[i - 1] + dz(x[i - 1], y[i - 1], z[i - 1], precision);
        x[i] = x[i - 1] + precision;
    }
    return pair<double*, double*>(y, z);
}

double* method_adams(double start_pos, double end_pos, double precision, double y_0, double z_0){
    int n = (end_pos - start_pos) / precision;
    double x[n + 1];
    double* y = new double[n + 1];
    double* z = new double[n + 1];
    pair<double*, double*> yz = RK_method(start_pos, end_pos, precision, y_0, z_0);
    for (int i = 0; i < 4; i++){
        x[i] = start_pos + precision * i;
        y[i] = yz.first[i];
        z[i] = yz.second[i];
    }
    for (int i = 4; i <= n; i++){
        y[i] = y[i - 1] + precision / 24 * (55 * z[i - 1] - 59 * z[i - 2] + 37 * z[i - 3] - 9 * z[i - 4]);
        z[i] = z[i - 1] + precision / 24 * (55 * f(x[i - 1], y[i - 1], z[i - 1]) - 59 * f(x[i - 2], y[i - 2], z[i - 2]) + 37 * f(x[i - 3], y[i - 3], z[i - 3]) - 9 * f(x[i - 4], y[i - 4], z[i - 4]));
        x[i] = x[i - 1] + precision;
    }
    return y;
}

double* RR_method(double y1[], double y2[], int n){
    double* R = new double[n];
    for (int i = 0; i < n; i++){
        R[i] = (y1[i * 2] - y2[i]) / (pow(2, 4) - 1);
    }
    return R;
}

double* deviation(double yt[], double Y[], int n){
    double* eps = new double[n];
    for (int i = 0; i < n; i++)
        eps[i] = abs(yt[i] - Y[i]);
    return eps;
}

int main(){
    double start_pos = PI/2, end_pos = PI/2 + 1, y_0 = PI/2, z_0 = 1 - PI/2, precision = 0.1;
    double* answer[5];
    int n = (end_pos - start_pos) / precision;
    double* X = new double[n + 1];
    for (int i = 0; i <= n; i++)
        X[i] = start_pos + precision * i;
    answer[0] = X;
    double* ye = method_eiler(start_pos, end_pos, precision, y_0, z_0);
    answer[1] = ye;
    auto [yr, Z1] = RK_method(start_pos, end_pos, precision, y_0, z_0);
    auto [yr2, Z2] = RK_method(start_pos, end_pos, precision * 2, y_0, z_0);
    answer[2] = yr;
    double* ya = method_adams(start_pos, end_pos, precision, y_0, z_0);
    answer[3] = ya;
    double yt[n + 1];
    for (int i = 0; i <= n; i++)
        yt[i] = y(X[i]);
    double* eps = deviation(yt, ya, n + 1);
    answer[4] = eps;
    for (int i = 0; i < 5; i++){
        for (int j = 0; j <= n; ++j)
            cout << fixed << answer[i][j] << " ";
        cout << "\n";
    }
    double* RR = RR_method(yr, yr2, n / 2 + 1);
    for (int i = 0; i <= n / 2; i++)
        cout << RR[i] << " ";
}