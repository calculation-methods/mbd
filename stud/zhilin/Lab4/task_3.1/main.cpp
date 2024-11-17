#include <bits/stdc++.h>

using namespace std;
using d_vect = vector<double >;

double K(double x, double y, double dy, double h, int i);
double L(double x, double y, double dy, double h, int i);
double d_f(double x, double y, double dy);
double correct_f(double x);
double diff_dy(double x, double y, double dy, double h);
double diffY(double x, double y, double dy, double h);
d_vect eiler_algo(double l, double r, double h, double y0, double dy0);
vector<d_vect> runge_algo(double l, double r, double h, double y0, double dy0);
d_vect adams(double l, double r, double h, double y0, double dy0);


double d_f(double x, double y, double dy) {
    return dy/pow(x, 1/2) - (x + pow(x, 1/2) - 8)*y/(4*x*x);
}

double correct_f(double x) {
    return (x*x + 1/x)*exp(pow(x, 1/2));
}

double L(double x, double y, double dy, double h, int i){
    if (i == 0)
        return h * dy;
    else if (i == 3)
        return h * (dy + K(x, y, dy, h, i - 1));
    else
        return h * (dy + K(x, y, dy, h, i - 1) / 2);
}

double K(double x, double y, double dy, double h, int i){
    if (i == 0)
        return h * d_f(x, y, dy);
    else if (i == 3)
        return h * d_f(x + h, y + L(x, y, dy, h, i - 1), dy + K(x, y, dy, h, i-1));
    else
        return h * d_f(x + h / 2, y + L(x, y, dy, h, i - 1) / 2, dy + K(x, y, dy, h, i-1) / 2);
}

double diffY(double x, double y, double dy, double h){
    double d = 0;
    for (int i = 0; i < 4; i++)
        if (i == 0 || i == 3)
            d += L(x, y, dy, h, i);
        else
            d += 2 *L(x, y, dy, h, i);
    return d / 6;
}

double diff_dy(double x, double y, double dy, double h){
    double d = 0;
    for (int i = 0; i < 4; i++)
        if (i == 0 || i == 3)
            d += K(x, y, dy, h, i);
        else
            d += 2 * K(x, y, dy, h, i);
    return d / 6;
}

d_vect eiler_algo(double l, double r, double h, double y0, double dy0) {
    double x = l, dy = dy0, y = y0;
    int n = static_cast<int>((r - l) / h);
    d_vect res(n);
    res[0] = y0;
    for(int i = 0; i < n-1; i++) {
        double dy1 = dy + h * d_f(x, y, dy);
        y = y + h * dy;
        res[i+1] = y;
        dy = dy1;
        x += h;
    }
    return res;
}

vector<d_vect> runge_algo(double l, double r, double h, double y0, double dy0) {
    double x = l, dy = dy0, y = y0;
    int n = static_cast<int>((r - l) / h);
    d_vect res(n), res_dy(n);
    res[0] = y0;
    res_dy[0] = dy0;
    for(int i = 0; i < n-1; i++) {
        double dy1 = dy + diff_dy(x, y, dy, h);
        y = y +  diffY(x, y,dy, h);
        res[i+1] = y;
        res_dy[i+1] = dy1;
        dy = dy1;
        x += h;
    }
    return {res, res_dy};
}

d_vect adams(double l, double r, double h, double y0, double dy0) {
    int n = static_cast<int>((r - l) / h);
    d_vect res(n), res_dy(n), x(n);
    auto runge = runge_algo(l, r, h, y0, dy0);
    for(int i = 0; i < 4; i++) {
        x[i] = l + h * i;
        res[i] = runge[0][i];
        res_dy[i] = runge[1][i];
    }
    for (int i = 4; i < n; i++){
        res[i] = res[i - 1] + h / 24 * (55 * res_dy[i - 1] - 59 * res_dy[i - 2] + 37 * res_dy[i - 3] - 9 * res_dy[i - 4]);
        res_dy[i] = res_dy[i - 1] + h / 24 * (55 * d_f(x[i - 1], res[i - 1], res_dy[i-1]) - 59 * d_f(x[i - 2], res[i - 2], res_dy[i-2]) + 37 * d_f(x[i - 3], res[i - 3], res_dy[i-3]) - 9 * d_f(x[i - 4], res[i - 4], res_dy[i-4]));
        x[i] = x[i - 1] + h;
    }
    return res;
}

d_vect RR(d_vect Y1, d_vect Y2, int n, int p){
    d_vect R(n);
    for (int i = 0; i < n; i++)
        R[i] = (Y1[i * 2] - Y2[i]) / (pow(2, p) - 1);
    return R;
}

d_vect deviation(d_vect y_t, d_vect Y, int n){
    d_vect eps(n);
    for (int i = 0; i < n; i++)
        eps[i] = abs(y_t[i] - Y[i]);
    return eps;
}

int main() {
    double h = 0.1, l = 1, r = 2, y0 = 2*exp(1), dy0 = 2*exp(1);
    int n = static_cast<int>((r - l) / h);

    d_vect res_eiler_ = eiler_algo(l, r, h, y0, dy0), res_eiler_2 = eiler_algo(l, r, 2*h, y0, dy0);
    d_vect resrunge_algo = runge_algo(l, r, h, y0, dy0)[0];
    d_vect resrunge_algo2 = runge_algo(l, r, 2*h, y0, dy0)[0];
    d_vect resadams = adams(l, r, h, y0, dy0);
    d_vect resadams2 = adams(l, r, 2*h, y0, dy0);

    double num = l;
    int iter = 0;
    d_vect realY(n);
    while (num < r-h) {
        realY[iter] = correct_f(num);
        cout << "eiler_algo: " << res_eiler_[iter] << " real func: " << correct_f(num) << " runge_algo: "<<resrunge_algo[iter] << " adams: "<<resadams[iter] << endl;
        num+=h;
        iter+=1;
    }
    
    d_vect r_eiler_ = RR(res_eiler_, res_eiler_2, n / 2, 2), rrunge_algo = RR(resrunge_algo, resrunge_algo2, n / 2, 5);
    d_vect radams = RR(resadams, resadams2, n / 2, 5), l_eiler_ = deviation(realY, res_eiler_, n);
    d_vect lrunge_algo = deviation(realY, resrunge_algo, n), ladams = deviation(realY, resadams, n);

    for(int i = 0; i < r_eiler_.size(); i++)
        cout << "eiler_algo: " << r_eiler_[i] << " runge_algo: "<<rrunge_algo[i] << " adams: "<<radams[i] << endl;
    cout << endl << endl;
    for(int i = 0; i < l_eiler_.size(); i++)
        cout << "eiler_algo: " << l_eiler_[i] << " runge_algo: "<<lrunge_algo[i] << " adams: "<<ladams[i] << endl;
}