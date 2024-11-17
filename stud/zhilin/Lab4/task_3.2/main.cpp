#include <bits/stdc++.h>

using namespace std;
using d_vect = vector<double >;

const double precision = 0.0001;

double func(double x) {
    return exp(x) * (x * x + 1);
}

double dfunc(double x, double y, double dy) {
    return ((2*x+1)*dy - (x+1)*y) / x;
}

double g(double y, double dy) {
    return dy - 2*y;
}

double p(double x) {
    return -2*(x+1) / x;
}

double q(double x) {
    return (x+1) / x;
}

double f(double x) {
    return 0;
}

double K(double x, double y, double dy, double h, int i);

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
        return h * dfunc(x, y, dy);
    else if (i == 3)
        return h * dfunc(x + h, y + L(x, y, dy, h, i - 1), dy + K(x, y, dy, h, i-1));
    else
        return h * dfunc(x + h / 2, y + L(x, y, dy, h, i - 1) / 2, dy + K(x, y, dy, h, i-1) / 2);
}

double diff_DY(double x, double y, double dy, double h){
    double d = 0;
    for (int i = 0; i < 4; ++i)
        if (i == 0 || i == 3)
            d += K(x, y, dy, h, i);
        else
            d += 2 * K(x, y, dy, h, i);
    return d / 6;
}

double diff_Y(double x, double y, double dy, double h){
    double d = 0;
    for (int i = 0; i < 4; ++i)
        if (i == 0 || i == 3)
            d += L(x, y, dy, h, i);
        else
            d += 2 *L(x, y, dy, h, i);
    return d / 6;
}


vector<d_vect> rung(double l, double r, double h, double y0, double dy0) {
    double x = l;
    int n = (int)((r - l) / h);
    double dy = dy0, y = y0;
    d_vect res(n+1), resDY(n+1);
    res[0] = y0;
    resDY[0] = dy0;
    for(int i = 1; i <= n; ++i) {
        double dy1 = dy + diff_DY(x, y, dy, h);
        y = y +  diff_Y(x, y,dy, h);
        res[i] = y;
        resDY[i] = dy1;
        dy = dy1;
        x += h;
    }
    return {res, resDY};
}

d_vect shooting_algo(double l, double r, double h, double dy0) {
    double n1 = 1, n2 = 0.8, n3, g1, g2, g3;
    vector<d_vect> res1 = rung(l, r, h, n1, dy0);
    double res1y = res1[0][res1[0].size()-1], res1dy = res1[1][res1[1].size()-1];
    vector<d_vect> res2 = rung(l, r, h, n2, dy0);
    double res2y = res2[0][res2[0].size()-1], res2dy = res2[1][res2[1].size()-1];
    g1 = g(res1y, res1dy);
    g2 = g(res2y, res2dy);
    vector<d_vect> res;
    while (abs(g2) > precision) {
        n3 = n2 - (n2 - n1) / (g2 - g1) * g2;
        res = rung(l, r, h, n3, dy0);
        double resy = res[0][res[0].size()-1], resdy = res[1][res[1].size()-1];
        g3 = g(resy, resdy);
        n1 = n2;
        n2 = n3;
        g1 = g2;
        g2 = g3;
    }
    return res[0];
}

d_vect difference_method(double l, double r, double h, double y0, double y1) {
    int n = (int) ((r - l) / h);
    vector<d_vect> A(n, d_vect(n));
    d_vect d(n);
    double x = l+h;
    for (int i = 0; i < n; ++i) {
        A[i][i] = -2 + h * h * q(x);
        if (i > 0) A[i][i - 1] = 1 - p(x) * h / 2;
        if (i < n - 1) A[i][i + 1] = (1 + p(x) * h / 2);
        x+=h;
    }
    d[0] = h * h * f(l+h) - (1 - p(l+h) * h / 2) * y0;
    d[n - 1] = h * h * f(r-h) - (1 + p(r-h) * h / 2) * y1;
    x = l+2*h;
    for (int i = 1; i < n - 1; ++i) {
        d[i] = h * h * f(x);
        x+=h;
    }
    d_vect P(n), Q(n);
    for(int i = 0; i < n; ++i) {
        if(i == 0) {
            P[i] = -A[i][i+1] / A[i][i];
            Q[i] = d[i] / A[i][i];
        } else if(i == n - 1) {
            P[i] = 0;
            Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i] + A[i][i - 1] * P[i - 1]);
        } else {
            P[i] = -A[i][i+1] / (A[i][i] + A[i][i - 1] * P[i - 1]);
            Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i] + A[i][i - 1] * P[i - 1]);
        }
    }
    d_vect y(n+1);
    for(int i = n - 1; i >= 0; --i) {
        if(i == n - 1) y[i] = Q[i];
        else y[i] = P[i] * y[i+1] + Q[i];
    }
    y[0] = y0;
    y[n] = y1;
    return y;
}

d_vect RR(d_vect Y1, d_vect Y2, int n, int p){
    d_vect R(n);
    for (int i = 0; i < n; ++i)
        R[i] = (Y1[i * 2] - Y2[i]) / (pow(2, p) - 1);
    return R;
}

d_vect deviation(d_vect Yt, d_vect Y, int n){
    d_vect eps(n);
    for (int i = 0; i < n; ++i)
        eps[i] = abs(Yt[i] - Y[i]);
    return eps;
}

int main() {
    double l = 1, r = 2, dy0 = 1, h = 0.1;
    double y0 = func(l), y1 = func(r);

    d_vect shooting = shooting_algo(l, r, h, dy0), difference = difference_method(l, r, h, y0, y1);
    d_vect real(shooting.size());

    double x = l;
    for(int i = 0; i < real.size(); i++) {
        real[i] = func(x);
        x+=h;
        cout << "correct function " << real[i] << " shooting_algo " << shooting[i] << " difference_method " << difference[i] << endl;
    }
    cout << endl;
    d_vect shooting2 = shooting_algo(l, r, h*2, dy0), difference2 = difference_method(l, r, h*2, y0, y1);
    d_vect runge_shoot = RR(shooting, shooting2, real.size() / 2, 2), runge_diff = RR(difference, difference2, real.size() / 2, 2);
    d_vect shoot_deviation = deviation(shooting, real, real.size() / 2), diff_deviation = deviation(difference, real, real.size() / 2);
    for(int i =0; i < real.size() / 2; i++)
        cout << "deviation shooting_algo " << shoot_deviation[i] << " rung shooting_algo " << runge_shoot[i] << " deviation difference_method " << diff_deviation[i] << " rung difference_method " << runge_diff[i] << endl;
}