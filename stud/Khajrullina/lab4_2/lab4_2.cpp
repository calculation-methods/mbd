#include <iostream>
#include <cmath>
#include <vector>

const double e = 0.00001;

double func(double x) {
    return x + 7/2 + 1/x + (x/2 + 1) * log(abs(x));
}

double dfunc(double x, double y, double z) {
    return (-(x+2)*z + y) / (x*(x+1));
}

double g(double y, double z) {
    return 4*z + y - 13 - 4*log(2);
}

double p(double x) {
    return (x + 2) / (x*(x + 1));
}

double q(double x) {
    return -1/ (x*(x+1));
}

double f(double x) {
    return (x + 1 / x)/ (x * (x + 1));
}

double K(double x, double y, double z, double h, int i);

double L(double x, double y, double z, double h, int i) {
    if (i == 0) {
        return h * z;
    }
    else if (i == 3) {
        return h * (z + K(x, y, z, h, i - 1));
    }
    else {
        return h * (z + K(x, y, z, h, i - 1) / 2);
    }
}

double K(double x, double y, double z, double h, int i) {
    if (i == 0) {
        return h * dfunc(x, y, z);
    }
    else if (i == 3) {
        return h * dfunc(x + h, y + L(x, y, z, h, i - 1), z + K(x, y, z, h, i - 1));
    }
    else {
        return h * dfunc(x + h / 2, y + L(x, y, z, h, i - 1) / 2, z + K(x, y, z, h, i - 1) / 2);
    }
}

double delta_z(double x, double y, double z, double h) {
    double d = 0;
    for (int i = 0; i < 4; ++i) {
        if (i == 0 || i == 3) {
            d += K(x, y, z, h, i);
        }
        else {
            d += 2 * K(x, y, z, h, i);
        }
    }
    return d / 6;
}

double delta_y(double x, double y, double z, double h) {
    double d = 0;
    for (int i = 0; i < 4; ++i) {
        if (i == 0 || i == 3) {
            d += L(x, y, z, h, i);
        }
        else {
            d += 2 * L(x, y, z, h, i);
        }
    }
    return d / 6;
}


std::vector<std::vector<double>> Runge_method(double l, double r, double h, double y0, double z0) {
    double x = l;
    int n = (int)((r - l) / h);
    double z = z0;
    double y = y0;
    std::vector<double> res(n + 1);
    std::vector<double> res_z(n + 1);
    res[0] = y0;
    res_z[0] = z0;
    for (int i = 1; i <= n; ++i) {
        double z1 = z + delta_z(x, y, z, h);
        y = y + delta_y(x, y, z, h);
        res[i] = y;
        res_z[i] = z1;
        z = z1;
        x += h;
    }
    return { res, res_z };
}

std::vector<double> Shooting(double l, double r, double h, double z0) {
    double n1 = 1, n2 = 0.8, n3;
    double g1, g2, g3;

    std::vector<std::vector<double>> res1 = Runge_method(l, r, h, n1, z0);
    double res1y = res1[0][res1[0].size() - 1];
    double res1z = res1[1][res1[1].size() - 1];
    std::vector<std::vector<double>> res2 = Runge_method(l, r, h, n2, z0);
    double res2y = res2[0][res2[0].size() - 1];
    double res2z = res2[1][res2[1].size() - 1];
    g1 = g(res1y, res1z);
    g2 = g(res2y, res2z);
    std::vector<std::vector<double>> res;
    while (std::abs(g2) > e) {
        n3 = n2 - (n2 - n1) / (g2 - g1) * g2;
        res = Runge_method(l, r, h, n3, z0);
        double resy = res[0][res[0].size() - 1];
        double res_z = res[1][res[1].size() - 1];
        g3 = g(resy, res_z);
        n1 = n2;
        n2 = n3;
        g1 = g2;
        g2 = g3;
    }
    return res[0];
}

std::vector<double> Difference(double l, double r, double h, double y0, double y1) {
    int n = (int)((r - l) / h);

    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<double> d(n);
    double x = l + h;

    for (int i = 0; i < n; ++i) {
        A[i][i] = -2 + h * h * q(x);
        if (i > 0) A[i][i - 1] = 1 - p(x) * h / 2;
        if (i < n - 1) A[i][i + 1] = (1 + p(x) * h / 2);
        x += h;
    }

    d[0] = h * h * f(l + h) - (1 - p(l + h) * h / 2) * y0;
    d[n - 1] = h * h * f(r - h) - (1 + p(r - h) * h / 2) * y1;
    x = l + 2 * h;
    for (int i = 1; i < n - 1; ++i) {
        d[i] = h * h * f(x);
        x += h;
    }

    std::vector<double> P(n);
    std::vector<double> Q(n);

    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            P[i] = -A[i][i + 1] / A[i][i];
            Q[i] = d[i] / A[i][i];
        }
        else if (i == n - 1) {
            P[i] = 0;
            Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i] + A[i][i - 1] * P[i - 1]);
        }
        else {
            P[i] = -A[i][i + 1] / (A[i][i] + A[i][i - 1] * P[i - 1]);
            Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i] + A[i][i - 1] * P[i - 1]);
        }
    }

    std::vector<double> y(n + 1);


    for (int i = n - 1; i >= 0; --i) {
        if (i == n - 1) y[i] = Q[i];
        else {
            y[i] = P[i] * y[i + 1] + Q[i];
        }
    }
    y[0] = y0;
    y[n] = y1;


    return y;
}



std::vector<double> RR_method(std::vector<double> Y1, std::vector<double> Y2, int n, int p) {
    std::vector<double> R(n);
    for (int i = 0; i < n; ++i) {
        R[i] = (Y1[i * 2] - Y2[i]) / (std::pow(2, p) - 1);
    }
    return R;
}

std::vector<double> Error(std::vector<double> Yt, std::vector<double> Y, int n) {
    std::vector<double> eps(n);
    for (int i = 0; i < n; ++i) {
        eps[i] = std::abs(Yt[i] - Y[i]);
    }
    return eps;
}

int main() {
    double l = 1, r = 2;
    double z0 = 3/2;
    double h = 0.1;
    double y0 = func(l);
    double y1 = func(r);

    std::vector<double> shooting = Shooting(l, r, h, z0);
    std::vector<double> difference = Difference(l, r, h, y0, y1);

    std::vector<double> real(shooting.size());
    double x = l;
    for (int i = 0; i < real.size(); i++) {
        real[i] = func(x);
        x += h;

        std::cout << "Real: " << real[i] << " Shooting: " << shooting[i] << " Difference: " << difference[i] << std::endl;
    }
    
    std::vector<double> shooting2 = Shooting(l, r, h * 2, z0);
    std::vector<double> difference2 = Difference(l, r, h * 2, y0, y1);

    std::vector<double> RShooting = RR_method(shooting, shooting2, real.size() / 2, 2);
    std::vector<double> RDifference = RR_method(difference, difference2, real.size() / 2, 2);

    std::vector<double> EShooting = Error(shooting, real, real.size() / 2);
    std::vector<double> EDifference = Error(difference, real, real.size() / 2);

    for (int i = 0; i < real.size() / 2; i++) {
        std::cout << "Error shooting: " << EShooting[i] << "Runge shooting: " << RShooting[i] << "Error difference: " << EDifference[i] << "Runge difference : " << RDifference[i] << std::endl;
    }
}