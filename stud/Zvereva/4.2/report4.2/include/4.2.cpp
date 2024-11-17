#include <iostream>
#include <cmath>
#include <vector>

const double EPSILON = 0.0001;



double func(double x) {
    return exp(x) * x * x;
}

double dfunc(double x, double y, double dy) {
    return ((2 * x + 1) * dy - (x + 1) * y) / x;
}

double g(double y, double dy) {
    return dy - 2 * y;
}

double p(double x) {
    return -2 * (x + 1) / x;
}

double q(double x) {
    return (x + 1) / x;
}

double f(double x) {
    return 0;
}

double K(double x, double y, double dy, double h, int i);

double L(double x, double y, double dy, double h, int i) {
    if (i == 0) {
        return h * dy;
    }
    else if (i == 3) {
        return h * (dy + K(x, y, dy, h, i - 1));
    }
    else {
        return h * (dy + K(x, y, dy, h, i - 1) / 2);
    }
}

double K(double x, double y, double dy, double h, int i) {
    if (i == 0) {
        return h * dfunc(x, y, dy);
    }
    else if (i == 3) {
        return h * dfunc(x + h, y + L(x, y, dy, h, i - 1), dy + K(x, y, dy, h, i - 1));
    }
    else {
        return h * dfunc(x + h / 2, y + L(x, y, dy, h, i - 1) / 2, dy + K(x, y, dy, h, i - 1) / 2);
    }
}

double deltaDY(double x, double y, double dy, double h) {
    double d = 0;
    for (int i = 0; i < 4; ++i) {
        if (i == 0 || i == 3) {
            d += K(x, y, dy, h, i);
        }
        else {
            d += 2 * K(x, y, dy, h, i);
        }
    }
    return d / 6;
}

double deltaY(double x, double y, double dy, double h) {
    double d = 0;
    for (int i = 0; i < 4; ++i) {
        if (i == 0 || i == 3) {
            d += L(x, y, dy, h, i);
        }
        else {
            d += 2 * L(x, y, dy, h, i);
        }
    }
    return d / 6;
}


std::vector<std::vector<double>> Runge(double l, double r, double h, double y0, double dy0) {
    double x = l;
    int n = (int)((r - l) / h);
    double dy = dy0;
    double y = y0;
    std::vector<double> res(n + 1);
    std::vector<double> resDY(n + 1);
    res[0] = y0;
    resDY[0] = dy0;
    for (int i = 1; i <= n; ++i) {
        double dy1 = dy + deltaDY(x, y, dy, h);
        y = y + deltaY(x, y, dy, h);
        res[i] = y;
        resDY[i] = dy1;
        dy = dy1;
        x += h;
    }
    return { res, resDY };
}

std::vector<double> Shooting(double l, double r, double h, double dy0) {
    double n1 = 1, n2 = 0.8, n3;
    double g1, g2, g3;

    std::vector<std::vector<double>> res1 = Runge(l, r, h, n1, dy0);
    double res1y = res1[0][res1[0].size() - 1];
    double res1dy = res1[1][res1[1].size() - 1];
    std::vector<std::vector<double>> res2 = Runge(l, r, h, n2, dy0);
    double res2y = res2[0][res2[0].size() - 1];
    double res2dy = res2[1][res2[1].size() - 1];
    g1 = g(res1y, res1dy);
    g2 = g(res2y, res2dy);
    std::vector<std::vector<double>> res;
    while (std::abs(g2) > EPSILON) {
        n3 = n2 - (n2 - n1) / (g2 - g1) * g2;
        res = Runge(l, r, h, n3, dy0);
        double resy = res[0][res[0].size() - 1];
        double resdy = res[1][res[1].size() - 1];
        g3 = g(resy, resdy);
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

    //метод прогонки
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



std::vector<double> RungeRomberg(std::vector<double> Y1, std::vector<double> Y2, int n, int p) {
    std::vector<double> R(n);
    for (int i = 0; i < n; ++i) {
        R[i] = (Y1[i * 2] - Y2[i]) / (std::pow(2, p) - 1);
    }
    return R;
}

std::vector<double> Loss(std::vector<double> Yt, std::vector<double> Y, int n) {
    std::vector<double> eps(n);
    for (int i = 0; i < n; ++i) {
        eps[i] = std::abs(Yt[i] - Y[i]);
    }
    return eps;
}

int main() {
    double l = 1, r = 2;
    double dy0 = 3 * 2.71;
    double h = 0.1;
    double y0 = func(l);
    double y1 = func(r);

    std::vector<double> shooting = Shooting(l, r, h, dy0);
    std::vector<double> difference = Difference(l, r, h, y0, y1);

    std::vector<double> real(shooting.size());
    double x = l;
    for (int i = 0; i < real.size(); i++) {
        real[i] = func(x);
        x += h;

        std::cout << "Real " << real[i] << " Shooting " << shooting[i] << " Difference " << difference[i] << std::endl;
    }
    std::cout << "================================================================" << std::endl;

    std::vector<double> shooting2 = Shooting(l, r, h * 2, dy0);
    std::vector<double> difference2 = Difference(l, r, h * 2, y0, y1);

    std::vector<double> RungeShooting = RungeRomberg(shooting, shooting2, real.size() / 2, 2);
    std::vector<double> RungeDifference = RungeRomberg(difference, difference2, real.size() / 2, 2);

    std::vector<double> ShootingLoss = Loss(shooting, real, real.size() / 2);
    std::vector<double> DifferenceLoss = Loss(difference, real, real.size() / 2);

    for (int i = 0; i < real.size() / 2; i++) {
        std::cout << "Loss Shooting " << ShootingLoss[i] << " Runge Shooting " << RungeShooting[i] << " Loss Difference " << DifferenceLoss[i] << " Runge Difference " << RungeDifference[i] << std::endl;
    }
}

