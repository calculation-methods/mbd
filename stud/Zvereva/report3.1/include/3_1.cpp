#include <iostream>
#include <cmath>
#include <vector>

std::vector<double> xLag = { -3, -1, 1, 3 }; //лагранж
std::vector<double> xNew = { -3, 0, 1, 3 }; //ньютон
double X = -0.5;

double func(double x) {
    return atan(x);
}

std::vector<double> getY(std::vector<double> x) {
    std::vector<double> y(x.size());
    for (int i = 0; i < x.size(); ++i) {
        y[i] = func(x[i]);
    }
    return y;
}

std::vector<std::vector<double>> getDelta(std::vector<double> x, std::vector<double> y) {
    int n = y.size();
    std::vector<std::vector<double>> delta(n, std::vector<double>(n));

    for (int j = 0; j < n; ++j) {
        for (int i = n - j - 1; i >= 0; --i) {
            if (j == 0) delta[i][j] = y[i];
            else {
                delta[i][j] = (delta[i][j - 1] - delta[i + 1][j - 1]) / (x[i] - x[i + j]);
            }
        }
    }
    return delta;
}

double getNewton(double x, std::vector<double> xa) {
    std::vector<double> y = getY(xa);
    std::vector<std::vector<double>> delta = getDelta(xa, y);
    double result = 0;
    for (int j = 0; j < y.size(); ++j) {
        double mult = 1;
        for (int i = 0; i < j; ++i) {
            mult *= (x - xa[i]);
        }
        result += mult * delta[0][j];
    }
    return result;
}

double getLagrangian(double x, std::vector<double> xa) {
    std::vector<double> y = getY(xa);
    double result = 0;
    for (int j = 0; j < y.size(); ++j) {
        double mult = 1;
        for (int i = 0; i < y.size(); ++i) {
            if (i != j) mult *= (x - xa[i]) / (xa[j] - xa[i]);
        }
        result += mult * y[j];
    }
    return result;
}

int main() {
    std::cout << "Lagrangian inaccuracy: " << std::abs(getLagrangian(X, xLag) - func(X)) << std::endl;
    std::cout << "Newton inaccuracy: " << std::abs(getNewton(X, xNew) - func(X)) << std::endl;
    return 0;
}
