#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense> // Для удобного решения линейных систем уравнений
#include <matplotlibcpp.h> // Для построения графиков

namespace plt = matplotlibcpp;

using namespace std;
using namespace Eigen;

// Функция для решения нормальной системы и нахождения коэффициентов
VectorXd solveNormalEquations(const MatrixXd& A, const VectorXd& b) {
    return (A.transpose() * A).ldlt().solve(A.transpose() * b);
}

// Функция для вычисления суммы квадратов ошибок
double calculateError(const vector<double>& x, const vector<double>& y, const VectorXd& coeffs) {
    double error = 0.0;
    int n = x.size();
    for (int i = 0; i < n; ++i) {
        double yi_approx = coeffs[0];
        for (int j = 1; j < coeffs.size(); ++j) {
            yi_approx += coeffs[j] * pow(x[i], j);
        }
        error += pow(y[i] - yi_approx, 2);
    }
    return error;
}

int main() {
    // Данные
    vector<double> x = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
    vector<double> y = {0.86603, 1.0, 0.86603, 0.50, 0.0, -0.50};
    int n = x.size();

    // Многочлен 1-й степени (линейная регрессия)
    MatrixXd A1(n, 2);
    VectorXd b1(n);
    for (int i = 0; i < n; ++i) {
        A1(i, 0) = 1;
        A1(i, 1) = x[i];
        b1[i] = y[i];
    }
    VectorXd coeffs1 = solveNormalEquations(A1, b1);
    double error1 = calculateError(x, y, coeffs1);

    // Многочлен 2-й степени (квадратичная регрессия)
    MatrixXd A2(n, 3);
    VectorXd b2(n);
    for (int i = 0; i < n; ++i) {
        A2(i, 0) = 1;
        A2(i, 1) = x[i];
        A2(i, 2) = x[i] * x[i];
        b2[i] = y[i];
    }
    VectorXd coeffs2 = solveNormalEquations(A2, b2);
    double error2 = calculateError(x, y, coeffs2);

    // Вывод результатов
    cout << "Coefficients for linear polynomial (1st degree): " << coeffs1.transpose() << endl;
    cout << "Sum of squared errors for linear polynomial: " << error1 << endl;
    cout << "Coefficients for quadratic polynomial (2nd degree): " << coeffs2.transpose() << endl;
    cout << "Sum of squared errors for quadratic polynomial: " << error2 << endl;

    // Построение графиков
    vector<double> y1_approx, y2_approx;
    for (double xi : x) {
        double y1i = coeffs1[0] + coeffs1[1] * xi;
        double y2i = coeffs2[0] + coeffs2[1] * xi + coeffs2[2] * xi * xi;
        y1_approx.push_back(y1i);
        y2_approx.push_back(y2i);
    }

    plt::figure();
    plt::plot(x, y, "bo", {{"label", "Data points"}});
    plt::plot(x, y1_approx, "r-", {{"label", "Linear polynomial"}});
    plt::plot(x, y2_approx, "g-", {{"label", "Quadratic polynomial"}});
    plt::legend();
    plt::title("Approximating polynomials using least squares");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::show();

    return 0;
}
