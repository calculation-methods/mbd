#include <iostream>
#include <cmath>
#include <vector>

#define _USE_MATH_DEFINES
#include<math.h>

using namespace std;

// Функция для вычисления факториала
double factorial(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Функция для вычисления Лагранжевого интерполяционного многочлена в точке x
double lagrangeInterpolation(const vector<double>& X, const vector<double>& Y, double x) {
    int n = X.size();
    double result = 0.0;

    for (int i = 0; i < n; ++i) {
        double term = Y[i];
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                term *= (x - X[j]) / (X[i] - X[j]);
            }
        }
        result += term;
    }

    return result;
}

// Функция для вычисления Ньютона интерполяционного многочлена в точке x
double newtonInterpolation(const vector<double>& X, const vector<double>& Y, double x) {
    int n = X.size();
    vector<vector<double>> dividedDifference(n, vector<double>(n));

    // Заполняем начальные значения разделенных разностей
    for (int i = 0; i < n; ++i) {
        dividedDifference[i][0] = Y[i];
    }

    // Вычисляем разделенные разности
    for (int j = 1; j < n; ++j) {
        for (int i = 0; i < n - j; ++i) {
            dividedDifference[i][j] = (dividedDifference[i + 1][j - 1] - dividedDifference[i][j - 1]) / (X[i + j] - X[i]);
        }
    }

    // Вычисляем значение интерполяционного многочлена в точке x
    double result = dividedDifference[0][0];
    double term = 1.0;
    for (int i = 1; i < n; ++i) {
        term *= (x - X[i - 1]);
        result += term * dividedDifference[0][i];
    }

    return result;
}

// Функция для вычисления точного значения функции cos(x) и погрешности
double calculateError(double exactValue, double interpolatedValue) {
    return fabs(exactValue - interpolatedValue);
}

int main() {
    // Задаем точки Xi и значения функции в этих точках Yi
    vector<double> X = { 0, M_PI / 6, 2 * M_PI / 6, 3 * M_PI / 6 };
    
    // Для пункта Б просто меняем значения
    // vector<double> X = { 0, M_PI / 6, 5 * M_PI / 12, M_PI / 2 };
    
    vector<double> Y;

    for (double x : X) {
        Y.push_back(cos(x));
    }

    // Точка, в которой вычисляется интерполяция и погрешность
    double X_star = M_PI / 4;
    double exactValue = cos(X_star);

    // Вычисляем интерполяцию Лагранжа и Ньютона в точке X_star
    double lagrangeValue = lagrangeInterpolation(X, Y, X_star);
    double newtonValue = newtonInterpolation(X, Y, X_star);

    // Вычисляем погрешности
    double lagrangeError = calculateError(exactValue, lagrangeValue);
    double newtonError = calculateError(exactValue, newtonValue);

    // Выводим результаты
    cout << "Lagrange Interpolation Value at X* = " << X_star << ": " << lagrangeValue << endl;
    cout << "Newton Interpolation Value at X* = " << X_star << ": " << newtonValue << endl;
    cout << "Exact Value at X* = " << exactValue << endl;
    cout << "Lagrange Interpolation Error: " << lagrangeError << endl;
    cout << "Newton Interpolation Error: " << newtonError << endl;

    return 0;
}
