#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

// Функция для вычисления центральной первой производной
double centralFirstDerivative(const vector<double>& x, const vector<double>& y, int i) {
    return (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
}

// Функция для вычисления левого дифференциала первой производной
double leftFirstDerivative(const vector<double>& x, const vector<double>& y, int i) {
    if (i == 0) {
        cerr << "Error: Left derivative not defined for the first point." << endl;
        return 0.0;
    }
    return (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
}

// Функция для вычисления правого дифференциала первой производной
double rightFirstDerivative(const vector<double>& x, const vector<double>& y, int i) {
    if (i == x.size() - 1) {
        cerr << "Error: Right derivative not defined for the last point." << endl;
        return 0.0;
    }
    return (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
}

// Функция для вычисления второй производной
double secondDerivative(const vector<double>& x, const vector<double>& y, int i) {
    if (i == 0 || i == x.size() - 1) {
        cerr << "Error: Second derivative not defined for the first or last point." << endl;
        return 0.0;
    }
    return (y[i + 1] - 2 * y[i] + y[i - 1]) / ((x[i] - x[i - 1]) * (x[i] - x[i - 1]));
}

int main() {
    vector<double> x = { -1.0, 0.0, 1.0, 2.0, 3.0 };
    vector<double> y = { -0.5, 0.0, 0.5, 0.86603, 1.0 };
    double x_star = 1.0;
    int i = 2; // индекс точки x* = 1.0 в массиве x

    // Вычисление первой и второй производных
    double central_first_derivative = centralFirstDerivative(x, y, i);
    double left_first_derivative = leftFirstDerivative(x, y, i);
    double right_first_derivative = rightFirstDerivative(x, y, i);
    double second_derivative = secondDerivative(x, y, i);

    // Вывод результатов
    cout << "Central first derivative at x* = " << x_star << ": " << fixed << setprecision(5) << central_first_derivative << endl;
    cout << "Left first derivative at x* = " << x_star << ": " << fixed << setprecision(5) << left_first_derivative << endl;
    cout << "Right first derivative at x* = " << x_star << ": " << fixed << setprecision(5) << right_first_derivative << endl;
    cout << "Second derivative at x* = " << x_star << ": " << fixed << setprecision(5) << second_derivative << endl;

    return 0;
}
