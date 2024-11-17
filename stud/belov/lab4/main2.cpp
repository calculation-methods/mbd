#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

const double h = 0.1;
const int N = (2 - 1) / h; // количество шагов (1/h)
const double x1 = 1.0;
const double x2 = 2.0;
const double y_initial = exp(-1); // переименованная переменная
const double y_final = 0.5 * exp(-2);

double exact_solution(double x) {
    return exp(-x) / x;
}

// Функции для метода стрельбы
double f1(double x, double y, double dy) {
    return dy;
}

double f2(double x, double y, double dy) {
    return (x * y - 2 * dy) / x;
}

std::vector<double> shooting_method(double alpha) {
    std::vector<double> y(N + 1), dy(N + 1);
    y[0] = y_initial;
    dy[0] = alpha;
    for (int i = 0; i < N; ++i) {
        double x = x1 + i * h;
        double k1_y = h * f1(x, y[i], dy[i]);
        double k1_dy = h * f2(x, y[i], dy[i]);
        double k2_y = h * f1(x + 0.5 * h, y[i] + 0.5 * k1_y, dy[i] + 0.5 * k1_dy);
        double k2_dy = h * f2(x + 0.5 * h, y[i] + 0.5 * k1_y, dy[i] + 0.5 * k1_dy);
        double k3_y = h * f1(x + 0.5 * h, y[i] + 0.5 * k2_y, dy[i] + 0.5 * k2_dy);
        double k3_dy = h * f2(x + 0.5 * h, y[i] + 0.5 * k2_y, dy[i] + 0.5 * k2_dy);
        double k4_y = h * f1(x + h, y[i] + k3_y, dy[i] + k3_dy);
        double k4_dy = h * f2(x + h, y[i] + k3_y, dy[i] + k3_dy);
        y[i + 1] = y[i] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6.0;
        dy[i + 1] = dy[i] + (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6.0;
    }
    return y;
}

std::vector<double> finite_difference_method() {
    std::vector<double> a(N + 1, 0), b(N + 1, 0), c(N + 1, 0), d(N + 1, 0), y(N + 1, 0);
    double x;
    for (int i = 1; i < N; ++i) {
        x = x1 + i * h;
        a[i] = 1.0 - h / (2 * x);
        b[i] = -(2.0 + h * h);
        c[i] = 1.0 + h / (2 * x);
        d[i] = 0; // Поправлено: значение правой части уравнения
    }
    d[0] = y_initial;
    d[N] = y_final;

    // Прямой ход метода прогонки
    for (int i = 1; i <= N; ++i) {
        double m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        d[i] = d[i] - m * d[i - 1];
    }

    // Обратный ход метода прогонки
    y[N] = d[N] / b[N];
    for (int i = N - 1; i >= 0; --i) {
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i];
    }
    return y;
}

void print_results(const std::vector<double>& y_numeric, const std::string& method) {
    std::cout << method << " результаты:\n";
    std::cout << "x\t\ty_numeric\ty_exact\t\terror\n";
    for (int i = 0; i <= N; ++i) {
        double x = x1 + i * h;
        double y_exact = exact_solution(x);
        std::cout << std::fixed << std::setprecision(6) << x << "\t" << y_numeric[i] << "\t" << y_exact << "\t" << fabs(y_exact - y_numeric[i]) << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Метод стрельбы
    double alpha = -1.0; // Начальное предположение для y'
    std::vector<double> y_shooting = shooting_method(alpha);

    // Конечно-разностный метод
    std::vector<double> y_finite_difference = finite_difference_method();

    // Печать результатов
    print_results(y_shooting, "Метод стрельбы");
    print_results(y_finite_difference, "Конечно-разностный метод");

    return 0;
}
