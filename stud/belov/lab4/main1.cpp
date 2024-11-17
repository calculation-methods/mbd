#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Константы задачи
const double h = 0.1;
const int N = 10; // количество шагов (1/h)
const double x0 = 0.0;
const double y0_initial = 1.0; // переименованная переменная
const double dy0_initial = 0.0;

// Правая часть уравнения
double f(double x, double y, double dy) {
    return 2 * cos(x) - y;
}

// Точное решение задачи
std::vector<double> exact_solution(double x) {
    return {x * sin(x) + cos(x), sin(x) + x * cos(x)};
}

// Метод Эйлера
std::vector<double> euler_method() {
    std::vector<double> y(N + 1), dy(N + 1);
    y[0] = y0_initial;
    dy[0] = dy0_initial;
    for (int i = 0; i < N; ++i) {
        double x = x0 + i * h;
        y[i + 1] = y[i] + h * dy[i];
        dy[i + 1] = dy[i] + h * f(x, y[i], dy[i]);
    }
    return y;
}

// Метод Рунге-Кутты 4-го порядка
std::vector<double> runge_kutta_method() {
    std::vector<double> y(N + 1), dy(N + 1);
    y[0] = y0_initial;
    dy[0] = dy0_initial;
    for (int i = 0; i < N; ++i) {
        double x = x0 + i * h;
        double k1 = h * dy[i];
        double l1 = h * f(x, y[i], dy[i]);
        double k2 = h * (dy[i] + 0.5 * l1);
        double l2 = h * f(x + 0.5 * h, y[i] + 0.5 * k1, dy[i] + 0.5 * l1);
        double k3 = h * (dy[i] + 0.5 * l2);
        double l3 = h * f(x + 0.5 * h, y[i] + 0.5 * k2, dy[i] + 0.5 * l2);
        double k4 = h * (dy[i] + l3);
        double l4 = h * f(x + h, y[i] + k3, dy[i] + l3);
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
        dy[i + 1] = dy[i] + (l1 + 2 * l2 + 2 * l3 + l4) / 6.0;
    }
    return y;
}

// Метод Адамса-Башфорта-Мултона
std::vector<double> adams_bashforth_moulton_method() {
    std::vector<double> y = runge_kutta_method(); // Используем метод Рунге-Кутты для начальных значений
    std::vector<double> dy(N + 1);
    dy[0] = dy0_initial;
    for (int i = 0; i < N; ++i) {
        double x = x0 + i * h;
        dy[i + 1] = dy[i] + h * f(x, y[i], dy[i]);
    }
    for (int i = 3; i < N; ++i) {
        double x = x0 + i * h;
        y[i + 1] = y[i] + h * (55 * dy[i] - 59 * dy[i - 1] + 37 * dy[i - 2] - 9 * dy[i - 3]) / 24.0;
        double dy_pred = dy[i] + h * f(x + h, y[i + 1], dy[i + 1]);
        y[i + 1] = y[i] + h * (9 * dy_pred + 19 * dy[i] - 5 * dy[i - 1] + dy[i - 2]) / 24.0;
        dy[i + 1] = dy_pred;
    }
    return y;
}

// Функция для печати результатов
void print_results(const std::vector<double>& y_numeric, const std::string& method) {
    std::cout << method << " результаты:\n";
    std::cout << "x\t\ty_numeric\ty_exact\t\terror\n";
    for (int i = 0; i <= N; ++i) {
        double x = x0 + i * h;
        double y_exact = exact_solution(x)[0];
        std::cout << std::fixed << std::setprecision(6) << x << "\t" << y_numeric[i] << "\t" << y_exact << "\t" << fabs(y_exact - y_numeric[i]) << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Решение методом Эйлера
    std::vector<double> y_euler = euler_method();
    // Решение методом Рунге-Кутты
    std::vector<double> y_runge_kutta = runge_kutta_method();
    // Решение методом Адамса-Башфорта-Мултона
    std::vector<double> y_adams = adams_bashforth_moulton_method();

    // Печать результатов
    print_results(y_euler, "Эйлер");
    print_results(y_runge_kutta, "Рунге-Кутта");
    print_results(y_adams, "Адамс-Башфорт-Мултон");

    return 0;
}
