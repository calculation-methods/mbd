#include <iostream>
#include <cmath>

double f(double x) {
    return pow(x,2) * pow (36 - pow(x,2), 0.5);
}

double rectangle_method(double a, double b, double h) {
    double res = 0;
    while (a + h < b) {
        res += f(a + h / 2);
        a += h;
    }
    return res * h;
}

double trapez_method(double a, double b, double h) {
    double res = 0;
    while (a + h < b) {
        res += (f(a + h) + f(a));
        a += h;
    }
    return res * h * 0.5;
}

double simpson_method(double a, double b, double h) {
    double res = 0;
    a += h;
    while (a + h < b) {
        res += f(a - h) + 4 * f(a - h / 2) + f(a);
        a += h;
    }
    return res * h / 6;
}

double rungeRombert(double h1, double h2, double i1, double i2, double p) {
    return i1 + (i1 - i2) / (pow((h2 / h1), p) - 1);
}

int main() {
    double a = 1;
    double b = 5;
    double h1 = 1.0;
    double h2 = 0.5;
    std::cout << "h = " << h1 << "\n";
    std::cout << "Integral rectangle: " << rectangle_method(a, b, h1) << "\n";
    std::cout << "Integral trapeze: " << trapez_method(a, b, h1) << "\n";
    std::cout << "Integral simpson: " << simpson_method(a, b, h1) << "\n";
    std::cout << "h = " << h2 << "\n";
    std::cout << "Integral rectangle: " << rectangle_method(a, b, h2) << "\n";
    std::cout << "Integral trapeze: " << trapez_method(a, b, h2) << "\n";
    std::cout << "Integral simpson: " << simpson_method(a, b, h2) << "\n";
    std::cout << "Runge Rombert:\n";
    std::cout << "Rombert rectangle: " << rungeRombert(h1, h2, rectangle_method(a, b, h1), rectangle_method(a, b, h2), 1) << "\n";
    std::cout << "Rombert trapeze: " << rungeRombert(h1, h2, trapez_method(a, b, h1), trapez_method(a, b, h2), 2) << "\n";
    std::cout << "Rombert simpson: " << rungeRombert(h1, h2, simpson_method(a, b, h1), simpson_method(a, b, h2), 4) << "\n";

    return 0;
}