#include <iostream>
#include <cmath>

double func(double x) {
    return x*x / (x*x + 16);
}

double rect(double a, double b, double h) {
    double sum = 0;
    while (a + h < b) {
        sum += func(a + h/2);
        a += h;
    }
    return sum * h;
}

double trapeze(double a, double b, double h) {
    double sum = 0;
    while (a + h < b) {
        sum += (func(a + h) + func(a));
        a += h;
    }
    return sum * h * 0.5;
}

double simpson(double a, double b, double h) {
    double sum = 0;
    a += h;
    while (a + h < b) {
        sum += func(a - h) + 4 * func(a - h/2) + func(a);
        a += h;
    }
    return sum * h / 6;
}

double rungeRombert(double h1, double h2, double i1, double i2, double p) {
    return i1 + (i1 - i2) / (pow((h2 / h1), p) - 1);
}

int main() {
    double a = 0;
    double b = 2;
    double h1 = 0.5;
    double h2 = 0.25;

    std::cout << "===========================\n";
    std::cout << "h = " << h1 << "\n";
    std::cout << "Integral rect: " << rect(a, b, h1) << "\n";
    std::cout << "Integral trapeze: " << trapeze(a, b, h1) << "\n";
    std::cout << "Integral simpson: " << simpson(a, b, h1) << "\n";

    std::cout << "===========================\n";
    std::cout << "h = " << h2 << "\n";
    std::cout << "Integral rect: " << rect(a, b, h2) << "\n";
    std::cout << "Integral trapeze: " << trapeze(a, b, h2) << "\n";
    std::cout << "Integral simpson: " << simpson(a, b, h2) << "\n";

    std::cout << "===========================\n";
    std::cout << "Runge Rombert:\n";
    std::cout << "Rombert rect: " << rungeRombert(h1, h2, rect(a, b, h1), rect(a, b, h2), 1) << "\n";
    std::cout << "Rombert trapeze: " << rungeRombert(h1, h2, trapeze(a, b, h1), trapeze(a, b, h2), 2) << "\n";
    std::cout << "Rombert simpson: " << rungeRombert(h1, h2, simpson(a, b, h1), simpson(a, b, h2), 4) << "\n";

    return 0;
}
