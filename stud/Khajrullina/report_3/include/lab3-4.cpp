#include <iostream>
#include <vector>

double derivative(std::vector<double> x, std::vector<double> y, double X) {
    int n = 0;
    double h = x[1] - x[0];
    for (int i = 0; i < x.size() - 1; ++i)
        if (X > x[i] && X < x[i + 1]) {
            if (i == 0) n = i;
            else if (i + 2 > x.size() - 1)
                n = i - (i + 2 - (x.size() - 1));
            else n = i - 1;
        }
    double a1 = (y[n + 1] - y[n]) / h;
    double a2 = ((y[n + 2] - 2 * y[n + 1] + y[n]) / (2 * h * h)) * (2 * X - x[n] - x[n + 1]);
    return a1 + a2;
}

double derivative2(std::vector<double> x, std::vector<double> y, double X) {
    int n = 0;
    double h = x[1] - x[0];
    for (int i = 0; i < x.size() - 1; ++i)
        if (X > x[i] && X < x[i + 1]) {
            if (i == 0) n = i;
            else if (i + 2 > x.size() - 1)
                n = i - (i + 2 - (x.size() - 1));
            else n = i - 1;
        }

    return (y[n + 2] - 2 * y[n + 1] + y[n]) / (h * h);
}

int main() {
    std::vector<double> x = { 0.0,    1.0,     2.0,    3.0,       4.0 };
    std::vector<double> y = { 0.0,    0.86603,    1.0,    0.0,    -2.0 };
    double X = 2.0;
    std::cout << "First derivative in X: " << derivative(x, y, X) << "\n";
    std::cout << "Second derivative in X: " << derivative2(x, y, X) << "\n";

    return 0;
}