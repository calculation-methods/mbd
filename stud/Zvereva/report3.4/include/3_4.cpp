#include <iostream>
#include <vector>

double df(std::vector<double> x, std::vector<double> y, double X) {
    int pos = 0;
    double h = x[1] - x[0];
    for (int i = 0; i < x.size() - 1; ++i)
        if (X > x[i] && X < x[i + 1]) {
            if (i == 0) pos = i;
            else if (i + 2 > x.size() - 1)
                pos = i - (i + 2 - (x.size() - 1));
            else pos = i - 1;
        }

    double a1 = (y[pos + 1] - y[pos]) / h;
    double a2 = ((y[pos + 2] - 2 * y[pos + 1] + y[pos]) / (2 * h * h)) * (2 * X - x[pos] - x[pos + 1]);
    return a1 + a2;
}

double df2(std::vector<double> x, std::vector<double> y, double X) {
    int pos = 0;
    double h = x[1] - x[0];
    for (int i = 0; i < x.size() - 1; ++i)
        if (X > x[i] && X < x[i + 1]) {
            if (i == 0) pos = i;
            else if (i + 2 > x.size() - 1)
                pos = i - (i + 2 - (x.size() - 1));
            else pos = i - 1;
        }

    return (y[pos + 2] - 2 * y[pos + 1] + y[pos]) / (h * h);
}

int main() {
    std::vector<double> x = { 0.0,    0.5,     1.0,    1.5,       2.0 };
    std::vector<double> y = { 0.0,    0.97943,    1.8415,    2.4975,    2.9093 };
    double X = 1.0;
    std::cout << "First derivative in X: " << df(x, y, X) << "\n";
    std::cout << "Second derivative in X: " << df2(x, y, X) << "\n";

    return 0;
}
