#include <iostream>
#include <vector>

using namespace std;

class CubicSpline {
private:
    vector<double> x, y;
    vector<double> h, alpha, l, mu, z, c, b, d;

public:
    CubicSpline(const vector<double> &x, const vector<double> &y) : x(x), y(y) {
        int n = x.size();
        h.resize(n);
        alpha.resize(n);
        l.resize(n);
        mu.resize(n);
        z.resize(n);
        c.resize(n);
        b.resize(n);
        d.resize(n);

        for (int i = 0; i < n - 1; ++i) {
            h[i] = x[i + 1] - x[i];
        }

        for (int i = 1; i < n - 1; ++i) {
            alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
        }

        l[0] = 1;
        mu[0] = 0;
        z[0] = 0;

        for (int i = 1; i < n - 1; ++i) {
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n - 1] = 1;
        z[n - 1] = 0;
        c[n - 1] = 0;

        for (int j = n - 2; j >= 0; --j) {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
            d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        }
    }

    double interpolate(double x_star) {
        int n = x.size();
        int j = 0;
        while (j < n && x[j] < x_star)
            j++;
        if (j >= n)
            j = n - 1;
        double dx = x_star - x[j - 1];
        return y[j - 1] + b[j - 1] * dx + c[j - 1] * dx * dx + d[j - 1] * dx * dx * dx;
    }
};

int main() {
    vector<double> x = {0.0, 0.9, 1.8, 2.7, 3.6};
    vector<double> y = {0.0, 0.72235, 1.5609, 2.8459, 7.7275};

    CubicSpline spline(x, y);

    double x_star = 1.5;
    double interpolated_value = spline.interpolate(x_star);

    cout << "f(" << x_star << ") = " << interpolated_value << endl;

    return 0;
}
