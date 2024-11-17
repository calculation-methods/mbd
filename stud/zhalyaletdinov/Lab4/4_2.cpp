#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include  <functional>
using namespace  std;

std::vector<double> triade(const vector<vector<double>>& matrix, const std::vector<double>& b) {
    size_t n = matrix.size();

    std::vector<double> C(n, 0);
    std::vector<double> D(n, 0);
    std::vector<double> x(n);

    C[0] = matrix[0][1] / matrix[0][0];
    D[0] = b[0] / matrix[0][0];

    for (size_t i = 1; i < n; ++i) {
        double m = 1 / (matrix[i][i] - matrix[i][i - 1] * C[i - 1]);
        C[i] = i < n - 1 ? matrix[i][i + 1] * m : 0;
        D[i] = (b[i] - matrix[i][i - 1] * D[i - 1]) * m;
    }

    x[n - 1] = D[n - 1];

    for (int i = n - 2; i >= 0; --i) {
        x[i] = D[i] - C[i] * x[i + 1];
    }

    return x;
}

double F(double x, double y1, double y2)
{
    return  (4 * y1 - 4 * x * y2) / (2 * x + 1);
}

double Fprec(double x)
{
    return 3 * x + pow(M_E, (-2) * x);
}

vector<vector<double>> Runge(double x0, double y1_0, double y2_0, double h, int n)
{
    vector<double> x(n);
    x[0] = x0;

    vector<double> y1(n);
    y1[0] = y1_0;

    vector<double> y2(n);
    y2[0] = y2_0;


    for (int i = 1; i < n; ++i)
    {
        x[i] = x[i - 1] + h;

        double k1_y1 = h * y2[i - 1];
        double k1_y2 = h * F(x[i - 1], y1[i - 1], y2[i - 1]);

        double k2_y1 = h * (y2[i - 1] + 0.5 * k1_y2);
        double k2_y2 = h * F(x[i - 1] + 0.5 * h, y1[i - 1] + 0.5 * k1_y1, y2[i - 1] + 0.5 * k1_y2);

        double k3_y1 = h * (y2[i - 1] + 0.5 * k2_y2);
        double k3_y2 = h * F(x[i - 1] + 0.5 * h, y1[i - 1] + 0.5 * k2_y1, y2[i - 1] + 0.5 * k2_y2);

        double k4_y1 = h * (y2[i - 1] + k3_y2);
        double k4_y2 = h * F(x[i - 1] + h, y1[i - 1] + k3_y1, y2[i - 1] + k3_y2);

        y1[i] = y1[i - 1] + (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1) / 6;
        y2[i] = y2[i - 1] + (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 6;
    }

    return vector<vector<double>>{x, y1, y2};
}

void Romberg(const function<vector<vector<double>>(double, double, double, double, int)> &Func, double x0, double y1_0, double y2_0, double h, int n)
{
    vector<vector<double>> dat1 = Func(x0, y1_0, y2_0, h, n);
    vector<vector<double>> dat2 = Func(x0, y1_0, y2_0, h / 2, 2 * n);

    vector<double> y1_h, y2_h, y1_h2, y2_h2;

    y1_h = dat1[1];
    y2_h = dat1[2];

    y1_h2 = dat2[1];
    y2_h2 = dat2[2];

    cout << "X\tY err\t\tY' err" << endl;
    for (int i = 0; i < n; ++i)
    {
        double x_i = x0 + i * h;
        double error_y1 = (y1_h2[2 * i] - y1_h[i]) / (pow(2, 4) - 1);
        double error_y2 = (y2_h2[2 * i] - y2_h[i]) / (pow(2, 4) - 1);
        cout <<  x_i << "\t" << error_y1 << "\t" << error_y2 << endl;
    }
}

void BangBang(const function<double(double, double, double)>& f, double a, double b, double alpha, double beta, double eps, double h, int n)
{
    vector<vector<double>> result;
    vector<double> y_trial(n);
    vector<double> s_values;
    double s0 = 0;
    double s1 = 1.0;
    double y_b0, y_b1, s_new;

    auto result0 = Runge(a, alpha, s0, h, n);
    y_b0 = result0[1].back();
    auto result1 = Runge(a, alpha, s1, h, n);
    y_b1 = result1[1].back();


    cout << "s\t\tf(b, y, s)\tP(s)" << endl;

    while (abs(y_b1 - beta) > eps)
    {
        s_new = s1 - (y_b1 - beta) * (s1 - s0) / (y_b1 - y_b0);
        s_values.push_back(s_new);
        cout << s1 << " \t\t" << y_b1 << "\t\t" << abs(y_b1 - beta) << endl;
        s0 = s1;
        y_b0 = y_b1;
        s1 = s_new;
        result = Runge(a, alpha, s1, h, n);
        y_b1 = result[1].back();
    }
    cout << s1 << "\t" << y_b1 << "\t" << abs(y_b1 - beta) << endl;

    cout << "\nx\ty" << endl;

    for (int i = 0; i < n; ++i)
    {
        cout << result[0][i] << "\t" << result[1][i] << endl;
    }
}


double p(double x)
{
    return 0;
}

double q(double x)
{
    return (-4 + 4 * x) / (2 * x + 1);
}

double f(double x)
{
    return 0.0;
};


void EndDiff(double a, double b, double y0, double y1, double h, vector<double>& x, vector<double>& y)
{
    int n = static_cast<int>((b - a) / h) + 1;

    x.resize(n);

    y.resize(n);

    vector<double> rhs(n);

    for (int i = 0; i < n; ++i) {
        x[i] = a + i * h;
    }

    rhs[0] = h * h * f(x[0]) - (1 - p(x[0]) * h / 2) * y0;

    rhs[n - 1] = h * h * f(x[n - 1]) - (1 + p(x[n - 1]) * h / 2) * y1;

    for (int i = 1; i < n - 1; ++i)
    {
        rhs[i] = h * h * f(x[i]);
    }

    vector<vector<double>> A(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        A[i][i] = -2 + h * h * q(x[i]);
        if (i > 0) A[i][i - 1] = 1 - p(x[i]) * h / 2;
        if (i < n - 1) A[i][i + 1] = (1 + p(x[i]) * h / 2);
    }
    y = triade(A, rhs);
    cout << "x\ty" << endl;
    cout << a << "\t" << y0 << endl;

    for (int i = 0; i < n; ++i)
    {
        cout << x[i] << "\t" << y[i] << endl;
    }
}

void PrecF(double a, double b, double h)
{
    vector<double> x, y;

    for (double xi = a; xi <= b; xi += h)
    {
        x.push_back(xi);
        y.push_back(Fprec(xi));
    }

    cout << "x\ty" << endl;

    for (size_t i = 0; i < x.size(); ++i)
    {
        cout << x[i] << "\t" << y[i] << endl;
    }
}

double RungeEr(const vector<double>& y_h, const vector<double>& y_h2, double r) {
    double error = 0.0;
    for (size_t i = 0; i < y_h.size(); ++i) {
        error = max(error, abs(y_h2[2 * i] - y_h[i]) / (pow(2, r) - 1));
    }
    return error;
}

int main()
{
    PrecF(-2, 0, 0.2);

    double x1 = -2, x2 = 0, y1 = 48.5982, y2 = 1;
    double e = 0.00001;
    double h = 0.2;


    cout << "\nShooting:" << "\n";
    BangBang(F, x1, x2, y1, y2, e, h, 6);


    cout << "\nErrorr:" << "\n";
    Romberg(Runge, x1, y1, 0, h, 6);

    vector<double> x_h, y_h, x_h2, y_h2;

    cout << "\nEndDiff:" << "\n";
    EndDiff(x1, x2, y1, y2, h, x_h, y_h);

    cout << "\nEndDiff h/2:" << "\n";
    EndDiff(x1, x2, y1, y2, h / 2, x_h2, y_h2);

    double error = RungeEr(y_h, y_h2, 2);

    cout << "\nError:" << endl;

    cout << "x\ty" << endl;

    double m = 0.0;
    for (size_t i = 0; i < x_h.size(); ++i) 
    {
        double ex = Fprec(x_h[i]);
        double err = abs(y_h[i] - ex);
        m = max(m, err);
        cout << x_h[i] << "\t" << err << endl;
    }

    return 0;
}