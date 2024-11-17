#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
using namespace  std;

double F(double x, double y1, double y2)
{
    return 4 * x * y2 - y1 * (4 * x * x - 3) + pow(M_E, pow(x, 2));
}

double Fprec(double x)
{
    return (pow(M_E, x) + pow(M_E, -x) - 1) * pow(M_E, pow(x, 2));
}

vector<vector<double>> Euler(double x0, double y1_0, double y2_0, double h, int n)
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
        y1[i] = y1[i - 1] + h * y2[i - 1];
        y2[i] = y2[i - 1] + h * F(x[i - 1], y1[i - 1], y2[i - 1]);
    }

    return vector<vector<double>>{x, y1, y2};
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

vector<vector<double>> Adam(double x0, double y1_0, double y2_0, double h, int n)
{
    vector<double> x(n);
    x[0] = x0;

    vector<double> y1(n);
    y1[0] = y1_0;

    vector<double> y2(n);
    y2[0] = y2_0;

    for (int i = 1; i < 4; ++i)
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

    for (int i = 4; i < n; ++i)
    {
        x[i] = x[i - 1] + h;

        y1[i] = y1[i - 1] + h / 24 * (55 * y2[i - 1] - 59 * y2[i - 2] + 37 * y2[i - 3] - 9 * y2[i - 4]);
        y2[i] = y2[i - 1] + h / 24 * (55 * F(x[i - 1], y1[i - 1], y2[i - 1]) - 59 * F(x[i - 2], y1[i - 2], y2[i - 2]) + 37 * F(x[i - 3], y1[i - 3], y2[i - 3]) - 9 * F(x[i - 4], y1[i - 4], y2[i - 4]));
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

    cout << "X\tY err\tY' err" << endl;
    for (int i = 0; i < n; ++i)
    {
        double x_i = x0 + i * h;
        double error_y1 = (y1_h2[2 * i] - y1_h[i]) / (pow(2, 4) - 1);
        double error_y2 = (y2_h2[2 * i] - y2_h[i]) / (pow(2, 4) - 1);
        cout <<  x_i << "\t" << error_y1 << "\t" << error_y2 << endl;
    }
}

void Error(vector<vector<double>>& data)
{
    cout << "X \t Y \t Yprec \t Error" << endl;

    for (size_t i = 0; i < data[0].size(); ++i)
    {
        double Yprec = Fprec(data[0][i]);
        double err = abs(data[1][i] - Yprec);
        cout << data[0][i] << " \t " << data[1][i] << " \t " << Yprec << " \t" << err << endl;
    }
}

int main() {
    double x = 0;
    double y1 = 1;
    double y2 = 0;
    double h = 0.1;

    int n = 11;

    vector<vector<double>> euler = Euler(x, y1, y2, h, n);
    vector<vector<double>> runge = Runge(x, y1, y2, h, n);
    vector<vector<double>> adam = Adam(x, y1, y2, h, n);

    cout << "\nEuler:" << "\n";
    Error(euler);

    cout << "\nEuler error by Romberg:" << "\n";
    Romberg(Euler, x, y1, y2, h, n);

    cout << "\nRunge:" << "\n";
    Error(runge);

    cout << "\nRunge error by Romberg:" << "\n";
    Romberg(Runge, x, y1, y2, h, n);

    cout << "\nAdam:" << "\n";
    Error(adam);

    cout << "\nAdams error by Romberg:" << "\n";
    Romberg(Adam, x, y1, y2, h, n);

    return 0;
}