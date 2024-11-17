#include <iostream>
#include <cmath>

using namespace std;

double Func(double x)
{
    return x / pow((3 * x + 4), 3);
}

double Rectangle(double X0, double Xk, double h)
{
    double sum = 0;
    while (X0 + h < Xk)
    {
        sum += Func(X0 + h / 2);
        X0 += h;
    }

    return sum * h;
}

double Trapeze(double X0, double Xk, double h)
{
    double sum = 0;

    while (X0 + h < Xk)
    {
        sum += (Func(X0 + h) + Func(X0));
        X0 += h;
    }

    return sum * h * 0.5;
}

double Simpson(double X0, double Xk, double h)
{
    double sum = 0;
    X0 += h;
    while (X0 + h < Xk)
    {
        sum += Func(X0 - h) + 4 * Func(X0 - h / 2) + Func(X0);
        X0 += h;
    }

    return sum * h / 6;
}

double rungeRombert(double h1, double h2, double i1, double i2, double p)
{
    return i1 + (i1 - i2) / (pow((h2 / h1), p) - 1);
}

int main() {
    double X0 = -1;
    double Xk = 1;
    double h1 = 0.5;
    double h2 = 0.25;

    cout << "__________H1_________" << endl;
    cout << "Rectangle: " << Rectangle(X0, Xk, h1) << endl;
    cout << "Trapeze: " << Trapeze(X0, Xk, h1) << endl;
    cout << "Simpson: " << Simpson(X0, Xk, h1) << endl << endl;

    cout << "__________H2_________" << endl;
    cout << "Rectangle: " << Rectangle(X0, Xk, h2) << endl;
    cout << "Trapeze: " << Trapeze(X0, Xk, h2) << endl;
    cout << "Simpson: " << Simpson(X0, Xk, h2) << endl << endl;

    cout << "________Error________" << endl;
    cout << "Rect: " << rungeRombert(h1, h2, Rectangle(X0, Xk, h1), Rectangle(X0, Xk, h2), 1) << endl;
    cout << "Trapeze: " << rungeRombert(h1, h2, Trapeze(X0, Xk, h1), Trapeze(X0, Xk, h2), 2) << endl;
    cout << "Simpson: " << rungeRombert(h1, h2, Simpson(X0, Xk, h1), Simpson(X0, Xk, h2), 4) << endl;

    return 0;
}