#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

bool leq(double a, double b)
{
    return (a < b) || (abs(b - a) < 1e-9);
}

double Der1(vector<double> X, vector<double> Y, double spec)
{
    for (int i = 0; i < X.size() - 1; ++i)
    {
        if (X[i] < spec && leq(spec, X[i + 1]))
        {
            double ANS = (Y[i + 1] - Y[i - 1]) / (X[i + 1] - X[i - 1]);
            return ANS;
        }
    }
    return 0;
}

double Der2(vector<double> X, vector<double> Y, double spec)
{
    for (int i = 0; i < X.size() - 1; ++i)
    {
        if (X[i] < spec && leq(spec, X[i + 1]))
        {
            double ANS = (Y[i - 1] - 2 * Y[i] + Y[i + 1]) / pow((X[i + 1] - X[i]), 2);
            return ANS;
        }
    }
    return 0;
}


int main()
{
    vector<double> X{ 1.0, 1.5, 2.0, 2.5, 3.0 };
    vector<double> Y{ 0.0, 0.40547,0.69315,0.91629,1.0986 };
    double spec = 2.0;

    cout << "First derivative in X*(" << spec << "),\tf'(X*) = " << Der1(X, Y, spec) << endl;
    cout << "Second derivative in X*(" << spec << "),\tf\"(X*) = " << Der2(X, Y, spec) << endl;
}