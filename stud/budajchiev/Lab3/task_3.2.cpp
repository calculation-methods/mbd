#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void FUNC(vector<double> X, vector<double> Y, double spec)
{
    vector<double> H(4);
    for (int i = 0; i < 4; ++i)
    {
        H[i] = X[i + 1] - X[i];
    }

    vector<vector<double>> A(3);
    for (int i = 0; i < A.size(); i++)
    {
        A[i].resize(3);
    }

    A[0][0] = 0;
    A[0][1] = 2 * (H[0] + H[1]);
    A[0][2] = H[1];

    vector<double> B(4);
    B[0] = 3 * ((Y[2] - Y[1]) / H[1] - (Y[1] - Y[0]) / H[0]);

    for (int i = 1; i < 2; ++i)
    {
        A[i][0] = H[i];
        A[i][1] = 2 * (H[i] + H[i + 1]);
        A[i][2] = H[i + 1];
        B[i] = 3 * ((Y[i + 2] - Y[i + 1]) / H[i + 1] - (Y[i + 1] - Y[i]) / H[i]);
    }

    A[2][0] = H[2];
    A[2][1] = 2 * (H[2] + H[3]);
    A[2][2] = 0;
    B[2] = 3 * ((Y[4] - Y[3]) / H[3] - (Y[3] - Y[2]) / H[2]);

    vector<double> P(3), C(4), Q(3);

    Q[0] = B[0] / A[0][1];
    P[0] = -A[0][2] / A[0][1];

    for (int i = 1; i < 3; ++i)
    {
        P[i] = -A[i][2] / (A[i][1] + A[i][0] * P[i - 1]);
        Q[i] = (B[i] - A[i][0] * Q[i - 1]) / (A[i][1] + A[i][0] * P[i - 1]);
    }

    C[0] = 0;
    C[3] = Q[2];
    for (int i = 2; i > 0; --i)
    {
        C[i] = P[i - 1] * C[i + 1] + Q[i - 1];
    }

    vector<double> a(4), b(4), d(4);
    for (int i = 0; i < 3; ++i)
    {
        a[i] = Y[i];
        b[i] = (Y[i + 1] - Y[i]) / H[i] - H[i] * (C[i + 1] + 2 * C[i]) / 3;
        d[i] = (C[i + 1] - C[i]) / 3 / H[i];
    }

    a[3] = Y[3];
    b[3] = (Y[4] - Y[3]) / H[3] - 2 / 3 * H[3] * C[3];
    d[3] = -C[3] / 3 / H[3];

    int i = 0;

    while (X[i] < spec && X[i + 1] < spec)
    {
        i += 1;
    }
    
    cout << "___FUNC inc.___" << endl << endl << "F(X*): " << a[i] + b[i] * (spec - X[i]) + C[i] * pow(spec - X[i], 2) + d[i] * pow(spec - X[i], 3) << endl << "_______________";
}

int main()
{
    vector<double> X{ 0.0, 0.9, 1.8, 2.7, 3.6 };
    vector<double> Y{ 0.0, 0.36892, 0.85408, 1.7856, 6.3138 };

    double spec = 1.5;
    FUNC(X, Y, spec);
}