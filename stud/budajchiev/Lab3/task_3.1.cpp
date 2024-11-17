#include <iostream>
#include <vector>
#include <utility>
#include <cmath>

using namespace std;

double F(double x)
{
    return tan(x);
}

double Lagrange(vector<double> X, vector<double> Y, double spec) 
{

    double sum = 0;
   
    for (int i = 0; i < X.size(); ++i)
    {
        double mult = 1;

        for (int j = 0; j < X.size(); ++j)
        {
            if (i == j)
                continue;

            mult = mult * ((spec - X[j]) / (X[i] - X[j]));
        }

        sum = sum + mult * Y[i];
    }

    return sum;
}

double Diff(vector<double> X,vector<double> Y, int n)
{
    switch (n)
    {
        case 0:
            return Y[0];
            break;

        case 1:
            return (Y[0] - Y[1]) / (X[0] - X[1]);
            break;

        default:
            pair<vector<double>, vector<double>> X_v;
            X_v.first.resize(n);
            X_v.second.resize(n);

            pair<vector<double>, vector<double>> Y_v;
            Y_v.first.resize(n);
            Y_v.second.resize(n);

            for (int i = 0; i < n; ++i)
            {
                X_v.first[i]    = X[i];
                X_v.second[i]   = X[i + 1];
                Y_v.first[i]    = Y[i];
                Y_v.second[i]   = Y[i + 1];
            }

            return (Diff(X_v.first, Y_v.first, n - 1) - Diff(X_v.second, Y_v.second, n - 1)) / (X[0] - X[n]);
            break;
    }
}

double Newton(vector<double> X, vector<double> Y, double spec) 
{
    vector<double> X_n(X.size());
    vector<double> Y_n(Y.size());

    double sum = 0;

    for (int i = 0; i < X.size(); i++)
    {
        X_n[i] = X[i];
        Y_n[i] = Y[i];
    }
    
    for (int i = 0; i < X_n.size(); ++i)
    {
        double mult = Diff(X_n, Y_n, i);
        
        for (int j = 0; j < i; ++j)
        {
            mult = mult * (spec - X[j]);
        }

        sum = sum + mult;
    }

    return sum;
}


int main()
{
    vector<double> X_a{ 0, M_PI / 8, M_PI / 4, 3 * M_PI / 8 };
    vector<double> X_b{ 0, M_PI / 8, M_PI / 3, 3 * M_PI / 8 };

    vector<double> Y_a;
    vector<double> Y_b;

    for (int i = 0; i < X_a.size(); i++)
    {
        Y_a.push_back(F(X_a[i]));
        Y_b.push_back(F(X_b[i]));
    }

    double spec = 3 * M_PI / 16;

    cout << "____________________________Lagrange Method______________________________" << endl;

    double A = Lagrange(X_a, Y_a, spec);
    cout << "A | Ans: " << A << "\t\tF(X*): " << F(spec) << "\t\tDiff: " << abs(F(spec) - A) << endl;
    double B = Lagrange(X_b, Y_b, spec);
    cout << "B | Ans: " << B << "\t\tF(X*): " << F(spec) << "\t\tDiff: " << abs(F(spec) - B) << endl;

    cout << "\n_____________________________Newton Method_______________________________" << endl;

    A = Newton(X_a, Y_a, spec);
    cout << "A | Ans: " << A << "\t\tF(X*): " << F(spec) << "\t\tDiff: " << abs(F(spec) - A) << endl;
    B = Newton(X_b, Y_b, spec);
    cout << "B | Ans: " << B << "\t\tF(X*): " << F(spec) << "\t\tDiff: " << abs(F(spec) - B) << endl;
}