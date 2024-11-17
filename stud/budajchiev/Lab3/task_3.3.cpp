#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

pair<vector<vector<double>>, vector<vector<double>>> createLU(vector<vector<double>> A)
{
    vector<vector<double>> U = A;

    vector<vector<double>> L(A.size());

    for (auto& elem : L)
    {
        for (int i = 0; i < A.size(); i++)
        {
            elem.push_back(0.0);
        }        
    }

    for (int k = 0; k < U.size(); ++k)
    {
        for (int i = k; i < U.size(); ++i)
        {
            L[i][k] = U[i][k] / U[k][k];
        }

        for (int i = k + 1; i < U.size(); ++i)
        {
            for (int j = k; j < U.size(); ++j)
            {
                U[i][j] = U[i][j] - L[i][k] * U[k][j];
            }
        }
    }

    return make_pair(L, U);
}

vector<double> solve(vector<vector<double>> A, vector<double> B)
{    
    pair<vector<vector<double>>, vector<vector<double>>> LU = createLU(A);

    vector<vector<double>> L = LU.first;
    vector<vector<double>> U = LU.second;

    vector<double> Z(A.size());

    Z[0] = B[0];

    for (int i = 1; i < A.size(); ++i)
    {
        double sum = 0;

        for (int j = 0; j < i; ++j)
        {
            sum += L[i][j] * Z[j];
        }

        Z[i] = B[i] - sum;
    }

    vector<double> x(A.size());

    x[A.size() - 1] = Z[A.size() - 1] / U[A.size() - 1][A.size() - 1];

    for (int i = A.size() - 2; i >= 0; --i)
    {
        double sum = 0;

        for (int j = i + 1; j < A.size(); ++j)
        {
            sum += U[i][j] * x[j];
        }

        x[i] = (Z[i] - sum) / U[i][i];
    }

    return x;
}

vector<double> FUNC(vector<double> X, vector<double> Y, int m)
{
    vector<vector<double>> A(m + 1, vector<double>(m + 1));

    vector<double> b(m + 1);

    for (int k = 0; k < m + 1; ++k)
    {
        for (int j = 0; j < m + 1; ++j)
        {
            double sum = 0;

            for (int i = 0; i < X.size(); ++i)
            {
                sum += pow(X[i], k + j);
            }

            A[k][j] = sum;
        }

        double sum = 0;

        for (int i = 0; i < X.size(); ++i)
        {
            sum += Y[i] * pow(X[i], k);
        }

        b[k] = sum;
    }

    vector<double> a = solve(A, b);

    return a;
}

double error(vector<double> X, vector<double> Y, vector<double> A)
{
    double s = 0;

    for (int i = 0; i < A.size(); ++i)
    {
        s = s + (A[i] - Y[i+1]) * (A[i] - Y[i+1]);
    }

    return s;
}


int main()
{
    vector<double> X{ -0.9, 0.0, 0.9, 1.8, 2.7, 3.6 };
    vector<double> Y{ -0.36892, 0.0, 0.36892, 0.85408, 1.7856, 6.3138 };

    vector<double> A = FUNC(X, Y, 1);

    cout << "____________________________FUNC inc.____________________________" << endl << "A | Coeff:";

    for(auto& elem : A)
    {
        cout << " " << elem << " ";
    }

    cout << "\t\t\t| Error: " << error(X, Y, A) << endl;

    A = FUNC(X, Y, 2);

    cout << "B | Coeff:";

    for (auto& elem : A)
    {
        cout << " " << elem << " ";
    }

    cout << "\t| Error: " << error(X, Y, A) << endl;

    return 0;
}